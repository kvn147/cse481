import json
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from numpy import array, matmul

import tf2_ros
from tf2_ros import TransformException, Buffer, TransformListener
from tf_transformations import euler_from_quaternion, quaternion_matrix
from std_srvs.srv import Trigger

import sys
import time
import threading
from math import atan2, sqrt
import numpy as np
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.time import Time
from trajectory_msgs.msg import JointTrajectoryPoint
from action_msgs.msg import GoalStatus
from stretch_nav2.robot_navigator import BasicNavigator, TaskResult

from scipy.spatial.transform import Rotation as R

CAN_START_POSE_FILE = "/home/hello-robot/kevin/cse481/final_project/aruco_data/trash_start.json"
CAN_PICKUP_POSE_FILE = "/home/hello-robot/kevin/cse481/final_project/joint_state_data/trash_pickup.json"
RECEPTACLE_START_POSE_FILE = "/home/hello-robot/kevin/cse481/final_project/aruco_data/receptacle_start.json"

TRASH_CAN_OFFSET_ORIENTATION = np.pi
RECEPTACLE_OFFSET_ORIENTATION = np.pi / 2

RECEPTACLE_WAYPOINTS = [
    [3.38, -0.75, 0.0, 1.0],
    [4.19, -1.5, 0.0, 1.0],
    [5.21, -2.3438, 0.0, 1.0],
    [5.246, -4.31, 0.259, 0.966]
]

HEAD_PAN_SEARCH = -1 * np.pi / 2
HEAD_PAN_NEUTRAL = 0.0
MINIMUM_ANGLE_THRESHOLD = 0.03
MAX_TF_AGE = 1.0
RECENT_TF_TIMEOUT = 5.0
RECENT_TF_POLL_TIME = 0.1
TRASH_CAN_MARKER_OFFSET_X = 0.17


class WasteDisposal(Node):
    def __init__(self):
        super().__init__('waste_disposal')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/stretch_controller/follow_joint_trajectory",
        )

        self._goal_handle = None
        self._goal_handle_lock = threading.Lock()
        self._abort_requested = False
        self._is_paused = False
        self._sequence_running = False

        if not self.trajectory_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("Unable to connect to trajectory server.")

        self.navigator = BasicNavigator()

        self.current_tilt = 0.0
        self.current_pan = 0.0
        self.joint_state_sub = self.create_subscription(
            JointState, "/stretch/joint_states", self.joint_states_callback, 10
        )

        self.subscription = self.create_subscription(
            String, 'task_execution', self.task_callback, 10
        )
        self._status_pub = self.create_publisher(String, 'task_status', 10)

        self.get_logger().info("Waiting for Nav2 to become active...")
        self.navigator.waitUntilNav2Active()
        self.get_logger().info("Waste Disposal node started and listening to /task_execution.")

    # ------------------------------------------------------------------ #
    #  Callbacks                                                           #
    # ------------------------------------------------------------------ #

    def joint_states_callback(self, msg):
        for name, pos in zip(msg.name, msg.position):
            if name == "joint_head_tilt":
                self.current_tilt = pos
            elif name == "joint_head_pan":
                self.current_pan = pos

    def task_callback(self, msg):
        task_type = msg.data.strip().lower()
        self.get_logger().info(f"Received task: {task_type}")

        # Handle "resume:extraction" / "resume:navigation" / "resume:disposal"
        if task_type.startswith("resume:"):
            phase = task_type.split(":", 1)[1]
            threading.Thread(
                target=self._execute_resume_from_phase, args=(phase,), daemon=True
            ).start()
            return

        handler = getattr(self, f"execute_{task_type}", None)
        if not handler:
            self.get_logger().error(f"Unknown task type: {task_type}")
            return

        # For anything that isn't pause/resume/stop, clear abort state.
        if task_type not in ["stop", "pause", "resume"]:
            self._abort_requested = False
            self._is_paused = False

        threading.Thread(target=handler, daemon=True).start()

    # ------------------------------------------------------------------ #
    #  Teleop                                                              #
    # ------------------------------------------------------------------ #

    def execute_base_forward(self):
        self.send_base_goal_blocking([("translate_mobile_base", 0.1)], duration=2.0, bypass=True)

    def execute_base_backward(self):
        self.send_base_goal_blocking([("translate_mobile_base", -0.1)], duration=2.0, bypass=True)

    def execute_base_left(self):
        self.send_base_goal_blocking([("rotate_mobile_base", 0.2)], duration=2.0, bypass=True)

    def execute_base_right(self):
        self.send_base_goal_blocking([("rotate_mobile_base", -0.2)], duration=2.0, bypass=True)

    def execute_camera_up(self):
        self.send_base_goal_blocking([("joint_head_tilt", self.current_tilt + 0.2)], duration=0.5, bypass=True)

    def execute_camera_down(self):
        self.send_base_goal_blocking([("joint_head_tilt", self.current_tilt - 0.2)], duration=0.5, bypass=True)

    def execute_camera_left(self):
        self.send_base_goal_blocking([("joint_head_pan", self.current_pan + 0.2)], duration=0.5, bypass=True)

    def execute_camera_right(self):
        self.send_base_goal_blocking([("joint_head_pan", self.current_pan - 0.2)], duration=0.5, bypass=True)

    # ------------------------------------------------------------------ #
    #  Pause / Resume / Stop                                               #
    # ------------------------------------------------------------------ #

    def _cancel_active_trajectory(self):
        """Cancel any in-flight trajectory goal. Safe to call from any thread."""
        with self._goal_handle_lock:
            gh = self._goal_handle
        if gh is not None:
            self.get_logger().info("Cancelling active trajectory.")
            gh.cancel_goal_async()

    def execute_pause(self):
        """
        Stop everything immediately, cancel Nav2 if running, then reset to
        neutral pose. Resume will restart the full sequence from scratch.
        """
        self.get_logger().info("Pause requested — stopping and resetting.")
        self._is_paused = True
        self._abort_requested = True  # causes send_base_goal_blocking to exit fast

        self._cancel_active_trajectory()

        try:
            self.navigator.cancelTask()
        except Exception:
            pass

        # Wait briefly for the sequence thread to notice the abort flag and exit.
        for _ in range(20):
            if not self._sequence_running:
                break
            time.sleep(0.1)

        self.switch_mode("position")
        # Clear abort so reset can run, but keep _is_paused=True.
        self._abort_requested = False
        self.get_logger().info("Returning to neutral (reset) position after pause.")
        self.execute_reset()
        self.get_logger().info("Paused and reset. Press Resume to restart the sequence.")

    def _execute_resume_from_phase(self, phase: str):
        valid = ("extraction", "navigation", "disposal")
        if phase not in valid:
            self.get_logger().error(f"Unknown resume phase '{phase}'. Must be one of {valid}.")
            return
        if self._sequence_running:
            self.get_logger().warn("Sequence thread still running; resume ignored.")
            return
        self.get_logger().info(f"Resuming sequence from phase: {phase}")
        self._is_paused = False
        self._abort_requested = False
        threading.Thread(target=self._run_sequence_from, args=(phase,), daemon=True).start()

    def _run_sequence_from(self, start_phase: str):
        """Run the sequence starting at start_phase, skipping earlier phases."""
        self._sequence_running = True
        try:
            self.get_logger().info(f"Sequence starting from: {start_phase}")
            phases = ["extraction", "navigation", "disposal"]
            idx = phases.index(start_phase)

            if idx <= 0 and not self._is_paused:
                self.execute_extraction()
                if self._abort_requested:
                    return

            if idx <= 1 and not self._is_paused:
                status = self.execute_go_to_receptacle()
                if status != "SUCCESS":
                    return

            # idx <= 2 always true, but explicit for clarity
            if not self._is_paused:
                self.execute_disposal()
            if self._abort_requested:
                return

            self.get_logger().info("Sequence completed successfully.")
            self._status_pub.publish(String(data="Sequence complete!"))
        finally:
            self._sequence_running = False

    def execute_stop(self):
        """Hard stop — cancels everything and does NOT reset or resume."""
        self.get_logger().warn("Stop requested! Halting immediately.")
        self._abort_requested = True
        self._is_paused = False
        self._cancel_active_trajectory()
        try:
            self.navigator.cancelTask()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    #  Core trajectory helper                                              #
    # ------------------------------------------------------------------ #

    def send_base_goal_blocking(self, joints_list, duration=5.0, bypass=False):
        """
        Send a FollowJointTrajectory goal and block until it finishes.
        Returns False immediately if abort is requested (which pause sets).
        execute_pause cancels the active goal handle directly, so the
        result_future resolves quickly with a cancelled/aborted status.
        """
        if self._abort_requested:
            self.get_logger().warn("Abort requested – skipping trajectory.")
            return False
    
        if self._is_paused and not bypass:
            return

        goal = self._build_goal(joints_list, duration)
        joint_names_str = ", ".join(goal.trajectory.joint_names)

        self.get_logger().info(f"Sending goal for joints: [{joint_names_str}]")
        send_future = self.trajectory_client.send_goal_async(goal)
        while not send_future.done():
            time.sleep(0.05)

        gh = send_future.result()
        if not gh.accepted:
            self.get_logger().error(f"Goal [{joint_names_str}] was rejected!")
            return False

        with self._goal_handle_lock:
            self._goal_handle = gh

        result_future = gh.get_result_async()
        while not result_future.done():
            time.sleep(0.05)
            if self._abort_requested:
                gh.cancel_goal_async()
                with self._goal_handle_lock:
                    self._goal_handle = None
                return False

        with self._goal_handle_lock:
            self._goal_handle = None

        result = result_future.result()
        success = result.status == GoalStatus.STATUS_SUCCEEDED
        if not success:
            self.get_logger().warn(f"Goal [{joint_names_str}] finished with status {result.status}.")
        else:
            self.get_logger().info(f"Goal [{joint_names_str}] succeeded.")
        return success

    @staticmethod
    def _build_goal(joints_list, duration):
        point = JointTrajectoryPoint()
        point.positions = [float(v) for _, v in joints_list]
        point.time_from_start = Duration(seconds=duration).to_msg()

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [n for n, _ in joints_list]
        goal.trajectory.points = [point]
        goal.goal_time_tolerance = Duration(seconds=10.0).to_msg()
        return goal

    # ------------------------------------------------------------------ #
    #  Utility / helpers (unchanged from original)                        #
    # ------------------------------------------------------------------ #

    def load_poses(self, file_path):
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception as e:
            self.get_logger().error(f"Could not load poses from {file_path}: {e}")
            return {}

    def switch_mode(self, mode):
        service_name = f"/switch_to_{mode}_mode"
        client = self.create_client(Trigger, service_name)
        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(f"Service {service_name} not available!")
            return False
        future = client.call_async(Trigger.Request())
        while not future.done():
            time.sleep(0.1)
        result = future.result()
        if result.success:
            self.get_logger().info(f"Switched to {mode} mode: {result.message}")
        else:
            self.get_logger().warn(f"Switch to {mode} mode failed: {result.message}")
        return result.success

    def turn_to_goal_rot(self, z, w):
        self.switch_mode("navigation")
        try:
            trans = self.tf_buffer.lookup_transform("map", "base_footprint", Time())
            current_x = trans.transform.translation.x
            current_y = trans.transform.translation.y
            current_rot = trans.transform.rotation
        except TransformException as e:
            self.get_logger().error(f"turn_to_goal_rot: TF error: {e}")
            return False

        initial_pose = PoseStamped()
        initial_pose.header.frame_id = 'map'
        initial_pose.header.stamp = self.navigator.get_clock().now().to_msg()
        initial_pose.pose.position.x = current_x
        initial_pose.pose.position.y = current_y
        initial_pose.pose.orientation = current_rot
        self.navigator.setInitialPose(initial_pose)

        current_quat_array = np.array([0.0, 0.0, current_rot.z, current_rot.w])
        yaw_quat = R.from_euler('z', 108, degrees=True)
        new_rotation = yaw_quat * R.from_quat(current_quat_array)
        new_quat_array = new_rotation.as_quat()

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.navigator.get_clock().now().to_msg()
        goal_pose.pose.position.x = current_x
        goal_pose.pose.position.y = current_y
        goal_pose.pose.orientation.z = new_quat_array[2]
        goal_pose.pose.orientation.w = new_quat_array[3]

        self.navigator.goToPose(goal_pose)
        while not self.navigator.isTaskComplete():
            time.sleep(0.1)

        result = self.navigator.getResult()
        self.switch_mode("position")
        if result == TaskResult.SUCCEEDED:
            self.get_logger().info("Turn to goal orientation succeeded.")
            return True
        self.get_logger().error(f"Turn to goal orientation failed: {result}")
        return False

    def compute_difference(self, target_frame, offset_x=0, offset_y=0, offset_z=0, offset_orientation=0):
        try:
            self.get_logger().info(
                f"aligning to offsets offset_x {offset_x}, offset_y {offset_y}, offset_z {offset_z}"
            )
            trans_base = self.tf_buffer.lookup_transform("base_link", target_frame, Time())
            x, y, z, w = (
                trans_base.transform.rotation.x,
                trans_base.transform.rotation.y,
                trans_base.transform.rotation.z,
                trans_base.transform.rotation.w,
            )
            Rot = quaternion_matrix((x, y, z, w))
            P_dash = np.array([[offset_x], [offset_y], [offset_z], [1]])
            P = np.array([
                [trans_base.transform.translation.x],
                [trans_base.transform.translation.y],
                [0], [1],
            ])
            X = np.matmul(Rot, P_dash)
            P_base = X + P
            P_base[3, 0] = 1
            base_position_x = P_base[0, 0]
            base_position_y = P_base[1, 0]
            phi = atan2(base_position_y, base_position_x)
            dist = sqrt(base_position_x**2 + base_position_y**2)
            _, _, z_rot_base = euler_from_quaternion([x, y, z, w])
            z_rot_base = -phi + z_rot_base + offset_orientation
            return phi, dist, z_rot_base
        except TransformException as e:
            self.get_logger().error(f"Transform error: {e}")
            return None, None, None

    def _get_recent_tf(self, source, target):
        try:
            tf = self.tf_buffer.lookup_transform(source, target, Time())
            tf_age = (self.get_clock().now() - Time.from_msg(tf.header.stamp)).nanoseconds / 1e9
            if tf_age <= MAX_TF_AGE:
                return tf
        except TransformException:
            pass
        return None

    def block_until_recent_tf(self, source, target):
        start = time.monotonic()
        tf = self._get_recent_tf(source, target)
        while tf is None:
            if time.monotonic() - start >= RECENT_TF_TIMEOUT:
                raise ValueError(f"TF {source}->{target} not available within {RECENT_TF_TIMEOUT}s.")
            time.sleep(RECENT_TF_POLL_TIME)
            tf = self._get_recent_tf(source, target)
        return tf

    def compute_angle_to_marker(self):
        tf = self.block_until_recent_tf("base_link", "trash_can")
        target_point_x, target_point_y = self._target_xy_from_tf(tf, TRASH_CAN_MARKER_OFFSET_X)
        angle = atan2(target_point_y, target_point_x)
        angle -= -np.pi / 2
        self.get_logger().info(f"Angle to offset marker: {angle}")
        return angle

    def _target_xy_from_tf(self, tf, x_offset):
        rotation_matrix = quaternion_matrix((
            tf.transform.rotation.x,
            tf.transform.rotation.y,
            tf.transform.rotation.z,
            tf.transform.rotation.w,
        ))
        offset_vector = array([[x_offset], [0], [tf.transform.translation.x], [1]])
        marker_vector = array([
            [tf.transform.translation.x],
            [tf.transform.translation.y],
            [0], [1],
        ])
        offset_direction = matmul(rotation_matrix, offset_vector)
        final_location = offset_direction + marker_vector
        return float(final_location[0, 0]), float(final_location[1, 0])

    def align_to_marker(self, target_frame, offset_x=0, offset_y=0, offset_z=0,
                        offset_orientation=0, use_trajectory=True,
                        offset_orientation_z=0, offset_orientation_w=0):
        self.get_logger().info(f"Aligning to {target_frame} with offset z={offset_z}")
        phi, dist, final_theta = self.compute_difference(
            target_frame, offset_x, offset_y, offset_z, offset_orientation
        )
        if phi is None:
            return False
        self.send_base_goal_blocking([("rotate_mobile_base", phi)])
        self.send_base_goal_blocking([("translate_mobile_base", dist)], 30.0)
        if use_trajectory:
            self.send_base_goal_blocking([("rotate_mobile_base", final_theta)])
        else:
            self.turn_to_goal_rot(offset_orientation_z, offset_orientation_w)
        return True

    def refine_and_hook_alignment(self, target_frame, offset_x, offset_y, offset_z, offset_orientation):
        self.get_logger().info("Initial approach done. Starting head pan and iterative refinement...")
        self.send_base_goal_blocking([("joint_head_pan", HEAD_PAN_SEARCH)])
        self.get_logger().info("Running visual servoing loop...")
        try:
            angle_to_marker = self.compute_angle_to_marker()
            while abs(angle_to_marker) > MINIMUM_ANGLE_THRESHOLD:
                self.get_logger().info(f"Correcting by {angle_to_marker}...")
                self.send_base_goal_blocking([("rotate_mobile_base", angle_to_marker)])
                angle_to_marker = self.compute_angle_to_marker()
        except Exception:
            self.get_logger().info("Aborting visual servoing – can't see marker.")
        self.get_logger().info("Iterative refinement complete.")
        self.send_base_goal_blocking([("joint_head_pan", HEAD_PAN_NEUTRAL)])
        time.sleep(0.5)

    def execute_named_pose_from_dict(self, pose_data):
        if "joints" in pose_data:
            joints = pose_data["joints"]
            def gj(key, default):
                return joints.get(key, default)
            lift_val = gj("joint_lift", 0.0)
            arm_total = gj("joint_arm_total", 0.0)
            yaw_val = gj("joint_wrist_yaw", 0.0)
            pitch_val = gj("joint_wrist_pitch", 0.0)
            roll_val = gj("joint_wrist_roll", 0.0)
        else:
            gripper_rpy = pose_data.get("gripper_rpy", {})
            def gf(key, default):
                return pose_data.get(key, default)
            def gr(key, default):
                return gripper_rpy.get(key, default)
            lift_val = gf("lift_height", 0.0)
            arm_total = gf("wrist_extension", 0.0)
            yaw_val = gr("joint_wrist_yaw", 0.0)
            pitch_val = gr("joint_wrist_pitch", 0.0)
            roll_val = gr("joint_wrist_roll", 0.0)

        arm_segment = arm_total / 4.0
        joints_list = [
            ("joint_lift",        lift_val),
            ("joint_arm_l0",      arm_segment),
            ("joint_arm_l1",      arm_segment),
            ("joint_arm_l2",      arm_segment),
            ("joint_arm_l3",      arm_segment),
            ("joint_wrist_yaw",   yaw_val),
            ("joint_wrist_pitch", pitch_val),
            ("joint_wrist_roll",  roll_val),
        ]
        self.get_logger().info(f"Joints list: {joints_list}")
        return self.send_base_goal_blocking(joints_list)

    # ------------------------------------------------------------------ #
    #  Task executors                                                      #
    # ------------------------------------------------------------------ #

    def execute_extraction(self):
        self.switch_mode("position")
        self.send_base_goal_blocking(
            [("joint_gripper_finger_left", -0.0757396), ("joint_head_tilt", -0.75)]
        )
        time.sleep(0.5)

        self.get_logger().info("Executing navigation (approaching trash can)...")
        start_poses = self.load_poses(CAN_START_POSE_FILE)

        if "trash_start" in start_poses:
            pose = start_poses["trash_start"]
            target_frame = pose.get("frame", "trash_can")
            offset_z = pose.get("position", {}).get("z")
            if self.align_to_marker(
                target_frame,
                offset_x=TRASH_CAN_MARKER_OFFSET_X,
                offset_z=offset_z,
                offset_orientation=TRASH_CAN_OFFSET_ORIENTATION,
                use_trajectory=False,
                offset_orientation_z=-0.906,
                offset_orientation_w=0.423,
            ):
                self.refine_and_hook_alignment(
                    target_frame=target_frame,
                    offset_x=TRASH_CAN_MARKER_OFFSET_X,
                    offset_y=0.0,
                    offset_z=0.1,
                    offset_orientation=TRASH_CAN_OFFSET_ORIENTATION,
                )
                self.execute_named_pose_from_dict(pose)

        self.execute_just_extraction()

    def execute_just_extraction(self):
        self.get_logger().info("Executing extraction (picking up trash)...")
        pickup_poses = self.load_poses(CAN_PICKUP_POSE_FILE)
        for pose_name in ["before_pickup", "during_pickup", "pickup_scoop", "pickup_high", "pickup_retracted"]:
            if self._is_paused or self._abort_requested:
                self.get_logger().warn(f"Extraction sequence aborted before executing: {pose_name}")
                return
            
            if pose_name in pickup_poses:
                self.get_logger().info(f"Executing pose: {pose_name}")
                self.execute_named_pose_from_dict(pickup_poses[pose_name])
                time.sleep(5.0)
                # Surface any abort/pause that fired during the sleep.
                if self._abort_requested:
                    return

    def execute_disposal(self):
        self.switch_mode("position")
        self.get_logger().info("Executing navigation (approaching receptacle)...")
        poses = self.load_poses(RECEPTACLE_START_POSE_FILE)

        if "receptacle_start" in poses:
            start_pose = poses["receptacle_start"]
            target_frame = start_pose.get("frame", "receptacle")
            offset_z = start_pose.get("position", {}).get("z", 0.0)
            offset_x = start_pose.get("position", {}).get("x", 0.0)
            if self.align_to_marker(
                target_frame,
                offset_x=offset_x,
                offset_z=offset_z,
                offset_orientation=RECEPTACLE_OFFSET_ORIENTATION,
                use_trajectory=True,
                offset_orientation_z=0.431,
                offset_orientation_w=0.903,
            ):
                self.execute_named_pose_from_dict(start_pose)
                self.send_base_goal_blocking([("translate_mobile_base", 0.7)])
                self.send_base_goal_blocking([("translate_mobile_base", 0.7)])
                time.sleep(2.0)

        self.execute_just_disposal()

    def execute_just_disposal(self):
        poses = self.load_poses(RECEPTACLE_START_POSE_FILE)
        self.get_logger().info("Executing disposal (dropping into receptacle)...")
        if "receptacle_drop" in poses:
            if self._is_paused or self._abort_requested:
                self.get_logger().warn(f"Extraction sequence aborted before executing: receptacle_drop")
                return
            self.execute_named_pose_from_dict(poses["receptacle_drop"])

    def execute_go_to_receptacle(self):
        """
        Navigate through RECEPTACLE_WAYPOINTS via Nav2.
        If paused mid-navigation, cancels Nav2 and publishes a message
        telling the operator to teleop the robot to the destination.
        Returns "SUCCESS", "PAUSED", "CANCELED", or "FAILED".
        """
        self.switch_mode("navigation")

        initial_pose = PoseStamped()
        initial_pose.header.frame_id = 'map'
        initial_pose.header.stamp = self.navigator.get_clock().now().to_msg()
        initial_pose.pose.position.x = 1.9755
        initial_pose.pose.position.y = 0.61291
        initial_pose.pose.orientation.z = 0.97553
        initial_pose.pose.orientation.w = -0.21984
        self.navigator.setInitialPose(initial_pose)

        route_poses = []
        for x, y, z, w in RECEPTACLE_WAYPOINTS:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = self.navigator.get_clock().now().to_msg()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.z = z
            pose.pose.orientation.w = w
            route_poses.append(pose)

        self.get_logger().info(f"Following {len(route_poses)} waypoints to receptacle...")
        self.navigator.followWaypoints(route_poses)

        i = 0
        while not self.navigator.isTaskComplete():
            if self._is_paused or self._abort_requested:
                self.get_logger().warn("Navigation interrupted — cancelling Nav2 task.")
                self.navigator.cancelTask()
                if self._is_paused:
                    self._status_pub.publish(
                        String(data=(
                            "PAUSED: Navigation cancelled. "
                            "Please teleop the robot to the receptacle, "
                            "then press Resume to run disposal."
                        ))
                    )
                return "PAUSED"

            i += 1
            feedback = self.navigator.getFeedback()
            if feedback and i % 5 == 0:
                self.get_logger().info(
                    f"Waypoint {feedback.current_waypoint + 1}/{len(route_poses)}"
                )
            time.sleep(0.1)

        result = self.navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            self.get_logger().info("Navigation to receptacle succeeded.")
            return "SUCCESS"
        elif result == TaskResult.CANCELED:
            self.get_logger().warn("Navigation to receptacle was cancelled.")
            return "CANCELED"
        else:
            self.get_logger().error(f"Navigation to receptacle failed: {result}")
            return "FAILED"

    def execute_sequence(self):
        """Full automatic sequence: reset → extraction → navigation → disposal."""
        self._abort_requested = False
        self._is_paused = False
        self.execute_reset()
        if not self._abort_requested:
            self._run_sequence_from("extraction")

    def execute_reset(self):
        self.get_logger().info("Executing reset (returning to neutral pose)...")
        joints_list = [
            ("joint_lift",        0.8),
            ("joint_arm_l0",      0.0),
            ("joint_arm_l1",      0.0),
            ("joint_arm_l2",      0.0),
            ("joint_arm_l3",      0.0),
            ("joint_wrist_yaw",   0.0),
            ("joint_wrist_pitch", 0.0),
            ("joint_wrist_roll",  0.0),
        ]
        self.send_base_goal_blocking(joints_list, bypass=True)


def main(args=None):
    rclpy.init(args=args)
    waste_disposal = WasteDisposal()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(waste_disposal)
    executor.add_node(waste_disposal.navigator)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()


if __name__ == '__main__':
    main()