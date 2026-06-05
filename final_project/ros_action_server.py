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
import threading  # <-- Added to handle blocking action calls safely
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

CAN_START_POSE_FILE = "/home/hello-robot/kevin/cse481/final_project/aruco_data/trash_start.json" # this is how stretch approaches the can
CAN_PICKUP_POSE_FILE = "/home/hello-robot/kevin/cse481/final_project/joint_state_data/trash_pickup.json" # this is the extraction poses
RECEPTACLE_START_POSE_FILE = "/home/hello-robot/kevin/cse481/final_project/aruco_data/receptacle_start.json" # this is the approach pose for the receptacle

TRASH_CAN_OFFSET_ORIENTATION = np.pi
RECEPTACLE_OFFSET_ORIENTATION = np.pi/2

# Hardcoded map-frame waypoints for navigating to the receptacle.
RECEPTACLE_WAYPOINTS = [
    [3.38, -0.75, 0.0, 1.0], # by table straight from trash
    [4.19, -1.5, 0.0, 1.0], # by table right after
    [5.21, -2.3438, 0.0, 1.0], #in doorway
    [5.246, -4.31, 0.259,  0.966]  # looking at receptacle
]

# --- NEW ADDITIONS FOR REFINEMENT AND CAMERA PAN ---
HEAD_PAN_SEARCH = -1 * np.pi / 2
HEAD_PAN_NEUTRAL = 0.0
MINIMUM_ANGLE_THRESHOLD = 0.03 
MAX_TF_AGE = 1.0                # seconds
RECENT_TF_TIMEOUT = 5.0         # seconds
RECENT_TF_POLL_TIME = 0.1       # seconds
TRASH_CAN_MARKER_OFFSET_X = 0.17
# ---------------------------------------------------

class WasteDisposal(Node):
    def __init__(self):
        super().__init__('waste_disposal')

        # TF and Action Client setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/stretch_controller/follow_joint_trajectory",
        )

        self._goal_handle = None
        self._abort_requested = False
        self._is_paused = False

        if not self.trajectory_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("Unable to connect to trajectory server.")

        # Nav2 navigator
        self.navigator = BasicNavigator()

        # for sequence state tracking
        self._current_sequence_step = "READY"
        self._last_saved_waypoint = 0

        # Joint states tracking
        self.current_tilt = 0.0
        self.current_pan = 0.0
        self.joint_state_sub = self.create_subscription(
            JointState,
            "/stretch/joint_states",
            self.joint_states_callback,
            10
        )

        # Subscriber
        self.subscription = self.create_subscription(
            String,
            'task_execution',
            self.task_callback,
            10
        )

        self.get_logger().info("Waiting for Nav2 to become active...")
        self.navigator.waitUntilNav2Active()

        self.get_logger().info("Waste Disposal node started and listening to /task_execution.")

    def joint_states_callback(self, msg):
        for name, pos in zip(msg.name, msg.position):
            if name == "joint_head_tilt":
                self.current_tilt = pos
            elif name == "joint_head_pan":
                self.current_pan = pos

    def task_callback(self, msg):
        task_type = msg.data.strip().lower()
        self.get_logger().info(f"Received task: {task_type}")
        
        handler = getattr(self, f"execute_{task_type}", None)
        
        if not handler:
            self.get_logger().error(f"Unknown task type: {task_type}")
            return
        
        if task_type not in ["stop", "pause", "resume"]:
            self._abort_requested = False
            self._is_paused = False
        
        # Run execution in a separate background thread so rclpy.spin() doesn't deadlock
        threading.Thread(target=handler, daemon=True).start()

    # --- TELEOP HANDLERS ---
    def execute_base_forward(self):
        self.send_base_goal_blocking([("translate_mobile_base", 0.1)], duration=2.0)

    def execute_base_backward(self):
        self.send_base_goal_blocking([("translate_mobile_base", -0.1)], duration=2.0)

    def execute_base_left(self):
        self.send_base_goal_blocking([("rotate_mobile_base", 0.2)], duration=2.0)

    def execute_base_right(self):
        self.send_base_goal_blocking([("rotate_mobile_base", -0.2)], duration=2.0)

    def execute_camera_up(self):
        new_tilt = self.current_tilt + 0.2
        self.get_logger().info(f"Moving camera UP to {new_tilt}")
        self.send_base_goal_blocking([("joint_head_tilt", new_tilt)], duration=0.5)

    def execute_camera_down(self):
        new_tilt = self.current_tilt - 0.2
        self.get_logger().info(f"Moving camera DOWN to {new_tilt}")
        self.send_base_goal_blocking([("joint_head_tilt", new_tilt)], duration=0.5)

    def execute_camera_left(self):
        new_pan = self.current_pan + 0.2
        self.get_logger().info(f"Moving camera LEFT to {new_pan}")
        self.send_base_goal_blocking([("joint_head_pan", new_pan)], duration=0.5)

    def execute_camera_right(self):
        new_pan = self.current_pan - 0.2
        self.get_logger().info(f"Moving camera RIGHT to {new_pan}")
        self.send_base_goal_blocking([("joint_head_pan", new_pan)], duration=0.5)
    # -----------------------

    def load_poses(self, file_path):
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception as e:
            self.get_logger().error(f"Could not load poses from {file_path}: {e}")
            return {}
        
    def switch_mode(self, mode):
        """Call /stretch/switch_to_position_mode or navigation_mode. mode='position' or 'navigation'"""
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
        """Use Nav2 to rotate in place to the given quaternion orientation (z, w components)."""
        self.switch_mode("navigation")

        # Get current position from TF
        try:
            trans = self.tf_buffer.lookup_transform("map", "base_footprint", Time())
            current_x = trans.transform.translation.x
            current_y = trans.transform.translation.y
            current_rot = trans.transform.rotation
        except TransformException as e:
            self.get_logger().error(f"turn_to_goal_rot: TF error: {e}")
            return False

        # Set initial pose from map -> base_footprint TF so AMCL is correctly localized
        initial_pose = PoseStamped()
        initial_pose.header.frame_id = 'map'
        initial_pose.header.stamp = self.navigator.get_clock().now().to_msg()
        initial_pose.pose.position.x = current_x
        initial_pose.pose.position.y = current_y
        initial_pose.pose.orientation = current_rot
        self.navigator.setInitialPose(initial_pose)
        self.get_logger().info(f"Set initial pose to x={current_x:.3f}, y={current_y:.3f}")

        # 1. Define your current pose orientation (Example: no rotation, [x, y, z, w])
        current_quat_array = np.array([0.0, 0.0, current_rot.z, current_rot.w])

        # 2. Create a quaternion for a 90 degree Yaw rotation (around the Z-axis)
        # Note: In most robotics conventions, Z is the yaw axis
        yaw_quat = R.from_euler('z', 108, degrees=True)

        # 3. Multiply them to combine the rotations
        # Note: SciPy allows direct multiplication of Rotation objects
        current_rotation = R.from_quat(current_quat_array)
        new_rotation = yaw_quat * current_rotation

        # Get the final quaternion [x, y, z, w]
        new_quat_array = new_rotation.as_quat()

        # Goal is same x,y but with target orientation
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.navigator.get_clock().now().to_msg()
        goal_pose.pose.position.x = current_x
        goal_pose.pose.position.y = current_y
        goal_pose.pose.orientation.z = new_quat_array[2]
        goal_pose.pose.orientation.w = new_quat_array[3]
        # goal_pose.pose.orientation.z = z
        # goal_pose.pose.orientation.w = w

        self.get_logger().info(f"Turning in place from pose x = {current_x} y = {current_y} z = {current_rot.z} w = {current_rot.w} to orientation z={z}, w={w}...")
        self.navigator.goToPose(goal_pose)

        while not self.navigator.isTaskComplete():
            time.sleep(0.1)

        result = self.navigator.getResult()
        self.switch_mode("position")
        if result == TaskResult.SUCCEEDED:
            self.get_logger().info("Turn to goal orientation succeeded.")
            return True
        elif result == TaskResult.CANCELED:
            self.get_logger().warn("Turn to goal orientation was cancelled.")
            return False
        else:
            self.get_logger().error(f"Turn to goal orientation failed: {result}")
            return False

    def send_base_goal_blocking(self, joints_list, duration=5.0):
        # Wait if paused
        while self._is_paused:
            time.sleep(0.1)
            if self._abort_requested:
                return False

        if self._abort_requested:
            self.get_logger().warn("Task aborted! Skipping trajectory execution.")
            return False

        point = JointTrajectoryPoint()
        point.positions = [float(inc) for _, inc in joints_list]
        point.time_from_start = Duration(seconds=duration).to_msg()

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [joint_name for joint_name, _ in joints_list]
        goal.trajectory.points = [point]
        goal.goal_time_tolerance = Duration(seconds=10.0).to_msg()

        joint_names_str = ", ".join(goal.trajectory.joint_names)
        self.get_logger().info(f"Sending goal for joints: [{joint_names_str}]")

        send_goal_future = self.trajectory_client.send_goal_async(goal)
        
        # Use a passive sleep loop instead of spin_until_future_complete to avoid deadlock
        while not send_goal_future.done():
            time.sleep(0.1)
            
        self._goal_handle = send_goal_future.result()

        if not self._goal_handle.accepted:
            self.get_logger().error(f"Goal for joints [{joint_names_str}] was rejected!")
            self._goal_handle = None
            return False

        result_future = self._goal_handle.get_result_async()
        while not result_future.done():
            time.sleep(0.1)
            
        result = result_future.result()
        self._goal_handle = None

        if result.status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().warn(f"Goal for joints [{joint_names_str}] did not succeed: status {result.status}")
            return False
        
        self.get_logger().info(f"Goal for joints [{joint_names_str}] succeeded.")
        return True

    def compute_difference(self, target_frame, offset_x=0, offset_y=0, offset_z=0, offset_orientation=0):
        try:
            self.get_logger().info(f"aligning to offsets offset_x {offset_x}, offset_y {offset_y}, offset_z {offset_z}")

            # Extract quaternion and rotation matrix of marker in base_link frame
            trans_base = self.tf_buffer.lookup_transform(
                    "base_link", target_frame, Time()
                    )
            x, y, z, w = (
                trans_base.transform.rotation.x,
                trans_base.transform.rotation.y,
                trans_base.transform.rotation.z,
                trans_base.transform.rotation.w,
            )
            R = quaternion_matrix((x, y, z, w))

            # Apply rotation to the offset vector, positive Z DIRECTION IN OUR CASE
            P_dash = np.array([[offset_x], [offset_y], [offset_z], [1]])
            P = np.array(
                [
                    [trans_base.transform.translation.x],
                    [trans_base.transform.translation.y],
                    [0],
                    [1],
                ]
            )
            X = np.matmul(R, P_dash)

            # Compute the marker position with offset in base_link frame
            P_base = X + P
            P_base[3, 0] = 1  # Homogeneous coordinate

            # Extract adjusted position
            base_position_x = P_base[0, 0]
            base_position_y = P_base[1, 0]

            # Compute rotation and translation needed
            phi = atan2(base_position_y, base_position_x)
            dist = sqrt(base_position_x**2 + base_position_y**2)

            _, _, z_rot_base = euler_from_quaternion([x, y, z, w])
            # Calculate final rotation: -phi (cancel rotation needed to align),
            # + z_rot_base (original marker rotation),
            # + pi (such that the base and the marker axis are aligned as shown in tutorial)
            
            # offset_orientation: np.pi for trash can start, np.pi/2 for receptacle
            z_rot_base = -phi + z_rot_base + offset_orientation

            return phi, dist, z_rot_base
        except TransformException as e:
            self.get_logger().error(f"Transform error: {e}")
            return None, None, None

    # --- NEW ADDITIONS FOR REFINEMENT ---
    def _get_recent_tf(self, source: str, target: str):
        """Helper to fetch TF only if it's fresh."""
        try:
            tf = self.tf_buffer.lookup_transform(source, target, Time())
            current_time = self.get_clock().now()
            tf_age = (current_time - Time.from_msg(tf.header.stamp)).nanoseconds / 1e9
            if tf_age <= MAX_TF_AGE:
                return tf
        except TransformException:
            pass
        return None

    def block_until_recent_tf(self, source: str, target: str):
        """Blocks and logs until a fresh TF is available to prevent extrapolation errors."""
        start = time.monotonic()
        tf = self._get_recent_tf(source, target)
        while tf is None:
            if time.monotonic() - start >= RECENT_TF_TIMEOUT:
                raise ValueError(f"TF {source}->{target} not available within {RECENT_TF_TIMEOUT}s.")
            time.sleep(RECENT_TF_POLL_TIME)
            tf = self._get_recent_tf(source, target)
        return tf

    def compute_angle_to_marker(self) -> float:
        tf = self.block_until_recent_tf("base_link", "trash_can")

        # Compute angle based on location.
        target_point_x, target_point_y = self._target_xy_from_tf(tf, TRASH_CAN_MARKER_OFFSET_X)
        angle = atan2(target_point_y, target_point_x)

        # Adjust for head pan offset.
        angle -= - np.pi/2



        self.get_logger().info(f"Angle to offset marker: {angle}")
        return angle


    def _target_xy_from_tf(self,
        tf: TransformStamped, x_offset: float
    ) -> tuple[float, float]:
        """Compute the target point in the robot frame after applying the marker offset."""
        rotation_matrix = quaternion_matrix(
            (
                tf.transform.rotation.x,
                tf.transform.rotation.y,
                tf.transform.rotation.z,
                tf.transform.rotation.w,
            )
        )
        offset_vector = array([[x_offset], [0], [tf.transform.translation.x], [1]])
        marker_vector = array(
            [[tf.transform.translation.x], [tf.transform.translation.y], [0], [1]]
        )
        offset_direction = matmul(rotation_matrix, offset_vector)
        final_location = offset_direction + marker_vector
        return float(final_location[0, 0]), float(final_location[1, 0])
    # ------------------------------------


    def align_to_marker(self, target_frame, offset_x=0, offset_y=0, offset_z=0, offset_orientation=0, use_trajectory=True, offset_orientation_z=0, offset_orientation_w=0):
        self.get_logger().info(f"Aligning to {target_frame} with offset z={offset_z}")
        phi, dist, final_theta = self.compute_difference(target_frame, offset_x, offset_y, offset_z, offset_orientation)
        
        if phi is None:
            return False

        # Split base goals because they are mutually exclusive in the hardware controller
        self.send_base_goal_blocking([("rotate_mobile_base", phi)])
        self.send_base_goal_blocking([("translate_mobile_base", dist)], 30.0)
        if use_trajectory:
            self.send_base_goal_blocking([("rotate_mobile_base", final_theta)])
        else:
            self.turn_to_goal_rot(offset_orientation_z, offset_orientation_w)
        return True

    def refine_and_hook_alignment(self, target_frame, offset_x, offset_y, offset_z, offset_orientation):
        """
        Executes precision local visual servoing and the final hook turn 
        AFTER the initial base alignment and translation are complete.
        """
        self.get_logger().info("Initial approach done. Starting head pan and iterative refinement...")

        # 1. PAN HEAD (Look Left to find the marker from the side)
        self.send_base_goal_blocking([("joint_head_pan", HEAD_PAN_SEARCH)])

        # 2. ITERATIVE REFINEMENT LOOP
        self.get_logger().info("Running visual servoing loop...")
        
        try:
            angle_to_marker = self.compute_angle_to_marker()
            while abs(angle_to_marker) > MINIMUM_ANGLE_THRESHOLD:
                # Rotate.
                self.get_logger().info(f"Correcting by {angle_to_marker}...")
                self.send_base_goal_blocking([("rotate_mobile_base",angle_to_marker)])

                # Compute current angle.
                angle_to_marker = self.compute_angle_to_marker()
        except:
            self.get_logger().info(f"Aborting visual servoing because can't see marker")

        self.get_logger().info("Iterative refinement complete.")

        # 3. PAN HEAD BACK TO NEUTRAL
        self.send_base_goal_blocking([("joint_head_pan", HEAD_PAN_NEUTRAL)])
        time.sleep(0.5)

    def execute_named_pose_from_dict(self, pose_data):
        if "joints" in pose_data:
            joints = pose_data["joints"]

            def get_joint(key, default):
                if key not in joints:
                    self.get_logger().warn(f"Joint '{key}' not found in joints dict, using default: {default}")
                return joints.get(key, default)

            lift_val = get_joint("joint_lift", 0.0)
            arm_total = get_joint("joint_arm_total", 0.0)
            yaw_val = get_joint("joint_wrist_yaw", 0.0)
            pitch_val = get_joint("joint_wrist_pitch", 0.0)
            roll_val = get_joint("joint_wrist_roll", 0.0)
        else:
            gripper_rpy = pose_data.get("gripper_rpy", {})

            def get_flat(key, default):
                if key not in pose_data:
                    self.get_logger().warn(f"Key '{key}' not found in pose_data, using default: {default}")
                return pose_data.get(key, default)

            def get_rpy(key, default):
                if key not in gripper_rpy:
                    self.get_logger().warn(f"Key '{key}' not found in gripper_rpy, using default: {default}")
                return gripper_rpy.get(key, default)

            lift_val = get_flat("lift_height", 0.0)
            arm_total = get_flat("wrist_extension", 0.0)
            yaw_val = get_rpy("joint_wrist_yaw", 0.0)
            pitch_val = get_rpy("joint_wrist_pitch", 0.0)
            roll_val = get_rpy("joint_wrist_roll", 0.0)

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

    def execute_extraction(self):
        self.switch_mode("position")
        self.send_base_goal_blocking([("joint_gripper_finger_left", -0.0757396), ("joint_head_tilt", -0.75)])
        time.sleep(0.5)

        # Approach
        self.get_logger().info("Executing navigation (approaching trash can)...")
        start_poses = self.load_poses(CAN_START_POSE_FILE)

        if "trash_start" in start_poses:
            pose = start_poses["trash_start"]
            target_frame = pose.get("frame", "trash_can")
            offset_z = pose.get("position", {}).get("z")
            if self.align_to_marker(target_frame, offset_x=TRASH_CAN_MARKER_OFFSET_X, offset_z=offset_z, offset_orientation=TRASH_CAN_OFFSET_ORIENTATION, use_trajectory=False, offset_orientation_z=-0.906, offset_orientation_w= 0.423):
                self.refine_and_hook_alignment(
                    target_frame=target_frame,
                    offset_x=TRASH_CAN_MARKER_OFFSET_X,
                    offset_y=0.0,
                    offset_z=0.1,
                    offset_orientation=TRASH_CAN_OFFSET_ORIENTATION
                )
                # self.turn_to_goal_rot(0.0, 0.0)
                # self.send_base_goal_blocking([("rotate_mobile_base", np.pi/2)])
                self.execute_named_pose_from_dict(pose)

        # Extraction
        self.execute_just_extraction()

    def execute_just_extraction(self):
        # Extraction
        self.get_logger().info("Executing extraction (picking up trash)...")
        pickup_poses = self.load_poses(CAN_PICKUP_POSE_FILE)
        # Sequence: before_pickup -> (grip) -> pickup_high -> pickup_retracted
        for pose_name in ["before_pickup", "during_pickup", "pickup_scoop", "pickup_high", "pickup_retracted"]:
            if pose_name in pickup_poses:
                self.get_logger().info(f"Executing pose: {pose_name}")
                self.execute_named_pose_from_dict(pickup_poses[pose_name])
                time.sleep(5.0)
    
    def execute_disposal(self):
        self.switch_mode("position")

        # Approach
        self.get_logger().info("Executing navigation (approaching receptacle)...")
        poses = self.load_poses(RECEPTACLE_START_POSE_FILE)

        if "receptacle_start" in poses:
            start_pose = poses["receptacle_start"]
            target_frame = start_pose.get("frame", "receptacle")

            offset_z = start_pose.get("position", {}).get("z", 0.0)
            offset_x = start_pose.get("position", {}).get("x", 0.0)
            if self.align_to_marker(target_frame, offset_x=offset_x, offset_z=offset_z, offset_orientation=RECEPTACLE_OFFSET_ORIENTATION, use_trajectory=True, offset_orientation_z =  0.431, offset_orientation_w = 0.903):
                self.execute_named_pose_from_dict(start_pose)
                self.send_base_goal_blocking([("translate_mobile_base", 0.7)])  # move forward
                self.send_base_goal_blocking([("translate_mobile_base", 0.7)])  # move forward
                self.send_base_goal_blocking([("translate_mobile_base", 0.2)])  # move forward
                time.sleep(2.0)

        self.execute_just_disposal()
    
    def execute_just_disposal(self):
        poses = self.load_poses(RECEPTACLE_START_POSE_FILE)

        # disposal is in same JSON as approach
        self.get_logger().info("Executing disposal (dropping into receptacle)...")

        if "receptacle_drop" in poses:
            drop_pose = poses["receptacle_drop"]
            self.execute_named_pose_from_dict(drop_pose)

    def execute_go_to_receptacle(self, start_index=0):
        """
        Use Nav2 to navigate through RECEPTACLE_WAYPOINTS in sequence.
        Blocks until all waypoints are visited, navigation fails, or is cancelled.
        Returns True on success, False otherwise.
        """
        self.switch_mode("navigation")

        if start_index == 0:
            # Set our demo's initial pose
            initial_pose = PoseStamped()
            initial_pose.header.frame_id = 'map'
            initial_pose.header.stamp = self.navigator.get_clock().now().to_msg()
            initial_pose.pose.position.x = 1.9755
            initial_pose.pose.position.y = 0.61291
            initial_pose.pose.orientation.z = 0.97553
            initial_pose.pose.orientation.w = -0.21984

            self.navigator.setInitialPose(initial_pose)

        # Build the list of PoseStamped waypoints from the constants at the top
        route_poses = []
        for x, y, z, w in RECEPTACLE_WAYPOINTS[start_index:]:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = self.navigator.get_clock().now().to_msg()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.z = z
            pose.pose.orientation.w = w

            route_poses.append(pose)

        self.get_logger().info(
            f"Following {len(route_poses)} waypoints to receptacle...and starting from index {start_index}..."
        )
        self.navigator.followWaypoints(route_poses)

        # Block until Nav2 finishes, logging current waypoint along the way
        i = 0
        last_completed_index_in_slice = 0
        
        while not self.navigator.isTaskComplete():
            if self._is_paused:
                self.get_logger().warn("Navigation paused. Canceling active Nav2 task and saving progress.")
                feedback = self.navigator.getFeedback()
                if feedback:
                    self._last_saved_waypoint = start_index + feedback.current_waypoint
                else:
                    self._last_saved_waypoint = start_index + last_completed_index_in_slice    
                
                self.navigator.cancelTask()
                return "PAUSED"

            i += 1
            feedback = self.navigator.getFeedback()
            if feedback and i % 5 == 0:
                self.get_logger().info(
                    f"Executing waypoint {feedback.current_waypoint + 1}/{len(route_poses)}"
                )

        result = self.navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            self.get_logger().info("Navigation to receptacle succeeded.")
            return "SUCCESS"
        elif result == TaskResult.CANCELED:
            self.get_logger().warn("Navigation to receptacle was cancelled.")
            return "CANCELED"
        elif result == TaskResult.FAILED:
            self.get_logger().error("Navigation to receptacle failed.")
            return "FAILED"
        else:
            self.get_logger().error(f"Navigation to receptacle returned unknown result: {result}")
            return "FAILED"

    def execute_sequence(self):
        self.get_logger().info("Starting automatic sequence: Extraction -> Disposal")
        
        if self._current_sequence_step == "READY":
            self.execute_reset()
            self._current_sequence_step = "EXTRACTION"
        
        if self._current_sequence_step == "EXTRACTION":
            self.execute_extraction()
            self._current_sequence_step = "GO_TO_RECEPTACLE"

        if self._current_sequence_step == "GO_TO_RECEPTACLE":    
            status = self.execute_go_to_receptacle(start_index=self._last_saved_waypoint)
            if status == "PAUSED":
                return
            elif status != "SUCCESS":
                self._current_sequence_step = "READY"
                return
            
            self._current_sequence_step = "DISPOSAL"
        
        if self._current_sequence_step == "DISPOSAL":
            self.execute_disposal()
            self._current_sequence_step = "READY"
            self.get_logger().info("Automatic sequence completed.")

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
        self.send_base_goal_blocking(joints_list)

    def execute_stop(self):
        self.get_logger().warn("Stop requested! Halting immediately.")
        self._abort_requested = True
        if self._goal_handle:
            self.get_logger().info("Attempting to cancel current trajectory...")
            self._goal_handle.cancel_goal_async()
            self._goal_handle = None
        else:
            self.get_logger().info("No active trajectory to cancel.")

    def execute_pause(self):
        self.get_logger().info("Pause requested.")
        self._is_paused = True

    def execute_resume(self):
        self.get_logger().info("Resume requested.")
        self._is_paused = False

        # spin thread back up if was in the pipeline
        if self._current_sequence_step in ["GO_TO_RECEPTACLE", "EXTRACTION", "DISPOSAL"]:
            threading.Thread(target=self.execute_sequence, daemon=True).start()
           
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

    waste_disposal._nav_executor.shutdown()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
