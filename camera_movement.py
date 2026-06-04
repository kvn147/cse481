import stretch_body.robot
import time
from stretch_body.hello_utils import deg_to_rad

robot = stretch_body.robot.Robot()
robot.startup() 


# Head
robot.head.move_to('head_pan', 0)
robot.head.move_to('head_tilt', 0)
robot.push_command()
time.sleep(2.0)

robot.head.move_to('head_tilt', deg_to_rad(-45.0))
robot.push_command()
time.sleep(2.0)

# Zero end-of-arm wrist
robot.end_of_arm.move_to('wrist_yaw', 0)
robot.end_of_arm.move_to('wrist_pitch', 0)  # Dex Wrist only
robot.end_of_arm.move_to('wrist_roll', 0)   # Dex Wrist only
robot.push_command()
time.sleep(2.0)

# Translate base
robot.base.rotate_by(deg_to_rad(90))
robot.push_command()
time.sleep(3.0)

robot.stop()