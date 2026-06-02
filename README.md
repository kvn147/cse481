Terminal 1: ros2 launch stretch_core stretch_driver.launch.py

Terminal 2: ros2 launch stretch_core d435i_high_resolution.launch.py

Terminal 3: ros2 launch stretch_core stretch_aruco.launch.py

Terminal 4: ros2 run rviz2 rviz2 -d /home/hello-robot/ament_ws/src/stretch_tutorials/rviz/aruco_detector_example.rviz

Terminal 5: ros2 launch rosbridge_server rosbridge_websocket_launch.xml

Terminal 6: 
```
cd ~/bresenham/lab13
python3 -m http.server 8000
```
Can open website in: http://slinky.hcrlab.cs.washington.edu:8000

Terminal 7: 
python3 aruco_navigator.py

Terminal 8 (optional): Keyboard teleop

cd /home/hello-robot/kevin/cse481/final_project && python3 ros_action_server.py
ros2 topic pub --once /task_execution std_msgs/msg/String "data: 'disposal'"

ros2 launch stretch_core stretch_driver.launch.py mode:=navigation broadcast_odom_tf:=True

tmux kill-server
