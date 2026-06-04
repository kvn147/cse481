#!/bin/bash

# Check argument
if [ "$1" == "slinky" ]; then
    WEB_DIR="$HOME/kevin/cse481/final_project"
    TARGET_URL="http://slinky.hcrlab.cs.washington.edu:8000"
elif [ "$1" == "weird-stretch" ]; then
    WEB_DIR="$HOME/bresenham/cse481/final_project"
    TARGET_URL="http://weird-stretch.cs.washington.edu:8000"
elif [ "$1" == "rocky" ]; then
    WEB_DIR="$HOME/kevin/cse481/final_project"
    TARGET_URL="http://rocky.hcrlab.cs.washington.edu:8000"
else
    echo "Usage: $0 [slinky|weird-stretch|rocky]"
    exit 1
fi

# echo "If necessary, uncomment correct url in lab13/pose_manager_frontend.html"

SESSION="dev"

# Ensure a clean start
tmux kill-session -t "$SESSION" 2>/dev/null
tmux new-session -d -s "$SESSION" -n 'driver'

# no renaming windows
tmux set-option -t "$SESSION:driver" allow-rename off
# tmux send-keys -t "$SESSION:camera" "ros2 launch stretch_core d435i_high_resolution.launch.py" C-m

# Terminal 2: camera
tmux new-window -t "$SESSION" -n 'camera'
tmux send-keys -t "$SESSION:camera" "ros2 launch stretch_core d435i_high_resolution.launch.py" C-m

# Terminal 3: aruco
tmux new-window -t "$SESSION" -n 'aruco'
tmux send-keys -t "$SESSION:aruco" "ros2 launch stretch_core stretch_aruco.launch.py" C-m

# Terminal 4: rviz
tmux new-window -t "$SESSION" -n 'rviz'
tmux send-keys -t "$SESSION:rviz" "ros2 run rviz2 rviz2 -d /home/hello-robot/ament_ws/src/stretch_tutorials/rviz/aruco_detector_example.rviz" C-m

# Terminal 5: rosbridge
tmux new-window -t "$SESSION" -n 'rosbridge'
tmux send-keys -t "$SESSION:rosbridge" "ros2 launch rosbridge_server rosbridge_websocket_launch.xml" C-m

# publish to camera topic
tmux new-window -t "$SESSION" -n 'web_video_server'
tmux send-keys -t "$SESSION:web_video_server" "ros2 run web_video_server web_video_server" C-m

# Terminal 6: web server
tmux new-window -t "$SESSION" -n 'webserver'
tmux send-keys -t "$SESSION:webserver" "cd ${WEB_DIR}/frontend && python3 -m http.server 8000" C-m

# Terminal 7: Nav2 with map
tmux new-window -t "$SESSION" -n 'map'
tmux send-keys -t "$SESSION:map" "ros2 launch stretch_nav2 navigation.launch.py map:=${WEB_DIR}/maps/mango_map_room_correct.yaml params_file:=/home/hello-robot/ament_ws/src/stretch_ros2/stretch_nav2/config/nav2_params.yaml autostart:=true" C-m

# Terminal 8: action_server
tmux new-window -t "$SESSION" -n 'action_server'
tmux send-keys -t "$SESSION:action_server" "cd ${WEB_DIR} && python3 ros_action_server.py" C-m

# Terminal 9: keyboard_teleop
tmux new-window -t "$SESSION" -n 'keyboard_teleop'
tmux send-keys -t "$SESSION:keyboard_teleop" "ros2 run stretch_core keyboard_teleop" C-m

# Terminal 10: cloudflare
tmux new-window -t "$SESSION" -n 'cloudflare'
tmux send-keys -t "$SESSION:cloudflare" "cloudflared tunnel run --token-file /home/hello-robot/kevin/cloudflare_token.txt" C-m
tmux attach-session -t "$SESSION"