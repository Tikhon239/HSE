```code
# Setup Xquartz access control
xhost + 127.0.0.1

# run docker
docker run -e DISPLAY=host.docker.internal:0 -v ~/Desktop/hse/self_driving:/root/self_driving -it ros

# Create workspace/src directory
mkdir -p ~/self_driving/hw1/src
cd ~/self_driving/hw1/src
# Create top-level CMakelists.txt
catkin_init_workspace
cd ~/self_driving/hw1
# Create workspace overlay scripts
catkin_make
# Activate the new workspace
source devel/setup.bash

cd ~/self_driving/hw1/src
# Create a ROS Package
catkin_create_pkg turtel_commander rospy

# Create solution file
touch ~/self_driving/hw1/src/turtel_commander/src/handle_seek.py

cd ~/self_driving/hw1
catkin_make
chmod +x src/turtel_commander/src/handle_seek.py
source devel/setup.bash

# Run solution
# First terminal
roscore
# Second terminal
rosrun turtlesim turtlesim_node
# Third terminal
rosrun turtel_commander handle_seek.py
```