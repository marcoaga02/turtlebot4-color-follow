This repository is a ROS2 package that contains a ROS2 node that allows the robot following the red color.
# Instructions to compile and run the node
This commands are meant to be executed from the root of the ros2 workspace.
```bash
colcon build --symlink-install
source install/setup.zsh # or .bash if using the default linux shell
ros2 run turtlebot_color_follow color_follow_node
```