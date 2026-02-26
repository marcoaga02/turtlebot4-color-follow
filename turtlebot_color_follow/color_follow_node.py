import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
from cv_bridge import CvBridge
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy

class ColorFollowNode(Node):
    def __init__(self):
        super().__init__('color_follow_node')
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image, '/robot/oakd/rgb/preview/image_raw', self.image_callback, 10
        )
        
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.cmd_pub = self.create_publisher(Twist, '/robot/cmd_vel_unstamped', qos)

        # Tracking parameters
        self.linear_speed = 0.2
        self.angular_speed = 0.5

        self.red_lower1 = np.array([0, 120, 70])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([170, 120, 70])
        self.red_upper2 = np.array([180, 255, 255])

    def image_callback(self, msg):
        # Conversion of ROS2 Image into OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        self.get_logger().info('Frame ricevuto')

        # Coversion to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Red mask (due range)
        mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        mask = mask1 + mask2

        # Find contour
        # RETR_EXTERNAL: takes only external contours
        # CHAIN_APPROX_SIMPLE: compresses redundant points to save memory
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.get_logger().info(f'Found contours: {len(contours)}')

        twist = Twist()

        if contours:
            # Takes the bigger contour
            c = max(contours, key=cv2.contourArea)

            # Compute the centroid of the contour using geometric moments.
            # m00 is the area, m10 and m01 are the first-order moments.
            # Dividing m10/m00 and m01/m00 gives the (cx, cy) center of mass of the blob.
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"]/M["m00"])
                # cy = int(M["m01"]/M["m00"])

                # Proportional controller: error_x measures how far the blob is from the center of the frame.
                # The angular velocity is normalized between -1 and +1 and scaled by angular_speed.
                # The negative sign ensures the robot turns toward the blob:
                # if the blob is on the right (error_x > 0), angular.z is negative (clockwise in ROS).
                # The further the blob is from the center, the faster the robot turns.
                # When the blob is perfectly centered, angular.z is zero and the robot goes straight.

                # Compute the horizontal error w.r.t. the image's center
                error_x = cx - frame.shape[1] // 2

                # Proportional control to turn toward the red
                # frame.shape returns the image dimensions as a tuple (height, width, channels)
                twist.linear.x = self.linear_speed
                twist.angular.z = -float(error_x) / (frame.shape[1] // 2) * self.angular_speed
            else:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
        else:
            # if no red is found, stop the robot
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = ColorFollowNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()