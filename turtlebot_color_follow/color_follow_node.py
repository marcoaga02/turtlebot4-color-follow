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

        # Parametri di tracking
        self.linear_speed = 0.2      # velocità avanti
        self.angular_speed = 0.5     # velocità di rotazione
        self.red_lower1 = np.array([0, 120, 70])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([170, 120, 70])
        self.red_upper2 = np.array([180, 255, 255])

    def image_callback(self, msg):
        # Converti ROS Image in OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        self.get_logger().info('Frame ricevuto')

        # Converti in HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Maschera per il rosso (due range)
        mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        mask = mask1 + mask2

        # Trova contorni
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.get_logger().info(f'Contorni trovati: {len(contours)}')

        twist = Twist()

        if contours:
            # Prendi il contorno più grande
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])

                # Calcola errore orizzontale rispetto al centro dell'immagine
                error_x = cx - frame.shape[1] // 2

                # Controllo proporzionale per ruotare verso il rosso
                twist.linear.x = self.linear_speed
                twist.angular.z = -float(error_x) / (frame.shape[1] // 2) * self.angular_speed
            else:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
        else:
            # Se non trova rosso, ferma il robot
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        # Pubblica comando
        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = ColorFollowNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()