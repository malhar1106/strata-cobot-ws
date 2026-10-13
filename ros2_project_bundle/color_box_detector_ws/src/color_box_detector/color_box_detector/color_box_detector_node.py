"""
ROS2 node: subscribes to a camera image topic, runs HSV color-box detection,
and publishes:
  - /color_box_detector/image_annotated  (sensor_msgs/Image)   -- for rviz/rqt_image_view
  - /color_box_detector/detections       (std_msgs/String, JSON) -- for other nodes

Run:
  ros2 run color_box_detector color_box_detector_node --ros-args -p image_topic:=/camera/image_raw
"""

import json

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

from color_box_detector.detector import detect_color_boxes, draw_detections


class ColorBoxDetectorNode(Node):
    def __init__(self):
        super().__init__('color_box_detector_node')

        # Parameters so the topic name isn't hardcoded -- override at launch
        # with --ros-args -p image_topic:=/your/topic
        self.declare_parameter('image_topic', '/camera/image_raw')
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value

        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(
            Image, image_topic, self.image_callback, 10)

        self.annotated_pub = self.create_publisher(
            Image, '/color_box_detector/image_annotated', 10)
        self.detections_pub = self.create_publisher(
            String, '/color_box_detector/detections', 10)

        self.get_logger().info(f'Subscribed to {image_topic}, waiting for frames...')

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge conversion failed: {e}')
            return

        detections = detect_color_boxes(frame)
        annotated = draw_detections(frame, detections)

        # Publish annotated image, preserving the original frame_id/stamp
        out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        out_msg.header = msg.header
        self.annotated_pub.publish(out_msg)

        # Publish detections as JSON. This is the simplest option to get
        # running; if other nodes need to consume this programmatically,
        # switch to vision_msgs/Detection2DArray instead (see notes below).
        payload = json.dumps([
            {
                'name': d['name'],
                'color': d['color'],
                'bbox': d['bbox'],
                'centroid': d['centroid'],
                'area': d['area'],
            }
            for d in detections
        ])
        self.detections_pub.publish(String(data=payload))

        if detections:
            self.get_logger().debug(f'Found {len(detections)} boxes')


def main(args=None):
    rclpy.init(args=args)
    node = ColorBoxDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
