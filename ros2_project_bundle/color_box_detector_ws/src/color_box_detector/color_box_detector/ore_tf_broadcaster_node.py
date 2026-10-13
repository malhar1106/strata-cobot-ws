"""
ROS2 node: detects ore boxes in the RGB image, looks up depth at each
detection's centroid, back-projects to 3D using the camera intrinsics,
and broadcasts each as a TF frame (e.g. azurite_ore_1) parented to the
camera's optical frame.

This is what makes ore detections show up as axis triads in RViz, the
same way camera_link already does.

Subscribes:
  - image_topic        (sensor_msgs/Image, default /camera/image)
  - depth_topic         (sensor_msgs/Image, default /camera/depth_image)
  - camera_info_topic   (sensor_msgs/CameraInfo, default /camera/camera_info)

Publishes:
  - TF transforms named per-detection (e.g. azurite_ore_1) under camera_frame

Run:
  ros2 run color_box_detector ore_tf_broadcaster_node
"""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
import tf2_ros

from color_box_detector.detector import detect_color_boxes


class OreTFBroadcasterNode(Node):
    def __init__(self):
        super().__init__('ore_tf_broadcaster_node')

        self.declare_parameter('image_topic', '/camera/image')
        self.declare_parameter('depth_topic', '/camera/depth_image')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        # Frame the TF transforms will be parented to. Confirmed correct
        # for this scene via: ros2 topic echo /camera/camera_info --once
        # (header.frame_id was camera_optical_frame, not camera_link).
        self.declare_parameter('camera_frame', 'camera_optical_frame')
        # Sanity filter: ores only ever appear at short range on the
        # conveyor in front of the camera. A detection whose depth reads
        # farther than this is almost certainly a false-positive color
        # match on background terrain/rocks, not a real ore -- reject it
        # rather than broadcasting a wrong TF frame. Adjust to your scene's
        # actual working distance if real ores are being rejected too.
        self.declare_parameter('max_valid_depth_m', 2.2)

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        self.camera_frame = self.get_parameter('camera_frame').get_parameter_value().string_value
        self.max_valid_depth_m = self.get_parameter('max_valid_depth_m').get_parameter_value().double_value

        self.bridge = CvBridge()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Camera intrinsics -- filled in once camera_info arrives
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.latest_depth = None  # most recent depth frame, as a numpy array

        self.create_subscription(CameraInfo, camera_info_topic, self.camera_info_callback, 10)
        self.create_subscription(Image, depth_topic, self.depth_callback, 10)
        self.create_subscription(Image, image_topic, self.image_callback, 10)

        self.get_logger().info(
            f'Waiting for camera_info on {camera_info_topic}, '
            f'depth on {depth_topic}, image on {image_topic}...'
        )

    def camera_info_callback(self, msg: CameraInfo):
        # K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        if self.fx is None:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.get_logger().info(
                f'Got camera intrinsics: fx={self.fx:.1f} fy={self.fy:.1f} '
                f'cx={self.cx:.1f} cy={self.cy:.1f}'
            )

    def depth_callback(self, msg: Image):
        try:
            # Depth encoding varies by sim -- 32FC1 is meters (most common
            # in Gazebo), 16UC1 is millimeters. Adjust the divide-by-1000
            # below if your depth turns out to be 16UC1.
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'Depth conversion failed: {e}')

    def image_callback(self, msg: Image):
        if self.fx is None:
            return  # haven't got intrinsics yet
        if self.latest_depth is None:
            return  # haven't got a depth frame yet

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Image conversion failed: {e}')
            return

        detections = detect_color_boxes(frame)
        depth = self.latest_depth

        for d in detections:
            cx_px, cy_px = d['centroid']
            if not (0 <= cy_px < depth.shape[0] and 0 <= cx_px < depth.shape[1]):
                continue

            z = float(depth[cy_px, cx_px])
            # NaN/0/inf depth means no valid reading at that pixel (e.g. sky,
            # or a sensor gap) -- skip rather than broadcasting garbage.
            if not np.isfinite(z) or z <= 0.0:
                continue
            # Reject implausibly-distant detections -- almost certainly a
            # false-positive color match on terrain/rocks far behind the
            # actual conveyor, not a real ore.
            if z > self.max_valid_depth_m:
                self.get_logger().info(
                    f'REJECTED {d["name"]}: depth={z:.3f}m '
                    f'(max_valid_depth_m={self.max_valid_depth_m})'
                )
                continue
            else:
                self.get_logger().info(f'ACCEPTED {d["name"]}: depth={z:.3f}m')

            # If your depth is 16UC1 (millimeters) instead of 32FC1 (meters),
            # uncomment the next line:
            # z = z / 1000.0

            x = (cx_px - self.cx) * z / self.fx
            y = (cy_px - self.cy) * z / self.fy

            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = self.camera_frame
            t.child_frame_id = d['name']  # e.g. azurite_ore_1
            t.transform.translation.x = x
            t.transform.translation.y = y
            t.transform.translation.z = z
            # No orientation info from a color blob -- identity rotation.
            t.transform.rotation.w = 1.0

            self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = OreTFBroadcasterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
