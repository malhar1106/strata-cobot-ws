"""
Drives the REAL, unmodified color_box_detector_node.py through its
image_callback, using mocked rclpy/sensor_msgs/std_msgs/cv_bridge.

This proves: the node imports cleanly, wires up subscriptions/publishers
without error, correctly unpacks the incoming Image message, calls the
detector, and publishes both outputs. It does NOT prove real ROS2
topic wiring, QoS behavior, or message serialization -- confirm those
with an actual `colcon build` + `ros2 run` once ROS2 is available.
"""

import sys
import os
import cv2

# Put our mocks on the import path BEFORE rclpy/sensor_msgs/etc. would
# normally be looked up, and before the real package's path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mocks'))
sys.path.insert(0, '/home/claude/color_box_detector_ws/src/color_box_detector')

from color_box_detector.color_box_detector_node import ColorBoxDetectorNode  # noqa: E402
from sensor_msgs.msg import Image  # this resolves to our mock  # noqa: E402


def main():
    node = ColorBoxDetectorNode()

    # Build a fake incoming camera frame from your actual screenshot
    frame = cv2.imread(
        '/mnt/user-data/uploads/1789799970221_Screenshot_from_2026-09-19_12-08-59.png'
    )
    assert frame is not None, "couldn't load test image"

    fake_msg = Image()
    fake_msg.data = frame
    fake_msg.header = 'fake_header_stamp_and_frame_id'

    print('\n--- driving image_callback ---')
    node.image_callback(fake_msg)

    print('\n--- results ---')
    annotated_msgs = node.annotated_pub.published
    detection_msgs = node.detections_pub.published

    assert len(annotated_msgs) == 1, 'expected exactly one annotated image published'
    assert len(detection_msgs) == 1, 'expected exactly one detections message published'

    print(f'Annotated image published: shape={annotated_msgs[0].data.shape}')
    print(f'Detections JSON published: {detection_msgs[0].data}')

    # Save the annotated frame the node actually produced, for visual proof
    out_path = '/home/claude/ros2_mock_test/node_output.png'
    cv2.imwrite(out_path, annotated_msgs[0].data)
    print(f'\nSaved node output to {out_path}')
    print('\nALL CHECKS PASSED')


if __name__ == '__main__':
    main()
