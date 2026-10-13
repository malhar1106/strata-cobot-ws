#!/usr/bin/env python3


'''
*****************************************************************************************
*
*        		===============================================
*           		        StrataCobot (SC) Theme (eYRC 2026-27)
*        		===============================================
*
*  This script should be used to implement Task 1A of StrataCobot (SC) Theme (eYRC 2026-27).
*
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:          1304
# Author List:		Swara Sukale,Malhar Pawle,Arman Khan,Sayali Katkar
# Filename:		    task1a_boilerplate.py
# Functions:
#			        detect_ores, ore_tf.__init__, ore_tf.depthimagecb, ore_tf.colorimagecb,
#                   ore_tf.caminfocb, ore_tf.process_image, ore_tf.match_or_assign_id, main
# Nodes:		    Add your publishing and subscribing node
#                   Example:
#			        Publishing Topics  - [/tf]
#                               Subscribing Topics - [/camera/camera/color/image_raw,
#                                                    /camera/camera/aligned_depth_to_color/image_raw,
#                                                    /camera/camera/color/camera_info,
#                                                    /tf, /tf_static]


################### IMPORT MODULES #######################

import rclpy
import os
import sys
import cv2
import math
import tf2_ros
import numpy as np
from rclpy.node import Node
from rclpy.parameter import Parameter
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped, PointStamped
from sensor_msgs.msg import CameraInfo, Image
from tf2_geometry_msgs import do_transform_point


##################### TASK CONSTANTS #######################

# Two ores of each type are spawned - six in all - told apart by an id of 1 or 2.
ore_types = ['azurite_ore', 'malachite_ore', 'vanadinite_ore']

# The RealSense topics. The depth image is ALIGNED to the colour image.
color_topic = '/camera/camera/color/image_raw'
depth_topic = '/camera/camera/aligned_depth_to_color/image_raw'
camera_info_topic = '/camera/camera/color/camera_info'

# The parent frame every ore transform is published against.
base_frame = 'base_link'

# HSV colour ranges per ore type -- sampled from a saved frame, not copied from
# a tutorial. Re-sample against your own camera/lighting if detection is off:
# print(hsv[y, x]) at a pixel you know sits on the ore.
HSV_RANGES = {
    'azurite_ore':    ((100, 100, 50), (130, 255, 255)),   # blue
    'malachite_ore':  ((40, 60, 40), (85, 255, 255)),      # green
    'vanadinite_ore': ((0, 200, 150), (18, 255, 255)),     # orange
}
MIN_CONTOUR_AREA = 100  # pixels; drops noise smaller than a real ore
MIN_EXTENT = 0.65      # contour area / bounding-box area; rocks are irregular, cubes are boxy
MAX_PER_TYPE = 2       # two ores of each type are spawned

# The ore cube's half-height (metres). The camera sees the TOP face, but the
# ore is named by its centre, so this much is subtracted off the transformed
# Z (assuming base_link's Z axis points up) to correct for that offset.
# Measure your actual ore model's height and update this -- this is a
# starting estimate, not a measured value.
ORE_HALF_HEIGHT_M = 0.025

# How far (metres, in base_frame) a new detection can be from a previously
# tracked ore of the same type and still count as "the same ore". Tune to
# your ores' actual spacing -- too large and two distinct ores merge into
# one id, too small and a slightly-moved ore looks new.
ID_MATCH_DIST_M = 0.07
TEAM_ID = 1304


##################### FUNCTION DEFINITIONS #######################

def detect_ores(image):
    '''
    Description:    Function to detect the ores present in a colour image frame and
                    return the pixel location and the type of each one found.

    Args:
        image                   (Image):    Input colour image frame received from the camera topic

    Returns:
        center_ore_list         (list):     Center pixel (cX, cY) of every ore detected in the frame
        ore_type_list           (list):     Type of each ore detected, taken from 'ore_types'
    '''

    center_ore_list = []
    ore_type_list = []

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    kernel = np.ones((3, 3), np.uint8)

    for ore_type, (lower, upper) in HSV_RANGES.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < MIN_CONTOUR_AREA:
                continue
            x, y, w, h = cv2.boundingRect(c)
            (_, _), (rw, rh), _ = cv2.minAreaRect(c)
            if rw == 0 or rh == 0:
                continue
            extent = area / float(rw * rh)          # rotation-invariant
            aspect = max(rw, rh) / float(min(rw, rh))
            # Cubes are compact and roughly square; rock patches are ragged.
            if extent < MIN_EXTENT or aspect > 2.0:
                continue
            M = cv2.moments(c)
            if M['m00'] == 0:
                continue
            candidates.append((area, int(M['m10'] / M['m00']),
                               int(M['m01'] / M['m00']), (x, y, w, h)))

        # Only MAX_PER_TYPE ores of each type exist: keep the biggest ones.
        candidates.sort(key=lambda t: t[0], reverse=True)
        for area, cX, cY, (x, y, w, h) in candidates[:MAX_PER_TYPE]:
            center_ore_list.append((cX, cY))
            ore_type_list.append(ore_type)

            # Draw while developing -- harmless to leave in for the run too.
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)

    return center_ore_list, ore_type_list


def draw_labels(image, items):
    '''
    Description:    Write each ore's published frame name in a tidy column to the left of
                    the ores, with a leader line to the ore, so labels never overlap.

    Args:
        image   (ndarray):  Annotated frame, drawn on in place
        items   (list):     (cX, cY, name) per ore; name is the transform's child frame id
    '''

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, step = 0.4, 16
    items = sorted(items, key=lambda t: t[1])
    x_edge = max(min(i[0] for i in items) - 45, 150)
    y0 = int(sum(i[1] for i in items) / len(items) - (len(items) - 1) * step / 2)
    y0 = max(y0, 14)
    for k, (cX, cY, name) in enumerate(items):
        y = y0 + k * step
        (tw, th), _ = cv2.getTextSize(name, font, scale, 1)
        x = x_edge - tw
        cv2.line(image, (x_edge + 4, y - th // 2), (cX, cY), (0, 255, 255), 1)
        cv2.putText(image, name, (x, y), font, scale, (0, 0, 0), 3)
        cv2.putText(image, name, (x, y), font, scale, (255, 255, 255), 1)


##################### CLASS DEFINITION #######################

class ore_tf(Node):
    '''
    ___CLASS___

    Description:    Class which serves the purpose to detect the ores in the cell and
                    broadcast a transform for each one.
    '''

    def __init__(self):
        '''
        Description:    Initialization of class ore_tf
        '''

        super().__init__('ore_tf_publisher',
                         parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)])                                            # registering node

        ############ Topic SUBSCRIPTIONS ############

        self.color_cam_sub = self.create_subscription(Image, color_topic, self.colorimagecb, 10)
        self.depth_cam_sub = self.create_subscription(Image, depth_topic, self.depthimagecb, 10)
        self.cam_info_sub = self.create_subscription(CameraInfo, camera_info_topic, self.caminfocb, 10)

        ############ Constructor VARIABLES/OBJECTS ############

        image_processing_rate = 0.2                                                     # rate of time to process image (seconds)
        self.bridge = CvBridge()                                                        # initialise CvBridge object for image conversion
        self.tf_buffer = tf2_ros.buffer.Buffer()                                        # buffer time used for listening transforms
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.br = tf2_ros.TransformBroadcaster(self)                                    # object as transform broadcaster to send transform wrt some frame_id
        self.timer = self.create_timer(image_processing_rate, self.process_image)       # creating a timer based function which gets called on every 0.2 seconds (as defined by 'image_processing_rate' variable)

        self.cv_image = None                                                            # colour raw image variable (from colorimagecb())
        self.depth_image = None                                                         # depth image variable (from depthimagecb())
        self.cam_info = None                                                            # camera intrinsics variable (from caminfocb())

        # Persistent per-type id tracking: each entry remembers the id (1 or 2)
        # and the last known base_frame position of that physical ore, so the
        # same ore keeps the same id for the whole run even though
        # 'detect_ores' returns detections in an arbitrary order every frame.
        self.tracked_ores = {t: [] for t in ore_types}
        self.image_saved = False                                                        # detection image is written once, when all six ores are labelled


    def depthimagecb(self, data):
        '''
        Description:    Callback function for the aligned depth camera topic.
                        Use this function to receive the depth image and convert it to a CV2 image.

        Args:
            data (Image):    Input depth image frame received from the aligned depth camera topic

        Returns:
        '''

        try:
            depth = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        except CvBridgeError as e:
            self.get_logger().error(f'Depth conversion failed: {e}')
            return

        # 16UC1 is millimetres -- convert to metres. 32FC1 is already metres.
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 1000.0

        self.depth_image = depth


    def colorimagecb(self, data):
        '''
        Description:    Callback function for the colour camera raw topic.
                        Use this function to receive the raw image and convert it to a CV2 image.

        Args:
            data (Image):    Input coloured raw image frame received from the image_raw camera topic

        Returns:
        '''

        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'Color conversion failed: {e}')
            return

        # The optical frame the colour image (and therefore the depth image,
        # since it's aligned to colour) is expressed in -- needed for the TF
        # lookup in process_image.
        self.color_frame_id = data.header.frame_id


    def caminfocb(self, data):
        '''
        Description:    Callback function for the camera info topic.
                        Use this function to receive the camera's intrinsic parameters.

        Args:
            data (CameraInfo):    Camera calibration published by the camera

        Returns:
        '''

        # k = [fx, 0, cx, 0, fy, cy, 0, 0, 1] -- read once, camera_info is static.
        if self.cam_info is None:
            self.cam_info = data


    def match_or_assign_id(self, ore_type, position, claimed_this_frame):
        '''
        Description:    Match a new detection to a previously tracked ore of the
                        same type by nearest position, or assign it a fresh id
                        (1 or 2) if it doesn't match either existing one.

        Args:
            ore_type            (str):      one of 'ore_types'
            position            (ndarray):  [x, y, z] of this detection in base_frame
            claimed_this_frame  (set):      ids of this ore_type already used THIS frame,
                                            so two detections in the same frame can't
                                            both claim the same id

        Returns:
            id (int or None):  1 or 2, or None if both ids of this type are
                                already claimed this frame and this detection
                                didn't match either by distance
        '''

        tracked = self.tracked_ores[ore_type]

        best_entry = None
        best_dist = float('inf')
        for entry in tracked:
            if entry['id'] in claimed_this_frame:
                continue
            dist = float(np.linalg.norm(entry['position'] - position))
            if dist < best_dist:
                best_dist = dist
                best_entry = entry

        if best_entry is not None and best_dist < ID_MATCH_DIST_M:
            best_entry['position'] = position
            return best_entry['id']

        # No close match -- this is a new ore of this type (or the very first
        # sighting of it). Assign whichever id (1 or 2) isn't taken yet.
        used_ids = {e['id'] for e in tracked}
        for candidate_id in (1, 2):
            if candidate_id not in used_ids:
                tracked.append({'id': candidate_id, 'position': position})
                return candidate_id

        # Both ids already exist for this type and neither matched closely
        # enough -- likely a false-positive detection. Drop it rather than
        # overwrite a correctly-tracked ore.
        return None


    def process_image(self):
        '''
        Description:    Timer function used to detect the ores and publish a transform for
                        each one on its estimated position.

        Args:
        Returns:
        '''

        if self.cv_image is None or self.depth_image is None or self.cam_info is None:
            return

        frame = self.cv_image.copy()
        centers, types = detect_ores(frame)

        fx = self.cam_info.k[0]
        fy = self.cam_info.k[4]
        cx_intr = self.cam_info.k[2]
        cy_intr = self.cam_info.k[5]
        optical_frame = self.cam_info.header.frame_id

        claimed_this_frame = {t: set() for t in ore_types}
        labelled = []

        for (cX, cY), ore_type in zip(centers, types):
            if not (0 <= cY < self.depth_image.shape[0] and 0 <= cX < self.depth_image.shape[1]):
                continue

            y0, y1 = max(cY - 4, 0), min(cY + 5, self.depth_image.shape[0])
            x0, x1 = max(cX - 4, 0), min(cX + 5, self.depth_image.shape[1])
            patch = self.depth_image[y0:y1, x0:x1]
            valid = patch[np.isfinite(patch) & (patch > 0.0)]
            if valid.size == 0:
                self.get_logger().warn(f'No valid depth for {ore_type} at pixel ({cX},{cY})',
                                       throttle_duration_sec=2.0)
                continue
            z = float(np.median(valid))

            x = (cX - cx_intr) * z / fx
            y = (cY - cy_intr) * z / fy

            point_camera = PointStamped()
            point_camera.header.frame_id = optical_frame
            point_camera.header.stamp = self.get_clock().now().to_msg()            # camera_optical_frame in this sim is x-forward, y-left, z-up (not true optical).

            point_camera.point.x = x
            point_camera.point.y = y
            point_camera.point.z = z
            try:
                tf = self.tf_buffer.lookup_transform(
                    base_frame, optical_frame, rclpy.time.Time())
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as e:
                self.get_logger().warn(f'TF lookup {optical_frame}->{base_frame} failed: {e}')
                continue

            point_base = do_transform_point(point_camera, tf)

            # Top-face -> centre-of-ore correction (assumes base_frame's Z
            # points up -- flip the sign if that's not the case for this rig).
            corrected_z = point_base.point.z - ORE_HALF_HEIGHT_M

            position = np.array([point_base.point.x, point_base.point.y, corrected_z])

            ore_id = self.match_or_assign_id(ore_type, position, claimed_this_frame[ore_type])
            if ore_id is None:
                self.get_logger().warn(f'Dropped {ore_type} at {position.round(3)}: no free id',
                                       throttle_duration_sec=2.0)
                continue
            claimed_this_frame[ore_type].add(ore_id)

            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = base_frame
            t.child_frame_id = f'{ore_type}_{ore_id}'
            t.transform.translation.x = float(position[0])
            t.transform.translation.y = float(position[1])
            t.transform.translation.z = float(corrected_z)
            t.transform.rotation.w = 1.0

            self.br.sendTransform(t)
            labelled.append((cX, cY, t.child_frame_id))

        if labelled:
            draw_labels(frame, labelled)
            if len(labelled) == len(ore_types) * 2 and not self.image_saved:
                path = os.path.abspath(f'SC#{TEAM_ID}_task1A_detection.png')
                if cv2.imwrite(path, frame):
                    self.image_saved = True
                    self.get_logger().info(f'Saved detection image: {path}')

        try:
            cv2.imshow('Ore Detections', frame)
            cv2.waitKey(1)
        except cv2.error:
            pass  # no display available; detection and TF still run


##################### FUNCTION DEFINITION #######################

def main():
    '''
    Description:    Main function which creates a ROS node and spins around for the ore_tf
                    class to perform its task
    '''

    rclpy.init(args=sys.argv)                                       # initialisation

    node = rclpy.create_node('ore_tf_process')                      # creating ROS node

    node.get_logger().info('Node created: Ore tf process')          # logging information

    ore_tf_class = ore_tf()                                         # creating a new object for class 'ore_tf'

    rclpy.spin(ore_tf_class)                                        # spining on the object to make it alive in ROS 2 DDS

    ore_tf_class.destroy_node()                                     # destroy node after spin ends

    rclpy.shutdown()                                                # shutdown process


if __name__ == '__main__':

    main()
