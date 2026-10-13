#!/usr/bin/env python3


'''
*****************************************************************************************
*
*        		===============================================
*           		        StrataCobot (SC) Theme (eYRC 2026-27)
*        		===============================================
*
*  This script should be used to implement Task 1C of StrataCobot (SC) Theme (eYRC 2026-27).
*
*****************************************************************************************
'''

# Team ID:          1304
# Author List:      Swara Sukale,Malhar Pawle,Arman Khan,Sayali Katkar
# Filename:         task1C.py
# Functions:        quat_to_yaw, wrap_angle, ebot_nav.__init__, ebot_nav.pathcb,
#                   ebot_nav.odomcb, ebot_nav.scancb, ebot_nav.process_navigation, main
# Nodes:
#                   Publishing Topics  - [ /cmd_vel ]
#                   Subscribing Topics - [ /ebot_path, /odom, /scan, /tf, /tf_static ]


################### IMPORT MODULES #######################

import rclpy
import sys
import math
import numpy as np
import tf2_ros
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from rclpy.qos import (qos_profile_sensor_data, QoSProfile, QoSHistoryPolicy,
                        QoSReliabilityPolicy, QoSDurabilityPolicy)
from sensor_msgs.msg import LaserScan


##################### TASK CONSTANTS #######################

path_topic = '/ebot_path'
odom_topic = '/odom'
scan_topic = '/scan'
cmd_topic = '/cmd_vel'

max_linear_mps = 0.5
max_angular_rps = 1.0

# The route is held for late subscribers with TRANSIENT_LOCAL durability. A default
# (VOLATILE) subscription never sees a message sent before it existed, so this has to
# match. Reliability guessed as RELIABLE -- check with 'ros2 topic info /ebot_path -v'
# once the sim is up, and flip to BEST_EFFORT here if that's what the publisher offers.
PATH_QOS = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)

WAYPOINT_TOLERANCE_M = 0.15    # how close counts as "reached" -- tune against waypoint spacing
SLOWDOWN_DIST_M = 0.6          # start easing off the throttle this close to the target
HEADING_LOCK_RAD = 0.6         # ~34 deg; above this, rotate in place before driving forward
KP_ANGULAR = 1.4               # while driving
KP_ANGULAR_INPLACE = 1.8       # while rotating to face the target

FORWARD_CONE_DEG = 55          # half-angle of the cone checked ahead for obstacles
SAFETY_STOP_M = 0.45           # closer than this in the cone: stop and turn away, don't drive through
SAFETY_SLOW_M = 0.9            # closer than this: slow down and start steering around
AVOID_GAIN = 1.3               # how hard to steer away from the closer side


##################### HELPER FUNCTIONS #######################

def quat_to_yaw(x, y, z, w):
    '''Yaw from a quaternion, assuming the robot stays roughly level.'''
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_angle(a):
    '''Wrap an angle to (-pi, pi], so steering always turns the short way.'''
    return math.atan2(math.sin(a), math.cos(a))


##################### CLASS DEFINITION #######################

class ebot_nav(Node):
    '''
    ___CLASS___

    Description:    Class which serves the purpose to drive the eBot along the route
                    published on /ebot_path, through its waypoints, in order.
    '''

    def __init__(self):

        super().__init__(
            'ebot_nav_node',
            parameter_overrides=[rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        ############ Topic SUBSCRIPTIONS ############

        self.path_sub = self.create_subscription(Path, path_topic, self.pathcb, PATH_QOS)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odomcb, 10)
        self.scan_sub = self.create_subscription(LaserScan, scan_topic, self.scancb, qos_profile_sensor_data)

        ############ Topic PUBLISHERS ############

        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)

        ############ Constructor VARIABLES/OBJECTS ############

        control_rate = 0.05
        self.timer = self.create_timer(control_rate, self.process_navigation)

        self.route = None            # list of (x, y), taken once from pathcb()
        self.odom = None             # (x, y, yaw), from odomcb()
        self.base_frame = None       # the base's own frame id, from odom's child_frame_id
        self.obstacle = None         # dict(min_front, left_room, right_room), from scancb()

        # Lidar mount offset in the base frame: (dx, dy, dyaw). Resolved once via /tf,
        # not typed in, since the lidar is not at the base's centre.
        self.scan_offset = None
        self.scan_offset_frame = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.wp_idx = 0
        self.route_done = False
        self.done_logged = False


    def pathcb(self, data):
        self.get_logger().info(
            f'Route received: {len(data.poses)} waypoints in frame "{data.header.frame_id}"')
        self.route = [(p.pose.position.x, p.pose.position.y) for p in data.poses]


    def odomcb(self, data):
        if self.base_frame is None:
            self.base_frame = data.child_frame_id
            self.get_logger().info(f'Base frame: "{self.base_frame}"')
        p = data.pose.pose.position
        q = data.pose.pose.orientation
        yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
        self.odom = (p.x, p.y, yaw)


    def scancb(self, data):
        # Resolve the lidar's mount offset once we know the base frame. Static mount,
        # so one lookup is enough; re-tried every scan until it succeeds.
        if self.scan_offset is None and self.base_frame is not None:
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.base_frame, data.header.frame_id, rclpy.time.Time())
                t = tf.transform.translation
                q = tf.transform.rotation
                self.scan_offset = (t.x, t.y, quat_to_yaw(q.x, q.y, q.z, q.w))
                self.scan_offset_frame = data.header.frame_id
                self.get_logger().info(
                    f'Lidar offset in {self.base_frame}: dx={t.x:.3f} dy={t.y:.3f}')
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as e:
                self.get_logger().warn(f'Lidar TF not ready yet: {e}', throttle_duration_sec=2.0)

        ranges = np.asarray(data.ranges, dtype=np.float64)
        n = ranges.shape[0]
        angles = data.angle_min + np.arange(n) * data.angle_increment
        valid = np.isfinite(ranges) & (ranges >= data.range_min) & (ranges <= data.range_max)

        # Points in the lidar's own frame.
        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)

        if self.scan_offset is not None:
            dx, dy, dyaw = self.scan_offset
            c, s = math.cos(dyaw), math.sin(dyaw)
            bx = xs * c - ys * s + dx
            by = xs * s + ys * c + dy
        else:
            # Offset not resolved yet -- use raw lidar-frame points as a rough stand-in
            # until the first /tf lookup succeeds, rather than doing nothing.
            bx, by = xs, ys

        ang_base = np.arctan2(by, bx)
        dist_base = np.hypot(bx, by)

        cone = valid & (np.abs(ang_base) <= math.radians(FORWARD_CONE_DEG))
        left = cone & (ang_base > math.radians(5))
        right = cone & (ang_base < -math.radians(5))

        min_front = float(np.min(dist_base[cone])) if np.any(cone) else float(data.range_max)
        left_room = float(np.mean(dist_base[left])) if np.any(left) else float(data.range_max)
        right_room = float(np.mean(dist_base[right])) if np.any(right) else float(data.range_max)

        self.obstacle = {'min_front': min_front, 'left_room': left_room, 'right_room': right_room}


    def process_navigation(self):
        try:
            self._process_navigation_impl()
        except Exception as e:
            self.get_logger().error(f'process_navigation failed: {e}', throttle_duration_sec=2.0)
            self._publish(0.0, 0.0)


    def _process_navigation_impl(self):
        if self.route is None or self.odom is None or self.obstacle is None:
            return

        if self.route_done:
            self._publish(0.0, 0.0)
            if not self.done_logged:
                self.get_logger().info('Route complete -- holding position.')
                self.done_logged = True
            return

        if self.wp_idx >= len(self.route):
            self.route_done = True
            return

        x, y, yaw = self.odom
        tx, ty = self.route[self.wp_idx]
        dx, dy = tx - x, ty - y
        distance = math.hypot(dx, dy)

        if distance < WAYPOINT_TOLERANCE_M:
            self.wp_idx += 1
            self.get_logger().info(f'Waypoint {self.wp_idx}/{len(self.route)} reached.')
            if self.wp_idx >= len(self.route):
                self.route_done = True
                self._publish(0.0, 0.0)
                return
            tx, ty = self.route[self.wp_idx]
            dx, dy = tx - x, ty - y
            distance = math.hypot(dx, dy)

        heading_error = wrap_angle(math.atan2(dy, dx) - yaw)

        # Obstacle avoidance: bias steering away from whichever side is tighter, and
        # cut speed the closer the nearest thing ahead is.
        avoid_angular = 0.0
        speed_scale = 1.0
        min_front = self.obstacle['min_front']
        if min_front < SAFETY_STOP_M:
            # Too close to drive through -- stop and turn toward the more open side.
            side = 1.0 if self.obstacle['right_room'] > self.obstacle['left_room'] else -1.0
            self._publish(0.0, side * KP_ANGULAR_INPLACE)
            return
        elif min_front < SAFETY_SLOW_M:
            room_diff = self.obstacle['right_room'] - self.obstacle['left_room']
            avoid_angular = -AVOID_GAIN * room_diff  # steer toward the side with more room
            speed_scale = max(0.2, (min_front - SAFETY_STOP_M) / (SAFETY_SLOW_M - SAFETY_STOP_M))

        if abs(heading_error) > HEADING_LOCK_RAD:
            # Badly misaligned: rotate in place first. Driving forward while a
            # skid-steer base is this far off mostly just scrubs the wheels.
            angular = KP_ANGULAR_INPLACE * heading_error + avoid_angular
            self._publish(0.0, angular)
            return

        linear = max_linear_mps * min(1.0, distance / SLOWDOWN_DIST_M) * speed_scale
        angular = KP_ANGULAR * heading_error + avoid_angular
        self._publish(linear, angular)


    def _publish(self, linear, angular):
        msg = Twist()
        msg.linear.x = max(-max_linear_mps, min(max_linear_mps, linear))
        msg.angular.z = max(-max_angular_rps, min(max_angular_rps, angular))
        self.cmd_pub.publish(msg)


##################### FUNCTION DEFINITION #######################

def main():
    rclpy.init(args=sys.argv)
    node = rclpy.create_node('ebot_nav_process')
    node.get_logger().info('Node created: eBot navigation process')
    ebot_nav_class = ebot_nav()
    rclpy.spin(ebot_nav_class)
    ebot_nav_class.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':

    main()
