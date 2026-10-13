#!/usr/bin/env python3
'''
*****************************************************************************************
*
*        		===============================================
*           		        StrataCobot (SC) Theme (eYRC 2026-27)
*        		===============================================
*
*  This script should be used to implement the manipulation part of Task 2A of
*  StrataCobot (SC) Theme (eYRC 2026-27): pick each ore off the conveyor with the
*  magnet and drop it in the container for its kind.
*
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:          1304
# Author List:      Swara Sukale,Malhar Pawle,Arman Khan,Sayali Katkar
# Filename:         task2A_manipulation.py
# Functions:        solve_pre_pose, numeric_ik_tilt, fk, ... ore_sorter.run
# Nodes:
#   Publishing Topics   - [/delta_twist_cmds, /delta_joint_cmds]
#   Subscribing Topics  - [/joint_states, /tcp_pose_raw, /arm_status,
#                          /ur7e/end_effector/force_magnitude, /tf, /tf_static]
#   Service Clients     - [/magnet, /ur_arm_controller/switch_controller,
#                          /ur_arm_controller/unlock_protective_stop]


################### IMPORT MODULES #######################

import sys
import math
import time
import threading
from collections import deque

import numpy as np
import rclpy
import tf2_ros
from rclpy.node import Node
from rclpy.parameter import Parameter
from control_msgs.msg import JointJog
from controller_manager_msgs.srv import SwitchController
from std_srvs.srv import Trigger, SetBool
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32, Float32


##################### TASK CONSTANTS #######################

base_frame = 'base_link'
ore_kinds = ['azurite', 'malachite', 'vanadinite']     # frames are <kind>_ore_<n>
MAX_ORE_INDEX = 4                                       # look for <kind>_ore_1 .. _4

# ---- CONTAINERS (on the cargo rover, arm's left) ---------------------------
# !!! PLACEHOLDERS -- MEASURE THEM. They are NOT confirmed. The XY values are
# the three y=0.528 waypoints from your Task 1B path, in the order the task
# image shows the containers (azurite far, vanadinite near the camera).
# Measure each one: in RViz (Fixed Frame base_link) add the PointCloud2 on
# /camera/camera/depth/color/points, use "Publish Point" on the middle of each
# container's floor, and read it with `ros2 topic echo /clicked_point`.
# For CONTAINER_RIM_Z click the top edge of a container wall.
CONTAINER_XY = {
    'azurite':    (0.0078, 0.7191),
    'malachite':  (-0.3421, 0.7504),
    'vanadinite': (-0.6149,0.659),
}
CONTAINER_RIM_Z = -0.030        # metres in base_link, top edge of the container walls

# ---- ORE GEOMETRY (from the Magnetic Tool page: 101.6 x 101.6 x 76.2 mm) ----
ORE_HEIGHT_M = 0.0762
ORE_HALF_HEIGHT_M = ORE_HEIGHT_M / 2.0     # MUST match ORE_HALF_HEIGHT_M in task2A_perception.py
GRASP_OFFSET_XY = (0.0, 0.0)               # fine trim if the magnet lands off-centre on the ore

# ---- HEIGHTS ---------------------------------------------------------------
PRE_GRASP_CLEAR = 0.14                     # TCP height above the ore's top face before descending
RELEASE_CLEAR = 0.14                       # TCP height above the rim at release (ore hangs 76 mm below TCP)
DROP_APPROACH_EXTRA = 0.12                 # extra height for the approach to the container

# ---- ARM / CONTROL ---------------------------------------------------------
controller_manager_ns = '/ur_arm_controller'
twist_controller = 'delta_twist_controller'
joint_controller = 'delta_joint_controller'
joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
               'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

# Input caps on the servo: |linear| 0.15 m/s, |angular| 0.35 rad/s, each joint 0.35 rad/s.
# A message over a cap is DISCARDED whole, so we stay well under.
CAP_LINEAR = 0.13
CAP_ANGULAR = 0.20
CAP_JOINT = 0.32
KP_JOINT = 1.2
JOINT_TOL = 0.015                          # rad
JOINT_RAMP_S = 1.0

# Transit pose: tool-down, elbow mid-range (the docs' "unfold first" idea), in front of the arm.
HOME = np.array([0.0, -1.20, -1.60, -1.90, 1.57, 1.57])
ELBOW_HARD_LIMIT = 2.75                    # URDF limit is 2.7925 rad
LIFT_RANGE = (-3.0, -0.8)                  # keeps the IK on the "elbow up, reaching out" family

APPROACH_SPEED = 0.07                      # m/s, coarse vertical moves
CONTACT_SPEED = 0.015                      # m/s, final descent onto the ore (coast is only a few mm)
LIFT_SPEED = 0.09
CONTACT_FORCE_N = 0.4                      # force_magnitude reads ~0 in free air
HOLD_FORCE_RANGE = (0.8, 2.5)              # holding one ore reads ~1.47 N
MAX_OVERSHOOT_BELOW_TOP = 0.04             # give up looking for contact this far below the expected top
KP_TILT = 1.5                              # holds the tool pointing straight down
KP_HOLD_XY = 1.0

MAX_ATTEMPTS = 2                           # tries per ore
CRITICAL_CODES = {20, 21, 22, 23}
UNRECOVERABLE_CODE = 92
ZD = np.array([0.0, 0.0, -1.0])            # desired tool axis: straight down


##################### KINEMATICS (UR7e, from your Task 1B) #######################

D1, A2, A3, D4, D5, D6 = 0.1625, -0.425, -0.3922, 0.1333, 0.0997, 0.0996
DH_D = [D1, 0, 0, D4, D5, D6]
DH_A = [0, A2, A3, 0, 0, 0]
DH_ALPHA = [math.pi / 2, 0, 0, math.pi / 2, -math.pi / 2, 0]


def _rpy_to_R(r, p, y):
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def _fixed_T(rpy, xyz):
    T = np.eye(4)
    T[:3, :3] = _rpy_to_R(*rpy)
    T[:3, 3] = xyz
    return T


T_WRIST3_TCP = (
    _fixed_T((0, -math.pi / 2, -math.pi / 2), (0, 0, 0)) @
    _fixed_T((math.pi / 2, 0, math.pi / 2), (0, 0, 0)) @
    _fixed_T((0, 0, 0), (0, 0, 0.02041))
)


def dh_transform(theta, d, a, alpha):
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([[ct, -st * ca, st * sa, a * ct],
                     [st, ct * ca, -ct * sa, a * st],
                     [0, sa, ca, d],
                     [0, 0, 0, 1]])


def fk(q):
    '''Returns (tcp position, tcp rotation matrix) in base_link for joint angles q.'''
    T = np.eye(4)
    for i in range(6):
        T = T @ dh_transform(q[i], DH_D[i], DH_A[i], DH_ALPHA[i])
    T = T @ T_WRIST3_TCP
    return T[:3, 3].copy(), T[:3, :3].copy()


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def numeric_ik_tilt(target_xyz, seed, iters=80, w_t=0.3):
    '''
    Damped least squares IK for TCP POSITION + tool pointing straight down.
    Yaw about the tool axis is left free (the magnet is a disc).
    Returns (q, position error, tilt error).
    '''
    q = np.array(seed, dtype=float)
    target = np.array(target_xyz, dtype=float)
    for _ in range(iters):
        p, R = fk(q)
        z = R[:, 2]
        c = w_t * np.cross(z, ZD)
        e = np.concatenate([target - p, -c])
        if np.linalg.norm(np.concatenate([target - p, c])) < 1e-5:
            break
        J = np.zeros((6, 6))
        h = 1e-6
        for j in range(6):
            qq = q.copy()
            qq[j] += h
            pp, RR = fk(qq)
            J[:3, j] = (pp - p) / h
            J[3:, j] = (w_t * np.cross(RR[:, 2], ZD) - c) / h
        A = J.T @ J + 1e-4 * np.eye(6)
        q = q + 0.7 * np.linalg.solve(A, J.T @ e)
    p, R = fk(q)
    return q, float(np.linalg.norm(target - p)), float(np.linalg.norm(np.cross(R[:, 2], ZD)))


def solve_pre_pose(target_xyz, ref):
    '''
    Joint angles that put the TCP at target_xyz, tool straight down, on the arm
    branch closest to 'ref' (elbow-up family, elbow inside its hardware limit).
    Returns None if no valid configuration exists.
    '''
    ref = np.array(ref, dtype=float)
    best = None
    for pan in np.linspace(-math.pi, math.pi, 9):
        for lift in (-2.5, -1.6):
            for w1 in (-1.9, -0.7):
                seed = [pan, lift, -1.6, w1, 1.57, ref[5]]
                q, ep, eo = numeric_ik_tilt(target_xyz, seed)
                if ep > 1e-3 or eo > 0.02:
                    continue
                for i in (0, 1, 3, 4, 5):
                    q[i] = ref[i] + wrap(q[i] - ref[i])    # nearest 2*pi copy to the reference
                q[2] = wrap(q[2])                          # the elbow cannot wrap
                if not (-ELBOW_HARD_LIMIT < q[2] < -0.3):
                    continue
                if not (LIFT_RANGE[0] < q[1] < LIFT_RANGE[1]):
                    continue
                if q[4] < 0.5:
                    continue
                cost = float(np.abs(q - ref).sum())
                if best is None or cost < best[0]:
                    best = (cost, q.copy())
    return None if best is None else best[1]


##################### CLASS DEFINITION #######################

class ArmFault(Exception):
    '''Raised when /arm_status latches a critical fault.'''


class ArmFatal(Exception):
    '''Raised for the unrecoverable status code.'''


class ore_sorter(Node):
    '''
    ___CLASS___

    Description:    Reads the ore frames published by task2A_perception.py and picks each
                    ore with the magnet, then drops it in the container for its kind.
                    The sequence runs in a worker thread; a 50 Hz timer publishes the
                    current velocity command so the servo never sees the stream stall.
    '''

    def __init__(self):
        super().__init__(
            'ore_sorter_node',
            parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)])

        self.twist_pub = self.create_publisher(TwistStamped, '/delta_twist_cmds', 10)
        self.joint_pub = self.create_publisher(JointJog, '/delta_joint_cmds', 10)

        self.create_subscription(JointState, '/joint_states', self.jointstatecb, 50)
        self.create_subscription(PoseStamped, '/tcp_pose_raw', self.tcpposecb, 20)
        self.create_subscription(Int32, '/arm_status', self.statuscb, 10)
        self.create_subscription(Float32, '/ur7e/end_effector/force_magnitude', self.forcecb, 50)

        self.switch_cli = self.create_client(
            SwitchController, f'{controller_manager_ns}/switch_controller')
        self.unlock_cli = self.create_client(
            Trigger, f'{controller_manager_ns}/unlock_protective_stop')
        self.magnet_cli = self.create_client(SetBool, '/magnet')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.q = None
        self.tcp = None
        self.tcp_R = None
        self.status = None
        self.force = 0.0
        self.force_hist = deque(maxlen=60)          # ~0.6 s at 100 Hz
        self.active = None                          # 'twist' or 'joint'
        self.cmd = None                             # (kind, 6 values, wall time)
        self._last_status = None

        self.create_timer(0.02, self.publish_cmd)
        self.worker = threading.Thread(target=self.run, daemon=True)
        self.worker.start()

    ############ CALLBACKS ############

    def jointstatecb(self, data):
        try:
            self.q = np.array([data.position[data.name.index(j)] for j in joint_names])
        except ValueError:
            pass

    def tcpposecb(self, data):
        p, o = data.pose.position, data.pose.orientation
        self.tcp = np.array([p.x, p.y, p.z])
        qx, qy, qz, qw = o.x, o.y, o.z, o.w
        self.tcp_R = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])

    def statuscb(self, data):
        self.status = data.data
        if self.status != self._last_status:
            self.get_logger().info(f'arm_status -> {self.status}')
            self._last_status = self.status

    def forcecb(self, data):
        self.force = float(data.data)
        self.force_hist.append(self.force)

    ############ COMMAND STREAM ############

    def set_cmd(self, kind, values):
        self.cmd = (kind, list(values), time.monotonic())

    def stop(self):
        self.cmd = None

    def publish_cmd(self):
        cmd = self.cmd
        if cmd is None:
            return
        kind, v, stamp = cmd
        if time.monotonic() - stamp > 0.2:      # worker stalled: let the stream lapse
            return
        if kind == 'twist' and self.active == 'twist':
            m = TwistStamped()
            m.header.stamp = self.get_clock().now().to_msg()
            m.header.frame_id = base_frame
            m.twist.linear.x, m.twist.linear.y, m.twist.linear.z = (float(x) for x in v[:3])
            m.twist.angular.x, m.twist.angular.y, m.twist.angular.z = (float(x) for x in v[3:])
            self.twist_pub.publish(m)
        elif kind == 'joint' and self.active == 'joint':
            m = JointJog()
            m.header.stamp = self.get_clock().now().to_msg()
            m.header.frame_id = base_frame
            m.joint_names = list(joint_names)
            m.velocities = [float(x) for x in v]
            self.joint_pub.publish(m)

    ############ SERVICE HELPERS (called from the worker thread) ############

    def call(self, client, request, timeout=5.0):
        fut = client.call_async(request)
        t0 = time.monotonic()
        while not fut.done():
            if time.monotonic() - t0 > timeout:
                return None
            time.sleep(0.01)
        return fut.result()

    def use_controller(self, kind):
        if self.active == kind:
            return
        self.stop()
        req = SwitchController.Request()
        req.activate_controllers = [twist_controller if kind == 'twist' else joint_controller]
        r = self.call(self.switch_cli, req)
        if r is None or not r.ok:
            raise RuntimeError(f'could not switch to {kind} controller')
        self.active = kind
        time.sleep(0.2)

    def magnet(self, on):
        r = self.call(self.magnet_cli, SetBool.Request(data=bool(on)))
        if r is None or not r.success:
            self.get_logger().error(f'magnet {"on" if on else "off"} failed: {r}')
        else:
            self.get_logger().info(f'magnet {"on" if on else "off"}')

    def check(self):
        if self.status == UNRECOVERABLE_CODE:
            raise ArmFatal('arm status 92: unrecoverable')
        if self.status in CRITICAL_CODES:
            raise ArmFault(f'arm status {self.status}')

    ############ MOTION PRIMITIVES ############

    def tilt_hold(self):
        '''Angular velocity that keeps the tool axis pointing straight down.'''
        if self.tcp_R is None:
            return np.zeros(3)
        w = KP_TILT * np.cross(self.tcp_R[:, 2], ZD)
        n = np.linalg.norm(w)
        if n > CAP_ANGULAR:
            w *= CAP_ANGULAR / n
        return w

    def move_joints(self, target, timeout=60.0):
        '''Joint-space move: all joints scaled together so none exceeds the cap.'''
        self.use_controller('joint')
        target = np.array(target, dtype=float)
        t0 = time.monotonic()
        while True:
            self.check()
            d = np.array([target[i] - self.q[i] if i == 2 else wrap(target[i] - self.q[i])
                          for i in range(6)])
            if np.max(np.abs(d)) < JOINT_TOL:
                break
            if time.monotonic() - t0 > timeout:
                self.stop()
                raise RuntimeError('joint move timed out')
            ramp = min(1.0, (time.monotonic() - t0) / JOINT_RAMP_S)
            v = KP_JOINT * d
            m = np.max(np.abs(v))
            if m > CAP_JOINT:
                v *= CAP_JOINT / m
            self.set_cmd('joint', v * ramp)
            time.sleep(0.02)
        self.stop()
        time.sleep(0.3)

    def move_vertical(self, target_z, speed, timeout=40.0, tol=0.004):
        '''Straight vertical move of the TCP, holding XY and the tool-down attitude.'''
        self.use_controller('twist')
        x0, y0 = self.tcp[0], self.tcp[1]
        t0 = time.monotonic()
        while True:
            self.check()
            ez = target_z - self.tcp[2]
            if abs(ez) < tol:
                break
            if time.monotonic() - t0 > timeout:
                self.stop()
                raise RuntimeError('vertical move timed out')
            vz = float(np.clip(1.5 * ez, -speed, speed))
            if abs(vz) < 0.006:
                vz = math.copysign(0.006, ez)
            ramp = min(1.0, (time.monotonic() - t0) / 1.0)
            vx = float(np.clip(KP_HOLD_XY * (x0 - self.tcp[0]), -0.02, 0.02))
            vy = float(np.clip(KP_HOLD_XY * (y0 - self.tcp[1]), -0.02, 0.02))
            self.set_cmd('twist', [vx, vy, vz * ramp, *self.tilt_hold()])
            time.sleep(0.02)
        self.stop()
        time.sleep(0.3)

    def descend_to_contact(self, top_z):
        '''Come down onto the ore's top face, energise just before touching, stop on contact.'''
        self.move_vertical(top_z + 0.025, APPROACH_SPEED)
        self.magnet(True)
        self.use_controller('twist')
        x0, y0 = self.tcp[0], self.tcp[1]
        t0 = time.monotonic()
        while True:
            self.check()
            recent = list(self.force_hist)[-3:]
            f = float(np.mean(recent)) if recent else 0.0
            if f > CONTACT_FORCE_N:
                self.get_logger().info(f'contact at z={self.tcp[2]:.3f} (expected top {top_z:.3f}), '
                                       f'force {f:.2f} N')
                break
            if self.tcp[2] <= top_z - MAX_OVERSHOOT_BELOW_TOP:
                self.get_logger().warn('no contact detected; stopping the descent')
                break
            if time.monotonic() - t0 > 20.0:
                self.get_logger().warn('contact search timed out')
                break
            vx = float(np.clip(KP_HOLD_XY * (x0 - self.tcp[0]), -0.02, 0.02))
            vy = float(np.clip(KP_HOLD_XY * (y0 - self.tcp[1]), -0.02, 0.02))
            self.set_cmd('twist', [vx, vy, -CONTACT_SPEED, *self.tilt_hold()])
            time.sleep(0.02)
        self.stop()
        time.sleep(0.5)

    def holding_ore(self):
        time.sleep(0.6)
        if not self.force_hist:
            return False
        f = float(np.mean(self.force_hist))
        self.get_logger().info(f'force after lift: {f:.2f} N')
        return HOLD_FORCE_RANGE[0] < f < HOLD_FORCE_RANGE[1]

    ############ PERCEPTION INPUT ############

    def scan_ores(self, window_s=4.0, timeout_s=40.0):
        '''Collect the ore frames the perception node is publishing right now.'''
        found = {}
        t_start = time.monotonic()
        t_window = None
        while time.monotonic() - t_start < timeout_s:
            now = self.get_clock().now()
            for kind in ore_kinds:
                for n in range(1, MAX_ORE_INDEX + 1):
                    name = f'{kind}_ore_{n}'
                    try:
                        tf = self.tf_buffer.lookup_transform(base_frame, name, rclpy.time.Time())
                    except Exception:
                        continue
                    age = (now - rclpy.time.Time.from_msg(tf.header.stamp)).nanoseconds / 1e9
                    if age > 3.0:
                        continue                      # stale: the arm hid it or it is gone
                    t = tf.transform.translation
                    found[name] = {'name': name, 'kind': kind,
                                   'pos': np.array([t.x, t.y, t.z])}
            if found and t_window is None:
                t_window = time.monotonic()
            if t_window is not None and time.monotonic() - t_window > window_s:
                break
            time.sleep(0.2)
        return list(found.values())

    ############ THE SORTING SEQUENCE ############

    def find_pose(self, xyz, label):
        '''IK with a few lower fallbacks if the requested height is unreachable.'''
        for dz in (0.0, -0.05, -0.10, -0.15):
            target = (xyz[0], xyz[1], xyz[2] + dz)
            q = solve_pre_pose(target, HOME)
            if q is not None:
                if dz != 0.0:
                    self.get_logger().warn(f'{label}: used a lower approach (dz={dz})')
                return q, target[2]
        return None, None

    def pick_and_place(self, ore):
        kind = ore['kind']
        px, py, pz = ore['pos']
        gx, gy = px + GRASP_OFFSET_XY[0], py + GRASP_OFFSET_XY[1]
        top_z = pz + ORE_HALF_HEIGHT_M
        pre_z = top_z + PRE_GRASP_CLEAR

        cx, cy = CONTAINER_XY[kind]
        release_z = CONTAINER_RIM_Z + RELEASE_CLEAR
        approach_z = release_z + DROP_APPROACH_EXTRA

        q_pre, pre_z_used = self.find_pose((gx, gy, pre_z), f'{ore["name"]} pre-grasp')
        q_drop, appr_z_used = self.find_pose((cx, cy, approach_z), f'{kind} container approach')
        if q_pre is None or q_drop is None:
            self.get_logger().error(f'{ore["name"]}: no reachable tool-down pose '
                                    f'(ore ok: {q_pre is not None}, container ok: {q_drop is not None})')
            return 'unreachable'

        self.get_logger().info(f'--- {ore["name"]} at ({px:.3f},{py:.3f},{pz:.3f}) -> {kind} container ---')
        self.move_joints(HOME)
        self.move_joints(q_pre)
        self.descend_to_contact(top_z)
        self.move_vertical(pre_z_used, LIFT_SPEED)

        if not self.holding_ore():
            self.get_logger().warn(f'{ore["name"]}: not held, releasing and retrying')
            self.magnet(False)
            return 'not_held'

        self.move_joints(HOME)
        self.move_joints(q_drop)
        self.move_vertical(release_z, LIFT_SPEED)
        self.magnet(False)
        time.sleep(1.0)
        self.move_vertical(appr_z_used, LIFT_SPEED)
        self.get_logger().info(f'{ore["name"]} sorted')
        return 'done'

    def recover(self):
        '''Unlock a latched stop and back the tool straight up.'''
        self.stop()
        try:
            self.magnet(False)
        except Exception:
            pass
        for _ in range(3):
            self.get_logger().warn(f'recovering from status {self.status}')
            self.call(self.unlock_cli, Trigger.Request())
            self.use_controller('twist')
            t0 = time.monotonic()
            while time.monotonic() - t0 < 2.5:
                self.set_cmd('twist', [0.0, 0.0, 0.05, *self.tilt_hold()])
                time.sleep(0.02)
            self.stop()
            time.sleep(0.5)
            if self.status is not None and self.status not in CRITICAL_CODES:
                return True
        return False

    def wait_ready(self, timeout=90.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if (self.q is not None and self.tcp is not None and self.tcp_R is not None
                    and self.status is not None
                    and self.switch_cli.service_is_ready() and self.magnet_cli.service_is_ready()
                    and self.unlock_cli.service_is_ready()):
                return True
            time.sleep(0.2)
        return False

    def run(self):
        try:
            self._run()
        except ArmFatal as e:
            self.stop()
            self.get_logger().fatal(str(e))
        except Exception as e:
            self.stop()
            self.get_logger().error(f'sequence stopped: {e!r}')

    def _run(self):
        if not self.wait_ready():
            self.get_logger().error('arm interfaces never came up; is the world running?')
            return

        # Sanity check: our FK must agree with the controller's /tcp_pose_raw, otherwise
        # the IK targets would be in the wrong frame.
        p_fk, _ = fk(self.q)
        err = float(np.linalg.norm(p_fk - self.tcp))
        if err > 0.015:
            self.get_logger().error(f'FK disagrees with /tcp_pose_raw by {err:.3f} m '
                                    f'(fk {np.round(p_fk, 3)}, measured {np.round(self.tcp, 3)})')
        else:
            self.get_logger().info(f'FK check ok ({err * 1000:.1f} mm)')

        self.magnet(False)

        ores = self.scan_ores()
        if not ores:
            self.get_logger().error('no ore frames found: is task2A_perception.py running?')
            return
        # nearest to the arm first
        ores.sort(key=lambda o: float(np.hypot(o['pos'][0], o['pos'][1])))
        self.get_logger().info('ores to sort: ' + ', '.join(o['name'] for o in ores))

        results = {}
        for ore in ores:
            outcome = 'failed'
            for attempt in range(1, MAX_ATTEMPTS + 1):
                try:
                    outcome = self.pick_and_place(ore)
                except ArmFault as e:
                    self.get_logger().warn(f'{ore["name"]} attempt {attempt}: {e}')
                    if not self.recover():
                        self.get_logger().error('could not clear the fault; stopping')
                        results[ore['name']] = 'fault'
                        self.report(results)
                        return
                    outcome = 'fault'
                if outcome in ('done', 'unreachable'):
                    break
            results[ore['name']] = outcome

        try:
            self.move_joints(HOME)
        except Exception:
            pass
        self.report(results)

    def report(self, results):
        self.get_logger().info('=== summary ===')
        for k, v in results.items():
            self.get_logger().info(f'{k}: {v}')


##################### FUNCTION DEFINITION #######################

def main():
    '''
    Description:    Creates the ore_sorter node and spins it. The sorting sequence
                    runs in the node's worker thread.
    '''

    rclpy.init(args=sys.argv)
    node = ore_sorter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':

    main()
