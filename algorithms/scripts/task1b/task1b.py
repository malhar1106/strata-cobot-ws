#!/usr/bin/env python3
'''
*****************************************************************************************
*
*        		===============================================
*           		        StrataCobot (SC) Theme (eYRC 2026-27)
*        		===============================================
*
*  This script should be used to implement Task 1B of StrataCobot (SC) Theme (eYRC 2026-27).
*
*****************************************************************************************
'''
# Team ID:          1304
# Author List:      Swara Sukale,Malhar Pawle,Arman Khan,Sayali Katkar
# Filename:         task1b.py
################### IMPORT MODULES #######################
import rclpy
import sys
import math
import numpy as np
from rclpy.node import Node
from rclpy.duration import Duration
from control_msgs.msg import JointJog
from controller_manager_msgs.srv import SwitchController
from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32

##################### ARM GEOMETRY #######################
D1, A2, A3, D4, D5, D6 = 0.1625, -0.425, -0.3922, 0.1333, 0.0997, 0.0996
DH_D = [D1, 0, 0, D4, D5, D6]
DH_A = [0, A2, A3, 0, 0, 0]
DH_ALPHA = [math.pi/2, 0, 0, math.pi/2, -math.pi/2, 0]
MAX_2LINK_REACH = abs(A2) + abs(A3)
MIN_2LINK_REACH = abs(abs(A2) - abs(A3))

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
    _fixed_T((0, -math.pi/2, -math.pi/2), (0, 0, 0)) @
    _fixed_T((math.pi/2, 0, math.pi/2), (0, 0, 0)) @
    _fixed_T((0, 0, 0), (0, 0, 0.02041))
)

##################### TASK CONSTANTS #######################
HOLD_WAYPOINTS = [
    (-0.4085, -0.5379, 0.1967),   # 1
    (-0.8000, -0.0005, 0.3967),   # 2
    (-0.7430,  0.5280, 0.1967),   # 3
    (-0.4097,  0.5280, 0.1967),   # 4
    (-0.0763,  0.5280, 0.1967),   # 5
]

VIA_Z = 0.45          # slightly higher for safety
path = [
    {'xyz': HOLD_WAYPOINTS[0], 'hold': True,  'label': 'WP1'},
    {'xyz': (HOLD_WAYPOINTS[0][0], HOLD_WAYPOINTS[0][1], VIA_Z), 'hold': False, 'label': 'via lift after WP1'},
    {'xyz': (-0.55, -0.25, VIA_Z), 'hold': False, 'label': 'via clear hopper'},
    {'xyz': HOLD_WAYPOINTS[1], 'hold': True,  'label': 'WP2'},
    {'xyz': (HOLD_WAYPOINTS[1][0], HOLD_WAYPOINTS[1][1], VIA_Z), 'hold': False, 'label': 'via lift after WP2'},
    {'xyz': (HOLD_WAYPOINTS[2][0], HOLD_WAYPOINTS[2][1], VIA_Z), 'hold': False, 'label': 'via high to WP3'},
    {'xyz': HOLD_WAYPOINTS[2], 'hold': True,  'label': 'WP3'},
    {'xyz': HOLD_WAYPOINTS[3], 'hold': True,  'label': 'WP4'},
    {'xyz': HOLD_WAYPOINTS[4], 'hold': True,  'label': 'WP5'},
]

controller_manager_ns = '/ur_arm_controller'
twist_controller = 'delta_twist_controller'
joint_controller = 'delta_joint_controller'
base_frame = 'base_link'
joint_names = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]
cap_linear_mps = 0.12
cap_angular_rps = 0.30
cap_joint_rps = 0.18          # slow and safe
SAFETY_MARGIN = 0.90
command_timeout_s = 0.15

POSITION_TOLERANCE_M = 0.03
HOLD_DURATION_S = 2.0
KP_LINEAR = 0.7
# Orientation is NOT scored for this task (confirmed from the marking scheme).
# Locking to DESIRED_R made several via points geometrically unreachable
# within tolerance (position-only IK converges to <0.1mm error at the exact
# same targets that fail combined pos+ori IK by ~0.08 -- verified numerically),
# which is what was driving the arm back toward the elbow singularity. Left
# at 0.0 so orientation error is still computed (harmless) but never commanded.
KP_ANGULAR = 0.0
KP_JOINT = 0.6
RAMP_DURATION_S = 1.2

# --- Seed-IK tuning (STRICT against singularity) ----------------------------
SEED_PULL_M = 0.18
# 0.3 rad (~17 deg) -- reverted from 1.0 rad (57 deg). The 1.0 rad floor
# rejected the only physically-available elbow bends at some via points
# (e.g. ~30 deg at "via high to WP3", verified numerically), forcing a
# Cartesian-only fallback that re-created the exact singularity this filter
# was meant to prevent.
MIN_ACCEPTABLE_ELBOW_RAD = 0.3
# elbow_joint's real hardware limit, confirmed from the URDF: +-2.7925 rad
# (+-160 deg). Unlike the other 5 joints (all +-2*pi, effectively continuous),
# elbow physically cannot wrap around -- so unwrap_joints() must never be
# applied to it (see the note in choose_seed_branch), and any solution beyond
# this bound (with a small safety margin) must be hard-rejected rather than
# picked as a "more bent" seed.
ELBOW_HARD_LIMIT_RAD = 2.75
JOINT_SEED_TOLERANCE_RAD = 0.06
JOINT_SEED_TIMEOUT_S = 15.0
NUMERIC_IK_MAX_ITERS = 60
NUMERIC_IK_TOL = 1e-4
NUMERIC_IK_DAMPING = 1e-3

# Desired tool orientation -- kept only as a reference frame for
# orientation_error(); no longer enforced (see KP_ANGULAR / numeric_ik above).
DESIRED_R = _rpy_to_R(0.0, math.pi, 0.0)

CRITICAL_CODES = {20, 21, 22, 23}
UNRECOVERABLE_CODE = 92
RECOVERY_WINDOW_S = 6.0
RECOVER_SPEED = 0.11
MAX_RELATCH_ATTEMPTS = 6

##################### KINEMATICS ############################################
def dh_transform(theta, d, a, alpha):
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d],
        [0,   0,      0,     1],
    ])

def fk_upto(thetas, n):
    T = np.eye(4)
    for i in range(n):
        T = T @ dh_transform(thetas[i], DH_D[i], DH_A[i], DH_ALPHA[i])
    return T

def tcp_pose(thetas6):
    T06 = fk_upto(thetas6, 6)
    T = T06 @ T_WRIST3_TCP
    return T[:3, 3].copy(), T[:3, :3].copy()

def orientation_error(R_cur, R_des):
    R_err = R_cur.T @ R_des
    cos_theta = max(-1.0, min(1.0, (np.trace(R_err) - 1.0) * 0.5))
    theta = math.acos(cos_theta)
    if abs(theta) < 1e-8:
        return np.zeros(3)
    k = np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1],
    ]) / (2.0 * math.sin(theta))
    return k * theta

def solve_shoulder_elbow(p4_target, elbow_sign, shoulder_sign):
    px, py, pz = p4_target
    r = math.hypot(px, py)
    if r < abs(D4) - 1e-9:
        return None
    phi = math.atan2(py, px)
    s = max(-1.0, min(1.0, D4 / r))
    if shoulder_sign > 0:
        theta1 = phi + math.asin(s)
    else:
        theta1 = phi + math.pi - math.asin(s)
    theta1 = (theta1 + math.pi) % (2*math.pi) - math.pi
    X = px*math.cos(theta1) + py*math.sin(theta1)
    Y = pz - D1
    r2sq = X*X + Y*Y
    r2 = math.sqrt(r2sq)
    if r2 > MAX_2LINK_REACH + 1e-9 or r2 < MIN_2LINK_REACH - 1e-9:
        return None
    cos3 = max(-1.0, min(1.0, (r2sq - A2*A2 - A3*A3) / (2*A2*A3)))
    theta3 = elbow_sign * math.acos(cos3)
    theta2 = math.atan2(Y, X) - math.atan2(A3*math.sin(theta3), A2 + A3*math.cos(theta3))
    return theta1, theta2, theta3

def numeric_ik(target_xyz, target_R, theta_seed, w_pos=1.0, w_ori=0.25):
    theta = np.array(theta_seed, dtype=float)
    target = np.array(target_xyz, dtype=float)
    err = np.zeros(6)
    for _ in range(NUMERIC_IK_MAX_ITERS):
        p, Rm = tcp_pose(theta)
        e_pos = target - p
        e_ori = orientation_error(Rm, target_R)
        err = np.concatenate([w_pos * e_pos, w_ori * e_ori])
        if np.linalg.norm(err) < NUMERIC_IK_TOL:
            break
        J = np.zeros((6, 6))
        h = 1e-6
        for j in range(6):
            tp = theta.copy()
            tp[j] += h
            pp, Rp = tcp_pose(tp)
            J[:3, j] = (pp - p) / h
            J[3:, j] = orientation_error(Rm, Rp) / h
        # FIX: weight J the same way as err before forming the normal
        # equations. Previously A = J.T @ J used the *unweighted* J while
        # the RHS used the weighted err -- with w_ori=0 that made A carry
        # full (unweighted) orientation sensitivity while the RHS carried
        # none, an inconsistent, poorly-conditioned system. Near-singular
        # targets (e.g. WP1) let this push theta through large, spurious
        # jumps -- including multi-revolution jumps in the elbow joint,
        # which is a periodic (sin/cos) DOF with no built-in preference for
        # the physically-reachable representation of a given position.
        Jw = J.copy()
        Jw[:3, :] *= w_pos
        Jw[3:, :] *= w_ori
        A = Jw.T @ Jw + NUMERIC_IK_DAMPING * np.eye(6)
        dtheta = np.linalg.solve(A, Jw.T @ err)
        theta = theta + dtheta
    return theta, float(np.linalg.norm(err))

def unwrap_joints(target, current):
    if current is None:
        return list(target)
    unwrapped = []
    for t, c in zip(target, current):
        diff = (t - c + math.pi) % (2 * math.pi) - math.pi
        unwrapped.append(c + diff)
    return unwrapped

def choose_seed_branch(target_xyz, current_joint_angles):
    target_xyz = np.array(target_xyz, dtype=float)
    radial = target_xyz / (np.linalg.norm(target_xyz) + 1e-9)
    p4_seed_guess = target_xyz - SEED_PULL_M * radial
    candidates = []

    for shoulder_sign in (+1, -1):
        for elbow_sign in (+1, -1):
            sol = solve_shoulder_elbow(p4_seed_guess, elbow_sign, shoulder_sign)
            if sol is None:
                continue
            t1, t2, t3 = sol
            wrist_seed = list(current_joint_angles[3:6]) if current_joint_angles is not None else [0.0, 0.0, 0.0]
            theta_seed = [t1, t2, t3] + wrist_seed

            # w_ori=0.0: orientation is unscored and, at some via points,
            # unreachable in combination with position within tolerance --
            # see the MIN_ACCEPTABLE_ELBOW_RAD / KP_ANGULAR notes above.
            theta_sol, err = numeric_ik(target_xyz, DESIRED_R, theta_seed, w_ori=0.0)
            if err > 0.03:
                continue

            # unwrap_joints picks the coterminal angle nearest the arm's
            # CURRENT position for every joint -- correct for the 5 continuous
            # joints, but for elbow (hard +-160 deg limit, cannot wrap) this
            # can turn a genuine, legal bend (e.g. 66 deg) into a physically
            # impossible representation (e.g. -294 deg) whenever the current
            # elbow position happens to be far away.
            #
            # FIX: numeric_ik's raw elbow output is NOT guaranteed to stay
            # near its closed-form seed. tcp_pose() is 2*pi-periodic in the
            # elbow angle (built from sin/cos), so Newton's method has no
            # built-in preference for the small, in-limit representation of
            # a given TCP position over one that's aliased by a full
            # revolution -- which branch it lands on is sensitive to
            # floating-point rounding order and is NOT reproducible across
            # machines/numpy builds. Observed on hardware: raw_elbow came
            # back as -5.131 rad (-294 deg) for a target whose true bend is
            # +1.152 rad (+66 deg) -- identical TCP position, aliased by
            # exactly 2*pi. Canonicalizing into (-pi, pi] before the limit
            # check (and before this value is ever commanded) removes that
            # aliasing so the hard-limit check evaluates the angle that will
            # actually be sent to the joint controller.
            raw_elbow = theta_sol[2]
            raw_elbow = ((raw_elbow + math.pi) % (2 * math.pi)) - math.pi
            theta_sol = unwrap_joints(theta_sol, current_joint_angles)
            theta_sol[2] = raw_elbow
            elbow_final = abs(theta_sol[2])

            # Hard reject near-singular AND out-of-hardware-limit configurations
            if elbow_final < MIN_ACCEPTABLE_ELBOW_RAD or elbow_final > ELBOW_HARD_LIMIT_RAD:
                continue

            if current_joint_angles is not None:
                # NOTE: elbow can't physically wrap, so its contribution to
                # travel is the plain (canonicalized) difference, not a
                # shortest-path-mod-2pi difference -- consistent with how
                # the 'seeding' phase must also treat it (see process_waypoints).
                diffs = [abs(a - b) for a, b in zip(theta_sol, current_joint_angles)]
                joint_travel = sum(diffs)
            else:
                joint_travel = 0.0

            # Prefer: largest elbow angle, then smallest travel
            candidates.append((-elbow_final, joint_travel, theta_sol, err))

    if not candidates:
        return None

    candidates.sort()   # most bent elbow first, then smallest travel
    return candidates[0][2]

##################### CLASS DEFINITION #######################
class arm_waypoints(Node):
    def __init__(self):
        super().__init__(
            'arm_waypoints_node',
            parameter_overrides=[rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        self.twist_pub = self.create_publisher(TwistStamped, '/delta_twist_cmds', 10)
        self.joint_pub = self.create_publisher(JointJog, '/delta_joint_cmds', 10)

        self.tcp_sub = self.create_subscription(PoseStamped, '/tcp_pose_raw', self.tcpposecb, 20)
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.jointstatecb, 50)
        self.status_sub = self.create_subscription(Int32, '/arm_status', self.armstatuscb, 10)

        self.switch_cli = self.create_client(SwitchController, f'{controller_manager_ns}/switch_controller')
        self.unlock_cli = self.create_client(Trigger, '/ur_arm_controller/unlock_protective_stop')

        self.tcp_pose = None
        self.tcp_quat = None
        self.joint_angles = None
        self.arm_status = None
        self._last_status = None
        self.path_idx = 0
        self.wp_idx = 0
        self.wp_reached_time = None
        self.route_done = False
        self.done_logged = False
        self._last_logged_idx = -1
        self.active_controller = 'twist_controller'
        self._switch_pending = False
        self._switch_succeeded = None

        self.step_phase = 'need_seed'
        self.seed_targets = None
        self.seed_start_time = None
        self.ramp_start_time = None
        self.in_recovery = False
        self.recovery_deadline = None
        self.consecutive_relatches = 0
        self.abandoned = False
        self._recovery_switch_retry_time = None
        self.force_safe_via = False          # after singularity recovery

        self.timer = self.create_timer(0.05, self.process_waypoints)
        self.get_logger().info('arm_waypoints_node started...')

    def tcpposecb(self, data):
        p = data.pose.position
        q = data.pose.orientation
        self.tcp_pose = (p.x, p.y, p.z)
        self.tcp_quat = (q.x, q.y, q.z, q.w)

    def jointstatecb(self, data):
        try:
            self.joint_angles = [data.position[data.name.index(j)] for j in joint_names]
        except ValueError:
            pass

    def armstatuscb(self, data):
        self.arm_status = data.data
        prev = self._last_status
        if self.arm_status != self._last_status:
            self.get_logger().info(f'arm_status changed to {self.arm_status}')
            self._last_status = self.arm_status

        if self.arm_status == UNRECOVERABLE_CODE:
            self.get_logger().fatal('Status 92 – unrecoverable. Halting.')
            self.route_done = True
            return

        fresh_latch = (self.arm_status in CRITICAL_CODES and not self.in_recovery
                       and prev not in CRITICAL_CODES)
        if fresh_latch:
            self.consecutive_relatches += 1
            heading = path[self.path_idx]['label'] if self.path_idx < len(path) else 'end'
            self.get_logger().warn(
                f'Status {self.arm_status} latched for {heading} '
                f'(attempt {self.consecutive_relatches}/{MAX_RELATCH_ATTEMPTS})')
            if self.consecutive_relatches >= MAX_RELATCH_ATTEMPTS:
                self.get_logger().error('Too many re-faults. Stopping.')
                self.abandoned = True
                self._publish_twist(0,0,0,0,0,0)
                return
            self.in_recovery = True
            self.recovery_deadline = self.get_clock().now() + Duration(seconds=RECOVERY_WINDOW_S)
            self.force_safe_via = True          # next seed will go high first
            self._call_unlock()
        elif self.arm_status == 0 and self.in_recovery:
            self.get_logger().info('Recovered – forcing safe high via then re-seeding.')
            self.in_recovery = False
            self.recovery_deadline = None
            self._recovery_switch_retry_time = None
            self.step_phase = 'need_seed'

    def _call_unlock(self):
        if not self.unlock_cli.service_is_ready():
            return
        future = self.unlock_cli.call_async(Trigger.Request())
        def _cb(fut):
            try:
                r = fut.result()
                self.get_logger().info(f'unlock: success={r.success}, msg={r.message}')
            except Exception as e:
                self.get_logger().error(f'unlock failed: {e}')
            self.in_recovery = True
            self.recovery_deadline = self.get_clock().now() + Duration(seconds=RECOVERY_WINDOW_S)
        future.add_done_callback(_cb)

    def _recovery_command(self):
        # Strong outward + upward retreat
        speed = RECOVER_SPEED
        if self.tcp_pose is None:
            return (0.0, 0.0, speed, 0,0,0)
        x, y, z = self.tcp_pose
        xy = math.hypot(x, y)
        if xy > 0.05:
            return (-x/xy * speed * 0.9, -y/xy * speed * 0.9, speed * 0.8, 0,0,0)
        return (0.0, 0.0, speed, 0,0,0)

    def switch_controller(self, controller):
        if self._switch_pending:
            return
        target = twist_controller if controller == 'twist_controller' else joint_controller
        other  = joint_controller if controller == 'twist_controller' else twist_controller
        if not self.switch_cli.service_is_ready():
            self._switch_succeeded = False
            return
        req = SwitchController.Request()
        req.activate_controllers = [target]
        req.deactivate_controllers = [other]
        req.strictness = SwitchController.Request.STRICT
        self._switch_pending = True
        future = self.switch_cli.call_async(req)
        def _cb(fut):
            self._switch_pending = False
            try:
                r = fut.result()
                if r.ok:
                    self.active_controller = controller
                    self._switch_succeeded = True
                else:
                    self._switch_succeeded = False
            except:
                self._switch_succeeded = False
        future.add_done_callback(_cb)

    def _publish_joint_jog(self, names, velocities):
        msg = JointJog()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = base_frame
        msg.joint_names = list(names)
        msg.velocities = list(velocities)
        msg.duration = command_timeout_s
        self.joint_pub.publish(msg)

    def _publish_twist(self, lx, ly, lz, ax=0.0, ay=0.0, az=0.0):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = base_frame
        msg.twist.linear.x = float(lx)
        msg.twist.linear.y = float(ly)
        msg.twist.linear.z = float(lz)
        msg.twist.angular.x = float(ax)
        msg.twist.angular.y = float(ay)
        msg.twist.angular.z = float(az)
        self.twist_pub.publish(msg)

    def process_waypoints(self):
        try:
            self._process_waypoints_impl()
        except Exception as e:
            self.get_logger().error(f'process failed: {e}', throttle_duration_sec=2.0)
            self._publish_twist(0,0,0,0,0,0)

    def _process_waypoints_impl(self):
        if self.tcp_pose is None or self.tcp_quat is None or \
           self.arm_status is None or self.joint_angles is None:
            return
        if self.abandoned:
            self._publish_twist(0,0,0,0,0,0)
            return

        # ---------- RECOVERY ----------
        if self.in_recovery:
            if self.recovery_deadline and self.get_clock().now() > self.recovery_deadline:
                self.consecutive_relatches += 1
                self.get_logger().warn(f'Recovery window expired (attempt {self.consecutive_relatches})')
                if self.consecutive_relatches >= MAX_RELATCH_ATTEMPTS:
                    self.get_logger().error('Could not clear latch. Stopping.')
                    self.abandoned = True
                    self.in_recovery = False
                    self._publish_twist(0,0,0,0,0,0)
                    return
                self.recovery_deadline = self.get_clock().now() + Duration(seconds=RECOVERY_WINDOW_S)
                self._call_unlock()
            if self.active_controller != 'twist_controller':
                now = self.get_clock().now()
                if not self._switch_pending and (self._recovery_switch_retry_time is None or now > self._recovery_switch_retry_time):
                    self.switch_controller('twist_controller')
                    self._recovery_switch_retry_time = now + Duration(seconds=0.4)
                return
            self._publish_twist(*self._recovery_command())
            return

        if self.route_done:
            self._publish_twist(0,0,0,0,0,0)
            if not self.done_logged:
                self.get_logger().info('All waypoints reached and held.')
                self.done_logged = True
            return

        step = path[self.path_idx]
        if self.path_idx != self._last_logged_idx:
            self._last_logged_idx = self.path_idx
            self.get_logger().info(f'Tracking {step["label"]} -> {step["xyz"]}')

        # ---------- NEED SEED ----------
        if self.step_phase == 'need_seed':
            # After singularity recovery, first go to a high safe point
            if self.force_safe_via:
                safe_xyz = (self.tcp_pose[0]*0.7, self.tcp_pose[1]*0.7, 0.48)
                self.get_logger().info(f'Forcing safe high via: {safe_xyz}')
                sol = choose_seed_branch(safe_xyz, self.joint_angles)
                self.force_safe_via = False
            else:
                sol = choose_seed_branch(step['xyz'], self.joint_angles)

            if sol is None:
                self.get_logger().warn(f'No good seed for {step["label"]} – Cartesian only')
                self.seed_targets = list(self.joint_angles)
            else:
                self.seed_targets = list(sol)
                self.get_logger().info(
                    f'{step["label"]} seed: elbow={math.degrees(self.seed_targets[2]):.1f}°, '
                    f'joints={[round(v,3) for v in self.seed_targets]}')
            self.seed_start_time = self.get_clock().now()
            if self.active_controller != 'joint_controller':
                self.switch_controller('joint_controller')
                self.step_phase = 'awaiting_joint_switch'
            else:
                self.step_phase = 'seeding'
            return

        if self.step_phase == 'awaiting_joint_switch':
            if self._switch_pending:
                return
            if self._switch_succeeded:
                self.step_phase = 'seeding'
            else:
                self.step_phase = 'switch_to_twist'
            self._switch_succeeded = None
            return

        # ---------- SEEDING (joint space) ----------
        if self.step_phase == 'seeding':
            elapsed = (self.get_clock().now() - self.seed_start_time).nanoseconds / 1e9
            # NOTE: elbow_joint (index 2) is hardware-limited (+-2.7925 rad)
            # and cannot physically wrap around like the other 5 (effectively
            # continuous) joints. choose_seed_branch() now guarantees
            # seed_targets[2] is already canonicalized into (-pi, pi], so a
            # plain (non-wrapped) difference is both correct and safe for it.
            # The generic shortest-path-mod-2pi formula is fine for the other
            # 5 joints, so we special-case just the elbow entry.
            diffs = []
            for idx, (t, c) in enumerate(zip(self.seed_targets, self.joint_angles)):
                if idx == 2:
                    diffs.append(t - c)
                else:
                    diffs.append((t - c + math.pi) % (2*math.pi) - math.pi)
            max_diff = max(abs(d) for d in diffs)
            if max_diff < JOINT_SEED_TOLERANCE_RAD or elapsed > JOINT_SEED_TIMEOUT_S:
                if elapsed > JOINT_SEED_TIMEOUT_S:
                    self.get_logger().warn(f'Seed timeout for {step["label"]}')
                self.step_phase = 'switch_to_twist'
                return
            ramp = min(1.0, elapsed / RAMP_DURATION_S)
            cap = cap_joint_rps * SAFETY_MARGIN
            vels = [max(-cap, min(cap, KP_JOINT * d * ramp)) for d in diffs]
            self._publish_joint_jog(joint_names, vels)
            return

        if self.step_phase == 'switch_to_twist':
            if self.active_controller != 'twist_controller':
                if not self._switch_pending and self._switch_succeeded is None:
                    self.switch_controller('twist_controller')
                    return
                if self._switch_pending:
                    return
                self._switch_succeeded = None
            self.step_phase = 'fine_track'
            self.ramp_start_time = None
            return

        # ---------- FINE TRACK (Cartesian) ----------
        if self.step_phase == 'fine_track':
            if self.ramp_start_time is None:
                self.ramp_start_time = self.get_clock().now()

            x, y, z = self.tcp_pose
            tx, ty, tz = step['xyz']
            err_pos = (tx-x, ty-y, tz-z)
            distance = math.sqrt(sum(e*e for e in err_pos))

            qx, qy, qz, qw = self.tcp_quat
            R_cur = np.array([
                [1-2*(qy*qy+qz*qz),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
                [  2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz),   2*(qy*qz-qx*qw)],
                [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)]
            ])
            e_ori = orientation_error(R_cur, DESIRED_R)

            if distance < POSITION_TOLERANCE_M:
                if not step['hold']:
                    self.get_logger().info(f'Passed {step["label"]}')
                    self._advance_step()
                    self._publish_twist(0,0,0,0,0,0)
                    return
                now = self.get_clock().now()
                if self.wp_reached_time is None:
                    self.wp_reached_time = now
                elif (now - self.wp_reached_time).nanoseconds / 1e9 >= HOLD_DURATION_S:
                    self.wp_idx += 1
                    self.get_logger().info(f'{step["label"]} held ({self.wp_idx}/5)')
                    self._advance_step()
                    if self.route_done:
                        self._publish_twist(0,0,0,0,0,0)
                    return
                omega = KP_ANGULAR * e_ori
                mag = np.linalg.norm(omega)
                if mag > cap_angular_rps * SAFETY_MARGIN:
                    omega *= (cap_angular_rps * SAFETY_MARGIN) / mag
                self._publish_twist(0,0,0, *omega)
                return
            else:
                self.wp_reached_time = None

            # linear
            cmd = [KP_LINEAR * e for e in err_pos]
            elapsed = (self.get_clock().now() - self.ramp_start_time).nanoseconds / 1e9
            ramp = min(1.0, elapsed / RAMP_DURATION_S)
            cmd = [c * ramp for c in cmd]
            mag = math.sqrt(sum(c*c for c in cmd))
            if mag > cap_linear_mps:
                scale = (cap_linear_mps * SAFETY_MARGIN) / mag
                cmd = [c * scale for c in cmd]

            # angular
            omega = KP_ANGULAR * e_ori * ramp
            mag_o = np.linalg.norm(omega)
            if mag_o > cap_angular_rps * SAFETY_MARGIN:
                omega *= (cap_angular_rps * SAFETY_MARGIN) / mag_o

            self._publish_twist(*cmd, *omega)
            return

    def _advance_step(self):
        self.path_idx += 1
        self.wp_reached_time = None
        self.ramp_start_time = None
        self.consecutive_relatches = 0
        self.step_phase = 'need_seed'
        if self.path_idx >= len(path):
            self.route_done = True

def main():
    rclpy.init(args=sys.argv)
    node = arm_waypoints()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
