import numpy as np
import cv2


class TimedWaypointPlanner:
   
    def __init__(
        self,
        path_waypoints,
        nominal_speed=2.8,
        disc_steps=26,
        min_seg_time=0.10,
        segment_speed_scales=None,
    ):
        self.path = [np.array(p, dtype=float) for p in path_waypoints]
        self.disc_steps = int(disc_steps)
        self.nominal_speed = float(nominal_speed)
        self.min_seg_time = float(min_seg_time)

        if segment_speed_scales is None:
            self.segment_speed_scales = np.ones(len(self.path) - 1, dtype=float)
        else:
            scales = np.array(segment_speed_scales, dtype=float)
            if len(scales) != len(self.path) - 1:
                raise ValueError("segment_speed_scales must have len(path_waypoints)-1 entries")
            self.segment_speed_scales = scales

        self.times = self.build_distance_based_times(self.path)
        poly_coeffs = self.compute_poly_coefficients(self.path)
        self.trajectory_setpoints, self.time_setpoints = self.poly_setpoint_extraction(poly_coeffs)

    def build_distance_based_times(self, path_waypoints):
        pts = [np.array(p, dtype=float) for p in path_waypoints]
        n = len(pts)

        turn_weight = np.ones(n, dtype=float)
        for i in range(1, n - 1):
            v1 = pts[i] - pts[i - 1]
            v2 = pts[i + 1] - pts[i]
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)

            if n1 > 1e-6 and n2 > 1e-6:
                c = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                ang = np.arccos(c)
                turn_weight[i] = 1.0 + 0.85 * (ang / np.pi)

        times = [0.0]
        for i in range(n - 1):
            seg_len = np.linalg.norm(pts[i + 1] - pts[i])
            speed_scale = float(np.clip(self.segment_speed_scales[i], 0.40, 1.70))
            base_dt = seg_len / max(self.nominal_speed * speed_scale, 0.20)
            dt = base_dt * max(turn_weight[i], turn_weight[i + 1])
            dt = max(self.min_seg_time, dt)
            times.append(times[-1] + dt)

        return np.array(times, dtype=float)

    def compute_poly_matrix(self, t):
        return np.array([
            [t**5,        t**4,       t**3,      t**2,   t, 1],
            [5*t**4,      4*t**3,     3*t**2,    2*t,    1, 0],
            [20*t**3,     12*t**2,    6*t,       2,      0, 0],
            [60*t**2,     24*t,       6,         0,      0, 0],
            [120*t,       24,         0,         0,      0, 0],
        ], dtype=float)

    def compute_poly_coefficients(self, path_waypoints):
        seg_times = np.diff(self.times)
        m = len(path_waypoints)
        poly_coeffs = np.zeros((6 * (m - 1), 3), dtype=float)

        for dim in range(3):
            A = np.zeros((6 * (m - 1), 6 * (m - 1)), dtype=float)
            b = np.zeros(6 * (m - 1), dtype=float)

            pos = np.array([p[dim] for p in path_waypoints], dtype=float)
            A_0 = self.compute_poly_matrix(0.0)

            row = 0
            for i in range(m - 1):
                pos_0 = pos[i]
                pos_f = pos[i + 1]
                v_0, a_0 = 0.0, 0.0
                v_f, a_f = 0.0, 0.0
                A_f = self.compute_poly_matrix(seg_times[i])

                if i == 0:
                    A[row, i*6:(i+1)*6] = A_0[0]
                    b[row] = pos_0
                    row += 1

                    A[row, i*6:(i+1)*6] = A_f[0]
                    b[row] = pos_f
                    row += 1

                    A[row, i*6:(i+1)*6] = A_0[1]
                    b[row] = v_0
                    row += 1

                    A[row, i*6:(i+1)*6] = A_0[2]
                    b[row] = a_0
                    row += 1

                    A[row:row+4, i*6:(i+1)*6] = A_f[1:]
                    A[row:row+4, (i+1)*6:(i+2)*6] = -A_0[1:]
                    row += 4

                elif i < m - 2:
                    A[row, i*6:(i+1)*6] = A_0[0]
                    b[row] = pos_0
                    row += 1

                    A[row, i*6:(i+1)*6] = A_f[0]
                    b[row] = pos_f
                    row += 1

                    A[row:row+4, i*6:(i+1)*6] = A_f[1:]
                    A[row:row+4, (i+1)*6:(i+2)*6] = -A_0[1:]
                    row += 4

                else:
                    A[row, i*6:(i+1)*6] = A_0[0]
                    b[row] = pos_0
                    row += 1

                    A[row, i*6:(i+1)*6] = A_f[0]
                    b[row] = pos_f
                    row += 1

                    A[row, i*6:(i+1)*6] = A_f[1]
                    b[row] = v_f
                    row += 1

                    A[row, i*6:(i+1)*6] = A_f[2]
                    b[row] = a_f
                    row += 1

            try:
                poly_coeffs[:, dim] = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                poly_coeffs[:, dim] = np.linalg.lstsq(A, b, rcond=None)[0]

        return poly_coeffs

    def poly_setpoint_extraction(self, poly_coeffs):
        n_seg = len(self.times) - 1
        n_pts = self.disc_steps * n_seg + 1
        time_setpoints = np.linspace(self.times[0], self.times[-1], n_pts)

        x_vals = np.zeros(n_pts, dtype=float)
        y_vals = np.zeros(n_pts, dtype=float)
        z_vals = np.zeros(n_pts, dtype=float)

        coeff_x = poly_coeffs[:, 0]
        coeff_y = poly_coeffs[:, 1]
        coeff_z = poly_coeffs[:, 2]

        for i, t in enumerate(time_setpoints):
            seg_idx = np.searchsorted(self.times, t, side="right") - 1
            seg_idx = int(np.clip(seg_idx, 0, n_seg - 1))

            tau = t - self.times[seg_idx]
            A_tau = self.compute_poly_matrix(tau)

            cxs = coeff_x[seg_idx * 6:(seg_idx + 1) * 6]
            cys = coeff_y[seg_idx * 6:(seg_idx + 1) * 6]
            czs = coeff_z[seg_idx * 6:(seg_idx + 1) * 6]

            x_vals[i] = A_tau[0] @ cxs
            y_vals[i] = A_tau[0] @ cys
            z_vals[i] = A_tau[0] @ czs

        yaw_vals = np.zeros(n_pts, dtype=float)
        lookahead_idx = 4
        yaw_alpha = 0.18

        for i in range(n_pts):
            j = min(i + lookahead_idx, n_pts - 1)
            dx = x_vals[j] - x_vals[i]
            dy = y_vals[j] - y_vals[i]

            if np.hypot(dx, dy) > 1e-6:
                yaw_vals[i] = np.arctan2(dy, dx)
            elif i > 0:
                yaw_vals[i] = yaw_vals[i - 1]
            else:
                yaw_vals[i] = 0.0

        if n_pts > 1:
            yaw_vals[-1] = yaw_vals[-2]

        yaw_vals = np.unwrap(yaw_vals)
        for i in range(1, n_pts):
            yaw_vals[i] = (1.0 - yaw_alpha) * yaw_vals[i - 1] + yaw_alpha * yaw_vals[i]
        yaw_vals = (yaw_vals + np.pi) % (2 * np.pi) - np.pi

        traj = np.column_stack((x_vals, y_vals, z_vals, yaw_vals))
        return traj, time_setpoints


class MyAssignment:
    def __init__(self):
        self.debug = True

        # Arena
        self.circle_center = np.array([4.0, 4.0], dtype=float)
        self.track_radius = 2.55
        self.search_height = 1.15
        self.takeoff_height = 1.0
        self.search_lookahead = 0.42

        self.num_gates = 5
        self.num_slices = 12
        self.slice_width = 2.0 * np.pi / self.num_slices
        self.gate_slices = [2, 4, 6, 8, 10]

        # Global progression
        self.state = "SEARCH_RING"
        self.last_slice = None
        self.lap = 0
        self.current_gate_idx = 0

        # Gate map from lap 1
        self.saved_gates = [None] * self.num_gates
        self.saved_gate_dirs = [None] * self.num_gates

        # Detection tuning
        self.min_gate_area = 105
        self.candidate_min_area = 85
        self.candidate_min_wh = 14
        self.start_track_area = 150
        self.min_wh = 18
        self.max_start_abs_norm_x = 0.82
        self.max_start_abs_norm_y = 0.70
        self.max_valid_gate_abs_x = 0.94
        self.center_tol_x = 0.10
        self.center_tol_y = 0.13
        self.obs_collect_tol_x = 0.32
        self.obs_collect_tol_y = 0.24
        self.max_lost_frames = 6
        self.lock_jump_tol = 0.23
        self.lock_area_ratio = 0.45

        self.expected_gate_x_weight = 8.0
        self.expected_gate_y_weight = 0.08
        self.expected_gate_center_weight = 0.20
        self.expected_gate_area_weight = 0.0002
        self.ambiguous_gate_area_weight = 0.0030

        self.expected_gate_x_side_deadband = 0.08
        self.expected_gate_x_side_slack = 0.02
        self.expected_gate_x_strict_deadband = 0.12
        self.expected_gate_x_min_fraction = 0.60
        self.expected_gate_x_side_memory = 0.0

        self.lock_follow_max_jump = 0.38
        self.lock_follow_area_ratio_min = 0.25
        self.wrong_tri_count = 0
        self.wrong_tri_limit = 2

        # Scan policy
        self.scan_dwell = 0.22
        self.scan_wait = 0.0
        self.scan_plan = [
            ("prev", -0.35, -0.18, 0.00),
            ("prev", -0.22, -0.05, 0.00),
            ("prev", -0.08,  0.02, 0.10),
            ("curr", -0.42, -0.10, 0.00),
            ("curr", -0.15, -0.02, 0.08),
            ("curr",  0.08, -0.02, 0.12),
        ]
        # self.scan_yaw_offsets = [0.0, -0.22, 0.22, -0.45, 0.45]
        self.scan_yaw_offsets = [0.0, -0.35, 0.35, -0.75, 0.75]
        self.scan_heights = [self.search_height, self.search_height + 0.22, self.search_height - 0.18]
        self.scan_pose_idx = 0
        self.scan_yaw_idx = 0
        self.scan_height_idx = 0
        self.scan_hover_xy = None
        self.scan_base_yaw = 0.0
        self.scan_retry_count = 0

        self.scan_base_outward = 0.24
        self.scan_retry_outward_step = 0.18
        self.scan_max_outward = 0.72
        self.approach_fail_count = 0

        # Setpoint tolerances for hover scan
        self.xy_tol = 0.10
        self.z_tol = 0.08
        self.yaw_tol = 0.16

        # Tracking / triangulation buffer
        self.locked_detection = None
        self.lost_frames = 0
        self.obs_buffer = []
        self.min_obs_for_triangulation = 3
        self.min_baseline = 0.22
        self.max_point_line_dist = 0.16
        self.min_obs_spacing = 0.045
        self.tri_motion_sign = 1.0
        self.max_track_backward_slices = 1

        # Passing geometry (lap 1 and fallback)
        self.pass_pre_dist = 0.42
        self.pass_post_dist = 0.52
        self.pass_center_tol = 0.12
        self.pass_post_tol = 0.16
        self.pass_phase = 0
        self.pass_pre_point = None
        self.pass_center = None
        self.pass_post_point = None
        self.pass_dir = None

        # Timed laps / smooth fast laps
        self.use_timed_laps = True
        self.timed_planner = None
        self.timed_start_time = None
        self.timed_end_anchor = None
        self.timed_ready_count = 0

        # Timed lap.
        self.timed_nominal_speed = 4.6
        self.timed_disc_steps = 24
        self.timed_min_seg_time = 0.08
        self.timed_path_cursor = 0
        self.timed_search_window = 150
        self.timed_lookahead_pts = 24
        self.timed_lookahead_pts_slow = 14
        self.timed_err_reduce = 0.32
        self.timed_err_zero = 0.62
        self.timed_finish_tol = 0.40

        # Gate guidance for laps 2 and 3.
        self.timed_gate_corridor = 0.50
        self.timed_gate_release_progress = 0.18
        self.timed_gate_attract_radius = 1.1
        self.timed_gate_attract_gain = 0.4

        # Gate 0 only.
        self.timed_gate0_corridor = 0.70
        self.timed_gate0_extra_pre = 0.38
        self.timed_gate0_release_progress = 0.30
        self.timed_gate0_lookahead_pts = 20
        self.timed_gate0_attract_radius = 1.30
        self.timed_gate0_attract_gain = 0.40

        # Ordered gate progression for timed laps
        self.timed_gate_idx = 0
        self.timed_gate_center_path_idx = []
        self.timed_gate_post_path_idx = []

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------
    def log(self, msg):
        if self.debug:
            print(msg, flush=True)

    def wrap_to_pi(self, angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def vec3(self, sensor_data):
        return np.array([
            sensor_data["x_global"],
            sensor_data["y_global"],
            sensor_data["z_global"],
        ], dtype=float)

    def get_angle_from_center(self, x, y):
        v = np.array([x, y], dtype=float) - self.circle_center
        n = np.linalg.norm(v)
        if n < 1e-8:
            return None
        return (np.arctan2(v[1], v[0]) + np.pi) % (2.0 * np.pi)

    def get_slice_idx(self, x, y):
        ang = self.get_angle_from_center(x, y)
        if ang is None:
            return -1
        shifted = (ang + 0.5 * self.slice_width) % (2.0 * np.pi)
        return int(shifted // self.slice_width)

    def get_previous_slice(self, idx):
        return (idx - 1) % self.num_slices

    def get_expected_gate_slice(self):
        return self.gate_slices[self.current_gate_idx]

    def get_scan_slice(self):
        return self.get_previous_slice(self.get_expected_gate_slice())

    def update_progress(self, x, y):
        s = self.get_slice_idx(x, y)
        if s == -1:
            return
        if self.last_slice is None:
            self.last_slice = s
            self.log(f"Initial slice: {s}")
            return
        if s != self.last_slice:
            self.log(f"Slice changed: {self.last_slice} -> {s}")
            self.last_slice = s

    def get_ring_frame(self, x, y):
        r = np.array([x, y], dtype=float) - self.circle_center
        nr = np.linalg.norm(r)
        if nr < 1e-8:
            u_r = np.array([0.0, -1.0], dtype=float)
        else:
            u_r = r / nr
        u_t = np.array([-u_r[1], u_r[0]], dtype=float)
        return u_r, u_t

    def point_at_slice_center(self, slice_idx, radius=None, z=None, tangential_offset=0.0):
        if radius is None:
            radius = self.track_radius
        if z is None:
            z = self.search_height
        angle = (slice_idx % self.num_slices) * self.slice_width
        u_r = np.array([np.cos(angle - np.pi), np.sin(angle - np.pi)], dtype=float)
        u_t = np.array([-u_r[1], u_r[0]], dtype=float)
        xy = self.circle_center + radius * u_r + tangential_offset * u_t
        return np.array([xy[0], xy[1], z], dtype=float)

    def all_saved_gates_ready(self):
        return all(g is not None for g in self.saved_gates)

    # ------------------------------------------------------------------
    # Ring following
    # ------------------------------------------------------------------
    def ring_command(self, sensor_data):
        x = sensor_data["x_global"]
        y = sensor_data["y_global"]
        u_r, u_t = self.get_ring_frame(x, y)
        target_xy = self.circle_center + self.track_radius * u_r + self.search_lookahead * u_t
        yaw_cmd = np.arctan2(u_t[1], u_t[0])
        return [target_xy[0], target_xy[1], self.search_height, yaw_cmd]

    # ------------------------------------------------------------------
    # Scan logic
    # ------------------------------------------------------------------
    def get_scan_pose(self, pose_idx=None, outward_extra=0.0):
        expected_slice = self.get_expected_gate_slice()
        prev_slice = self.get_previous_slice(expected_slice)
        if pose_idx is None:
            pose_idx = self.scan_pose_idx

        pose_type, pose_offset, radial_offset, tangential_offset = self.scan_plan[pose_idx]
        base_slice = prev_slice if pose_type == "prev" else expected_slice
        scan_angle = ((base_slice + pose_offset) * self.slice_width) % (2.0 * np.pi)
        expected_angle = (expected_slice * self.slice_width) % (2.0 * np.pi)

        u_r_scan = np.array([np.cos(scan_angle - np.pi), np.sin(scan_angle - np.pi)], dtype=float)
        u_t_scan = np.array([-u_r_scan[1], u_r_scan[0]], dtype=float)
        u_r_expected = np.array([np.cos(expected_angle - np.pi), np.sin(expected_angle - np.pi)], dtype=float)

        scan_radius = self.track_radius + radial_offset + outward_extra
        hover_xy = self.circle_center + scan_radius * u_r_scan + tangential_offset * u_t_scan
        look_point = self.circle_center + (self.track_radius + 0.20) * u_r_expected
        yaw_gate = np.arctan2(look_point[1] - hover_xy[1], look_point[0] - hover_xy[0])
        return hover_xy, yaw_gate

    def get_scan_outward_extra(self):
        tries = max(self.scan_retry_count, self.approach_fail_count)
        return min(self.scan_max_outward, self.scan_base_outward + self.scan_retry_outward_step * tries)

    def reset_slice_scan(self, outward_extra=None):
        if outward_extra is None:
            outward_extra = self.get_scan_outward_extra()
        self.scan_pose_idx = 0
        self.scan_yaw_idx = 0
        self.scan_height_idx = 0
        self.scan_wait = 0.0
        self.scan_hover_xy, self.scan_base_yaw = self.get_scan_pose(outward_extra=outward_extra)

    def get_slice_scan_command(self):
        if self.scan_hover_xy is None:
            return None
        yaw_cmd = self.wrap_to_pi(self.scan_base_yaw + self.scan_yaw_offsets[self.scan_yaw_idx])
        z_cmd = float(np.clip(self.scan_heights[self.scan_height_idx], 0.75, 1.85))
        return [self.scan_hover_xy[0], self.scan_hover_xy[1], z_cmd, yaw_cmd]

    def advance_slice_scan(self, outward_extra=0.0):
        self.scan_yaw_idx += 1
        if self.scan_yaw_idx < len(self.scan_yaw_offsets):
            return True

        self.scan_yaw_idx = 0
        self.scan_height_idx += 1
        if self.scan_height_idx < len(self.scan_heights):
            return True

        self.scan_height_idx = 0
        self.scan_pose_idx += 1
        if self.scan_pose_idx < len(self.scan_plan):
            self.scan_hover_xy, self.scan_base_yaw = self.get_scan_pose(outward_extra=outward_extra)
            return True
        return False

    def at_setpoint(self, sensor_data, cmd):
        xy_err = np.linalg.norm(np.array([sensor_data["x_global"], sensor_data["y_global"]]) - np.array(cmd[:2]))
        z_err = abs(sensor_data["z_global"] - cmd[2])
        yaw_err = abs(self.wrap_to_pi(sensor_data["yaw"] - cmd[3]))
        return xy_err < self.xy_tol and z_err < self.z_tol and yaw_err < self.yaw_tol

    # ------------------------------------------------------------------
    # Vision
    # ------------------------------------------------------------------
   
    def project_world_point_to_image_norm(self, sensor_data, world_point, image_w, image_h):
        p_c, R_wc = self.get_camera_pose_world(sensor_data)
        v_c = R_wc.T @ (np.array(world_point, dtype=float) - p_c)

        if v_c[2] <= 1e-5:
            return None

        f_pixels = image_w / (2.0 * np.tan(1.5 / 2.0))
        cx = image_w / 2.0 + f_pixels * (v_c[0] / v_c[2])
        cy = image_h / 2.0 + f_pixels * (v_c[1] / v_c[2])

        norm_x = (cx - image_w / 2.0) / (image_w / 2.0)
        norm_y = (cy - image_h / 2.0) / (image_h / 2.0)
        return float(norm_x), float(norm_y)

    def get_expected_gate_image_norm(self, sensor_data, image_w, image_h):
        if self.saved_gates[self.current_gate_idx] is not None:
            expected_point = np.array(self.saved_gates[self.current_gate_idx], dtype=float)
        else:
            expected_slice = self.get_expected_gate_slice()
            z_guess = float(np.clip(sensor_data["z_global"], 0.80, 1.75))
            expected_point = self.point_at_slice_center(
                expected_slice,
                radius=self.track_radius,
                z=z_guess,
            )

        expected_norm = self.project_world_point_to_image_norm(
            sensor_data,
            expected_point,
            image_w,
            image_h,
        )

        if expected_norm is None:
            return None

        if abs(expected_norm[0]) > 2.2 or abs(expected_norm[1]) > 2.2:
            return None

        return expected_norm

    def get_expected_gate_x_side(self, expected_norm):
        if expected_norm is not None:
            ex = float(expected_norm[0])
            if abs(ex) > self.expected_gate_x_side_deadband:
                self.expected_gate_x_side_memory = float(np.sign(ex))
                return self.expected_gate_x_side_memory

        self.expected_gate_x_side_memory = 0.0
        return 0.0

    def filter_candidates_by_expected_x_side(self, candidates, expected_norm):
        side = self.get_expected_gate_x_side(expected_norm)
        if side == 0.0:
            return candidates, side, False

        same_side = [
            det for det in candidates
            if det["norm_x"] * side >= -self.expected_gate_x_side_slack
        ]

        if not same_side:
            return [], side, True

        if expected_norm is not None:
            ex = float(expected_norm[0])
            if abs(ex) >= self.expected_gate_x_strict_deadband:
                min_abs_x = self.expected_gate_x_min_fraction * abs(ex)
                meaningful_same_side = [
                    det for det in same_side
                    if det["norm_x"] * side >= min_abs_x
                ]
                if meaningful_same_side:
                    return meaningful_same_side, side, True
                return [], side, True

        return same_side, side, True

    def score_gate_candidate(self, det, expected_norm=None, locked_detection=None):
        expected_is_clear = False
        if expected_norm is not None:
            expected_is_clear = abs(float(expected_norm[0])) > self.expected_gate_x_side_deadband

        area_weight = self.expected_gate_area_weight if expected_is_clear else self.ambiguous_gate_area_weight
        x_weight = self.expected_gate_x_weight if expected_is_clear else 0.8

        area_bonus = area_weight * det["area"]
        center_penalty = self.expected_gate_center_weight * (
            abs(det["norm_x"]) + 0.5 * abs(det["norm_y"])
        )

        score = area_bonus - center_penalty

        if expected_norm is not None:
            ex, ey = expected_norm
            score -= x_weight * abs(det["norm_x"] - ex)
            score -= self.expected_gate_y_weight * abs(det["norm_y"] - ey)

        if locked_detection is not None:
            dx = det["norm_x"] - locked_detection["norm_x"]
            dy = det["norm_y"] - locked_detection["norm_y"]
            jump = np.hypot(dx, dy)

            area_ratio = det["area"] / max(locked_detection["area"], 1.0)
            area_consistency = -abs(np.log(max(area_ratio, 1e-3)))
            score += -3.0 * jump + 0.20 * area_consistency

        return score

    def choose_candidate_by_visual_lock(self, candidates):
        if self.locked_detection is None:
            return None

        best = None
        best_score = -np.inf
        best_jump = np.inf

        for det in candidates:
            dx = det["norm_x"] - self.locked_detection["norm_x"]
            dy = det["norm_y"] - self.locked_detection["norm_y"]
            jump = float(np.hypot(dx, dy))
            area_ratio = det["area"] / max(self.locked_detection["area"], 1.0)

            if jump > self.lock_follow_max_jump:
                continue
            if area_ratio < self.lock_follow_area_ratio_min:
                continue

            area_consistency = -abs(np.log(max(area_ratio, 1e-3)))
            score = -5.0 * jump + 0.25 * area_consistency

            if score > best_score:
                best_score = score
                best = det
                best_jump = jump

        if best is None:
            return None

        if self.debug and len(candidates) > 1:
            msg = []
            for det in candidates:
                msg.append(f"x={det['norm_x']:.2f}, y={det['norm_y']:.2f}, A={det['area']:.0f}")
            self.log(
                f"Multiple gates visible. visual-lock=on Chose x={best['norm_x']:.2f}, "
                f"y={best['norm_y']:.2f}, jump={best_jump:.2f}. Candidates: " + " | ".join(msg)
            )

        return best

    def detect_gate(self, camera_data, sensor_data=None):
        if camera_data is None:
            return None

        if len(camera_data.shape) == 3 and camera_data.shape[2] == 4:
            bgr = cv2.cvtColor(camera_data, cv2.COLOR_BGRA2BGR)
        else:
            bgr = camera_data.copy()

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lower_pink = np.array([135, 45, 75], dtype=np.uint8)
        upper_pink = np.array([178, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_pink, upper_pink)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        image_h, image_w = mask.shape[:2]
        candidates = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_gate_area:
                continue

            rect = cv2.minAreaRect(c)
            (cx, cy), (w, h), _ = rect
            w = max(w, 1.0)
            h = max(h, 1.0)

            aspect = max(w, h) / min(w, h)
            if aspect > 6.0:
                continue

            norm_x = (cx - image_w / 2.0) / (image_w / 2.0)
            norm_y = (cy - image_h / 2.0) / (image_h / 2.0)

            if abs(norm_x) > self.max_valid_gate_abs_x:
                continue

            candidates.append({
                "cx": float(cx),
                "cy": float(cy),
                "w": float(w),
                "h": float(h),
                "area": float(area),
                "norm_x": float(norm_x),
                "norm_y": float(norm_y),
            })

        if not candidates:
            return None

        
        if self.locked_detection is not None and self.state in ("APPROACH_GATE", "TRACK_GATE"):
            locked_best = self.choose_candidate_by_visual_lock(candidates)
            if locked_best is not None:
                return locked_best
            return None

        expected_norm = None
        if sensor_data is not None:
            expected_norm = self.get_expected_gate_image_norm(sensor_data, image_w, image_h)

        scored_candidates, expected_side, side_filter_used = self.filter_candidates_by_expected_x_side(
            candidates,
            expected_norm,
        )

        if not scored_candidates:
            if self.debug and len(candidates) > 1 and expected_norm is not None:
                msg = []
                for det in candidates:
                    msg.append(f"x={det['norm_x']:.2f}, y={det['norm_y']:.2f}, A={det['area']:.0f}")
                self.log(
                    f"Multiple gates visible. Expected image pos x={expected_norm[0]:.2f}, "
                    f"y={expected_norm[1]:.2f}. side-filter=on No valid candidate on expected side. "
                    "Candidates: " + " | ".join(msg)
                )
            return None

        lock_for_scoring = self.locked_detection
        if expected_side != 0.0 and self.locked_detection is not None:
            if self.locked_detection["norm_x"] * expected_side < -self.expected_gate_x_side_slack:
                lock_for_scoring = None

        best = None
        best_score = -np.inf

        for det in scored_candidates:
            score = self.score_gate_candidate(
                det,
                expected_norm=expected_norm,
                locked_detection=lock_for_scoring,
            )

            if score > best_score:
                best_score = score
                best = det


        if self.debug and len(candidates) > 1:
            msg = []
            for det in candidates:
                msg.append(f"x={det['norm_x']:.2f}, y={det['norm_y']:.2f}, A={det['area']:.0f}")
            if expected_norm is not None:
                side_msg = " side-filter=on" if side_filter_used else " side-filter=off"
                self.log(
                    f"Multiple gates visible. Expected image pos x={expected_norm[0]:.2f}, "
                    f"y={expected_norm[1]:.2f}.{side_msg} Chose x={best['norm_x']:.2f}, "
                    f"y={best['norm_y']:.2f}. Candidates: " + " | ".join(msg)
                )
            else:
                self.log(
                    f"Multiple gates visible. Chose x={best['norm_x']:.2f}, "
                    f"y={best['norm_y']:.2f}. Candidates: " + " | ".join(msg)
                )

        return best

    def detection_candidate(self, det):
        if det is None:
            return False
        return (
            det["area"] >= self.candidate_min_area
            and det["w"] >= self.candidate_min_wh
            and det["h"] >= self.candidate_min_wh
        )

    def detection_good_for_start(self, det):
        if det is None:
            return False
        if det["area"] < self.start_track_area:
            return False
        if abs(det["norm_x"]) > self.max_start_abs_norm_x:
            return False
        if abs(det["norm_y"]) > self.max_start_abs_norm_y:
            return False
        return det["w"] >= self.candidate_min_wh and det["h"] >= self.candidate_min_wh

    def get_detection_to_use(self, det):
        if det is not None:
            if self.locked_detection is None:
                if self.detection_good_for_start(det):
                    self.locked_detection = det.copy()
                    self.lost_frames = 0
                    return self.locked_detection, True
                return None, False

            dx = det["norm_x"] - self.locked_detection["norm_x"]
            dy = det["norm_y"] - self.locked_detection["norm_y"]
            jump = np.hypot(dx, dy)
            area_ratio = det["area"] / max(self.locked_detection["area"], 1.0)

            expected_side = self.expected_gate_x_side_memory
            old_lock_wrong_side = (
                expected_side != 0.0
                and self.locked_detection["norm_x"] * expected_side < -self.expected_gate_x_side_slack
                and det["norm_x"] * expected_side >= -self.expected_gate_x_side_slack
            )

            if (jump < self.lock_jump_tol and area_ratio > self.lock_area_ratio) or old_lock_wrong_side:
                self.locked_detection = det.copy()
                self.lost_frames = 0
                return self.locked_detection, True

            det = None

        if self.locked_detection is not None and self.lost_frames < self.max_lost_frames:
            self.lost_frames += 1
            return self.locked_detection, False

        return None, False

    # ------------------------------------------------------------------
    # Camera geometry / triangulation
    # ------------------------------------------------------------------
    def get_R_world_body(self, roll, pitch, yaw):
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        R_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
        R_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
        R_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
        return R_z @ R_y @ R_x

    def get_camera_pose_world(self, sensor_data):
        R_wb = self.get_R_world_body(sensor_data["roll"], sensor_data["pitch"], sensor_data["yaw"])
        cam_offset_b = np.array([0.03, 0.0, 0.01], dtype=float)
        R_bc = np.array([
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ], dtype=float)
        p_b = self.vec3(sensor_data)
        p_c = p_b + R_wb @ cam_offset_b
        R_wc = R_wb @ R_bc
        return p_c, R_wc

    def pixel_to_camera_ray(self, cx, cy, image_w, image_h):
        f_pixels = image_w / (2.0 * np.tan(1.5 / 2.0))
        u = cx - image_w / 2.0
        v = cy - image_h / 2.0
        ray_c = np.array([u, v, f_pixels], dtype=float)
        ray_c /= np.linalg.norm(ray_c)
        return ray_c

    def build_observation(self, sensor_data, det, camera_data):
        image_h, image_w = camera_data.shape[:2]
        p_c, R_wc = self.get_camera_pose_world(sensor_data)
        ray_c = self.pixel_to_camera_ray(det["cx"], det["cy"], image_w, image_h)
        ray_w = R_wc @ ray_c
        ray_w /= np.linalg.norm(ray_w)
        return {"cam_pos": p_c, "ray_w": ray_w}

    def clear_tracking_memory(self):
        self.locked_detection = None
        self.lost_frames = 0
        self.obs_buffer = []
        self.wrong_tri_count = 0
        self.expected_gate_x_side_memory = 0.0

    def maybe_store_observation(self, obs):
        if not self.obs_buffer:
            self.obs_buffer.append(obs)
            return
        d = np.linalg.norm(obs["cam_pos"] - self.obs_buffer[-1]["cam_pos"])
        if d >= self.min_obs_spacing:
            self.obs_buffer.append(obs)

    def triangulate_from_buffer(self):
        if len(self.obs_buffer) < self.min_obs_for_triangulation:
            return None, None

        cam_positions = np.array([o["cam_pos"] for o in self.obs_buffer], dtype=float)
        baseline = np.max(np.linalg.norm(cam_positions - cam_positions[0], axis=1))
        if baseline < self.min_baseline:
            return None, None

        A = np.zeros((3, 3), dtype=float)
        b = np.zeros(3, dtype=float)
        for obs in self.obs_buffer:
            p = obs["cam_pos"]
            r = obs["ray_w"]
            M = np.eye(3) - np.outer(r, r)
            A += M
            b += M @ p

        try:
            H = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            H = np.linalg.lstsq(A, b, rcond=None)[0]

        dists = []
        for obs in self.obs_buffer:
            p = obs["cam_pos"]
            r = obs["ray_w"]
            dists.append(np.linalg.norm(np.cross(H - p, r)))
        mean_dist = float(np.mean(dists))

        if mean_dist > self.max_point_line_dist:
            return None, mean_dist
        return H, mean_dist

    # ------------------------------------------------------------------
    # Motion commands around the gate
    # ------------------------------------------------------------------
    def center_on_gate_command(self, sensor_data, det, allow_forward=False, tangent_nudge=0.0):
        R_wb = self.get_R_world_body(sensor_data["roll"], sensor_data["pitch"], sensor_data["yaw"])
        p = self.vec3(sensor_data)
        x_b_w = R_wb[:, 0]
        y_b_w = R_wb[:, 1]
        _, u_t = self.get_ring_frame(sensor_data["x_global"], sensor_data["y_global"])

        k_lat = 0.20
        k_z = 0.28
        k_yaw = 0.42
        k_forward = 0.06 if allow_forward and abs(det["norm_x"]) < 0.06 and abs(det["norm_y"]) < 0.08 else 0.0

        tangent_vec = np.array([u_t[0], u_t[1], 0.0], dtype=float)
        target_pos = p + k_forward * x_b_w - k_lat * det["norm_x"] * y_b_w + tangent_nudge * tangent_vec
        z_cmd = float(np.clip(p[2] - k_z * det["norm_y"], 0.75, 1.85))
        yaw_cmd = sensor_data["yaw"] - k_yaw * det["norm_x"]
        return [target_pos[0], target_pos[1], z_cmd, yaw_cmd]

    def approach_gate_command(self, sensor_data, det):
        R_wb = self.get_R_world_body(sensor_data["roll"], sensor_data["pitch"], sensor_data["yaw"])
        p = self.vec3(sensor_data)
        x_b_w = R_wb[:, 0]
        y_b_w = R_wb[:, 1]

        k_lat = 0.18
        k_z = 0.22
        k_yaw = 0.34

        if abs(det["norm_x"]) < 0.45:
            k_forward = 0.09
        elif abs(det["norm_x"]) < 0.70:
            k_forward = 0.04
        else:
            k_forward = 0.0

        target_pos = p + k_forward * x_b_w - k_lat * det["norm_x"] * y_b_w
        z_cmd = float(np.clip(p[2] - k_z * det["norm_y"], 0.75, 1.85))
        yaw_cmd = sensor_data["yaw"] - k_yaw * det["norm_x"]
        return [target_pos[0], target_pos[1], z_cmd, yaw_cmd]

    def regularize_gate_dir(self, H, gate_dir):
        _, ring_tan = self.get_ring_frame(H[0], H[1])

        if gate_dir is None:
            d = ring_tan.copy()
        else:
            d = np.array(gate_dir, dtype=float)
            n = np.linalg.norm(d)
            if n < 1e-6:
                d = ring_tan.copy()
            else:
                d = d / n
            if np.dot(d, ring_tan) < 0:
                d = -d
            d = 0.35 * d + 0.65 * ring_tan
            d = d / np.linalg.norm(d)
        return d

    def prepare_gate_pass(self, sensor_data, gate_center, gate_dir=None):
        H = np.array(gate_center, dtype=float).copy()
        H[2] = float(np.clip(H[2], 0.78, 1.85))
        d = self.regularize_gate_dir(H, gate_dir)

        pre = H.copy()
        pre[:2] = H[:2] - self.pass_pre_dist * d

        post = H.copy()
        post[:2] = H[:2] + self.pass_post_dist * d

        self.pass_center = H
        self.pass_pre_point = pre
        self.pass_post_point = post
        self.pass_dir = d
        self.pass_phase = 0

    def pass_gate_command(self, sensor_data):
        p = self.vec3(sensor_data)
        yaw = sensor_data["yaw"]

        if self.pass_phase == 0:
            target = self.pass_pre_point
            tol = 0.16
            if np.linalg.norm(p - target) < tol or np.dot(p[:2] - self.pass_center[:2], self.pass_dir) > -0.28:
                self.pass_phase = 1
                target = self.pass_center
        elif self.pass_phase == 1:
            target = self.pass_center
            if np.dot(p[:2] - self.pass_center[:2], self.pass_dir) > 0.02 or np.linalg.norm(p - target) < self.pass_center_tol:
                self.pass_phase = 2
                target = self.pass_post_point
        else:
            target = self.pass_post_point
            post_progress = np.dot(p[:2] - self.pass_center[:2], self.pass_dir)
            if post_progress > 0.38 or np.linalg.norm(p - target) < self.pass_post_tol:
                self.log(f"Passed gate {self.current_gate_idx}")
                self.finish_current_gate()
                return self.ring_command(sensor_data)

        vec_xy = target[:2] - p[:2]
        yaw_cmd = yaw if np.linalg.norm(vec_xy) < 1e-6 else np.arctan2(vec_xy[1], vec_xy[0])
        return [target[0], target[1], target[2], yaw_cmd]

    def finish_current_gate(self):
        if self.lap == 0 and self.pass_dir is not None:
            self.saved_gate_dirs[self.current_gate_idx] = self.pass_dir.copy()

        self.state = "SEARCH_RING"
        self.clear_tracking_memory()
        self.scan_hover_xy = None
        self.scan_wait = 0.0
        self.scan_retry_count = 0
        self.approach_fail_count = 0

        self.pass_pre_point = None
        self.pass_center = None
        self.pass_post_point = None
        self.pass_dir = None
        self.pass_phase = 0
        self.expected_gate_x_side_memory = 0.0

        if self.current_gate_idx < self.num_gates - 1:
            self.current_gate_idx += 1
        else:
            self.current_gate_idx = 0
            self.lap += 1
            self.log(f"Finished gate cycle. Lap count = {self.lap}")

    # ------------------------------------------------------------------
    # Timed laps (smooth fast laps)
    # ------------------------------------------------------------------
    def get_timed_start_pose(self):
        gate0 = np.array(self.saved_gates[0], dtype=float)
        z0 = float(np.clip(gate0[2], 0.80, 1.75))
        p = self.point_at_slice_center(0, radius=self.track_radius, z=z0)
        yaw0 = np.arctan2(gate0[1] - p[1], gate0[0] - p[0])
        return p, yaw0

    def build_timed_waypoints(self, start_pos):
        z0 = float(np.clip(start_pos[2], 0.80, 1.75))
        waypoints = [np.array(start_pos, dtype=float)]
        speed_scales = []
        center_waypoint_indices = []
        post_waypoint_indices = []

        def add_point(p, scale_before=1.0):
            speed_scales.append(scale_before)
            waypoints.append(np.array(p, dtype=float))
            return len(waypoints) - 1

        for i in range(self.num_gates):
            H = np.array(self.saved_gates[i], dtype=float)
            H[2] = float(np.clip(H[2], 0.80, 1.75))
            d = self.regularize_gate_dir(H, self.saved_gate_dirs[i])

            corridor = self.timed_gate0_corridor if i == 0 else self.timed_gate_corridor

            pre = H.copy()
            pre[:2] = H[:2] - corridor * d

            post = H.copy()
            post[:2] = H[:2] + corridor * d

            if i == 0:
                pre_far = H.copy()
                pre_far[:2] = H[:2] - (corridor + self.timed_gate0_extra_pre) * d
                add_point(pre_far, scale_before=1.05)
                add_point(pre, scale_before=1.00)
                center_waypoint_indices.append(add_point(H, scale_before=0.95))
                post_waypoint_indices.append(add_point(post, scale_before=1.15))
            else:
                add_point(pre, scale_before=1.25)
                center_waypoint_indices.append(add_point(H, scale_before=1.08))
                post_waypoint_indices.append(add_point(post, scale_before=1.35))

        end_anchor = self.point_at_slice_center(0, radius=self.track_radius, z=z0)
        add_point(end_anchor, scale_before=1.35)

        return [tuple(w) for w in waypoints], speed_scales, center_waypoint_indices, post_waypoint_indices


    def start_timed_lap(self, sensor_data, start_pos):
        waypoints, speed_scales, center_waypoint_indices, post_waypoint_indices = self.build_timed_waypoints(start_pos)

        self.timed_planner = TimedWaypointPlanner(
            waypoints,
            nominal_speed=self.timed_nominal_speed,
            disc_steps=self.timed_disc_steps,
            min_seg_time=self.timed_min_seg_time,
            segment_speed_scales=speed_scales,
        )

        self.timed_start_time = sensor_data["t"]
        self.timed_end_anchor = np.array(waypoints[-1], dtype=float)
        self.timed_path_cursor = 0
        self.timed_gate_idx = 0
        self.timed_gate_center_path_idx = []
        self.timed_gate_post_path_idx = []

        for wi in center_waypoint_indices:
            t_wp = self.timed_planner.times[wi]
            idx = int(np.searchsorted(self.timed_planner.time_setpoints, t_wp))
            idx = min(idx, len(self.timed_planner.trajectory_setpoints) - 1)
            self.timed_gate_center_path_idx.append(idx)

        for wi in post_waypoint_indices:
            t_wp = self.timed_planner.times[wi]
            idx = int(np.searchsorted(self.timed_planner.time_setpoints, t_wp))
            idx = min(idx, len(self.timed_planner.trajectory_setpoints) - 1)
            self.timed_gate_post_path_idx.append(idx)

        self.state = "TIMED_LAP"
        self.log(
            f"Starting timed lap {self.lap + 1}: "
            f"{len(waypoints)} waypoints, planned_time={self.timed_planner.time_setpoints[-1]:.2f} s"
        )

    def finish_timed_lap(self):
        self.log(f"Finished timed lap {self.lap + 1}")
        self.current_gate_idx = 0
        self.lap += 1
        self.state = "SEARCH_RING"
        self.timed_planner = None
        self.timed_start_time = None
        self.timed_end_anchor = None
        self.timed_ready_count = 0
        self.timed_path_cursor = 0
        self.timed_gate_idx = 0
        self.timed_gate_center_path_idx = []
        self.timed_gate_post_path_idx = []

    def timed_command(self, sensor_data):
        p = self.vec3(sensor_data)
        current_slice = self.get_slice_idx(sensor_data["x_global"], sensor_data["y_global"])
        traj = self.timed_planner.trajectory_setpoints

        gate_progress = None
        lateral_err = None
        gate_line_target = None
        max_allowed_idx = len(traj) - 1

        if self.timed_gate_idx < self.num_gates:
            H = np.array(self.saved_gates[self.timed_gate_idx], dtype=float)
            H[2] = float(np.clip(H[2], 0.80, 1.75))
            d = self.regularize_gate_dir(H, self.saved_gate_dirs[self.timed_gate_idx])

            rel = p[:2] - H[:2]
            gate_progress = float(np.dot(rel, d))
            closest_on_gate_axis = H[:2] + gate_progress * d
            lateral_err = float(np.linalg.norm(p[:2] - closest_on_gate_axis))

            release_progress = (
                self.timed_gate0_release_progress
                if self.timed_gate_idx == 0
                else self.timed_gate_release_progress
            )

            if gate_progress > release_progress:
                self.log(f"Timed lap: gate {self.timed_gate_idx} passed.")
                self.timed_gate_idx += 1
            else:
                center_idx = self.timed_gate_center_path_idx[self.timed_gate_idx]
                post_idx = self.timed_gate_post_path_idx[self.timed_gate_idx]

                if self.timed_gate_idx == 0:
                    if gate_progress < -0.08:
                        max_allowed_idx = min(max_allowed_idx, center_idx + 16)
                    else:
                        max_allowed_idx = min(max_allowed_idx, post_idx + 24)
                    forward_progress = np.clip(gate_progress + 0.45, 0.04, 0.62)
                else:
                    if gate_progress < -0.08:
                        max_allowed_idx = min(max_allowed_idx, center_idx + 24)
                    else:
                        max_allowed_idx = min(max_allowed_idx, post_idx + 38)
                    forward_progress = np.clip(gate_progress + 0.55, 0.06, 0.72)

                gate_line_target = H.copy()
                gate_line_target[:2] = H[:2] + forward_progress * d

        lo = self.timed_path_cursor
        hi = min(len(traj), self.timed_path_cursor + self.timed_search_window, max_allowed_idx + 1)
        local = traj[lo:hi, :3]

        if len(local) == 0:
            lo = min(self.timed_path_cursor, len(traj) - 1)
            local = traj[lo:lo + 1, :3]

        dists = np.linalg.norm(local - p, axis=1)
        best_local = int(np.argmin(dists))
        best_idx = lo + best_local

        if best_idx > self.timed_path_cursor:
            self.timed_path_cursor = best_idx

        err = float(dists[best_local])

        if err > self.timed_err_zero:
            lookahead_pts = self.timed_lookahead_pts_slow
        elif err > self.timed_err_reduce:
            lookahead_pts = int(0.70 * self.timed_lookahead_pts)
        else:
            lookahead_pts = self.timed_lookahead_pts

        if self.timed_gate_idx == 0:
            lookahead_pts = min(lookahead_pts, self.timed_gate0_lookahead_pts)

        ref_idx = min(self.timed_path_cursor + lookahead_pts, max_allowed_idx, len(traj) - 1)
        ref = traj[ref_idx].copy()

        if self.timed_gate_idx < self.num_gates:
            release_progress = (
                self.timed_gate0_release_progress
                if self.timed_gate_idx == 0
                else self.timed_gate_release_progress
            )
            attract_radius = (
                self.timed_gate0_attract_radius
                if self.timed_gate_idx == 0
                else self.timed_gate_attract_radius
            )
            attract_gain = (
                self.timed_gate0_attract_gain
                if self.timed_gate_idx == 0
                else self.timed_gate_attract_gain
            )
            lower_progress = -1.20 if self.timed_gate_idx == 0 else -0.85
        else:
            release_progress = self.timed_gate_release_progress
            attract_radius = self.timed_gate_attract_radius
            attract_gain = self.timed_gate_attract_gain
            lower_progress = -0.85

        if (
            self.timed_gate_idx < self.num_gates
            and gate_line_target is not None
            and gate_progress is not None
            and lateral_err is not None
            and lower_progress < gate_progress < release_progress
            and lateral_err < attract_radius
        ):
            closeness = 1.0 - np.clip(lateral_err / attract_radius, 0.0, 1.0)
            w = attract_gain * (0.40 + 0.60 * closeness)
            ref[:3] = (1.0 - w) * ref[:3] + w * gate_line_target[:3]
            ref[3] = np.arctan2(gate_line_target[1] - p[1], gate_line_target[0] - p[0])

        if (
            self.timed_gate_idx >= self.num_gates
            and current_slice == 0
            and np.linalg.norm(p - self.timed_end_anchor) < self.timed_finish_tol
        ):
            self.finish_timed_lap()
            return self.ring_command(sensor_data)

        if self.timed_gate_idx >= self.num_gates and ref_idx >= len(traj) - 5:
            yaw_to_end = np.arctan2(
                self.timed_end_anchor[1] - p[1],
                self.timed_end_anchor[0] - p[0],
            )
            return [self.timed_end_anchor[0], self.timed_end_anchor[1], self.timed_end_anchor[2], yaw_to_end]

        return [ref[0], ref[1], ref[2], ref[3]]

    # ------------------------------------------------------------------
    # Main controller
    # ------------------------------------------------------------------
    def compute_command(self, sensor_data, camera_data, dt):
        x = sensor_data["x_global"]
        y = sensor_data["y_global"]
        z = sensor_data["z_global"]
        yaw = sensor_data["yaw"]
        p = np.array([x, y, z], dtype=float)

        if z < 0.5:
            return [x, y, self.takeoff_height, yaw]

        self.update_progress(x, y)
        current_slice = self.get_slice_idx(x, y)
        gate_detection = self.detect_gate(camera_data, sensor_data)

        if self.state == "TIMED_LAP":
            return self.timed_command(sensor_data)

        expected_slice = self.get_expected_gate_slice()
        scan_slice = self.get_scan_slice()

        if self.state == "SEARCH_RING":
            if (
                self.use_timed_laps
                and self.lap in [1, 2]
                and self.current_gate_idx == 0
                and self.all_saved_gates_ready()
            ):
                start_anchor, start_yaw = self.get_timed_start_pose()
                pos_err = np.linalg.norm(p - start_anchor)
                yaw_err = abs(self.wrap_to_pi(yaw - start_yaw))
                speed_xy = np.hypot(sensor_data["v_x"], sensor_data["v_y"])
                speed_z = abs(sensor_data["v_z"])
                yaw_rate = abs(sensor_data["rate_yaw"])

                ready_pose = pos_err < 0.14 and yaw_err < 0.16
                ready_motion = speed_xy < 0.35 and speed_z < 0.14 and yaw_rate < 0.70

                if ready_pose and ready_motion:
                    self.timed_ready_count += 1
                else:
                    self.timed_ready_count = 0
                    return [start_anchor[0], start_anchor[1], start_anchor[2], start_yaw]

                if self.timed_ready_count < 1:
                    return [start_anchor[0], start_anchor[1], start_anchor[2], start_yaw]

                self.timed_ready_count = 0
                self.start_timed_lap(sensor_data, start_anchor)
                return self.timed_command(sensor_data)

            if current_slice == scan_slice:
                if self.lap >= 1 and self.saved_gates[self.current_gate_idx] is not None:
                    self.prepare_gate_pass(
                        sensor_data,
                        self.saved_gates[self.current_gate_idx],
                        self.saved_gate_dirs[self.current_gate_idx],
                    )
                    self.state = "PASS_GATE"
                    self.log(f"Reached previous slice {scan_slice} for gate {self.current_gate_idx}. Using saved gate.")
                    return self.pass_gate_command(sensor_data)

                self.reset_slice_scan()
                self.clear_tracking_memory()
                self.state = "SCAN_GATE"
                self.log(f"Reached previous slice {scan_slice} for gate {self.current_gate_idx}. Stopping and scanning.")
                return self.get_slice_scan_command()

            return self.ring_command(sensor_data)

        if self.state == "SCAN_GATE":
            cmd = self.get_slice_scan_command()
            if cmd is None:
                self.reset_slice_scan()
                cmd = self.get_slice_scan_command()

            if not self.at_setpoint(sensor_data, cmd):
                self.scan_wait = 0.0
                return cmd

            if self.detection_good_for_start(gate_detection):
                self.clear_tracking_memory()
                self.locked_detection = gate_detection.copy()
                self.tri_motion_sign = 1.0 if (self.current_gate_idx % 2 == 0) else -1.0
                self.state = "TRACK_GATE"
                self.log(
                    f"Detected gate {self.current_gate_idx} during scan "
                    f"(pose idx {self.scan_pose_idx}, height idx {self.scan_height_idx}, yaw idx {self.scan_yaw_idx})."
                )
                return self.center_on_gate_command(sensor_data, gate_detection, allow_forward=False)

            if self.detection_candidate(gate_detection):
                self.locked_detection = gate_detection.copy()
                self.lost_frames = 0
                self.state = "APPROACH_GATE"
                self.log(
                    f"Candidate gate {self.current_gate_idx} seen during scan "
                    f"(pose idx {self.scan_pose_idx}, height idx {self.scan_height_idx}, yaw idx {self.scan_yaw_idx}). Centering before approach."
                )
                return self.center_on_gate_command(sensor_data, gate_detection, allow_forward=False)

            self.scan_wait += dt
            if self.scan_wait >= self.scan_dwell:
                self.scan_wait = 0.0
                outward = self.get_scan_outward_extra()
                has_more = self.advance_slice_scan(outward_extra=outward)
                if not has_more:
                    self.scan_retry_count += 1
                    outward = self.get_scan_outward_extra()
                    self.log(
                        f"No gate found for gate {self.current_gate_idx} after full scan cycle. "
                        f"Retrying with outward_extra={outward:.2f}."
                    )
                    self.reset_slice_scan(outward_extra=outward)
            return cmd

        if self.state == "APPROACH_GATE":
            allowed_approach_slices = {scan_slice, expected_slice}
            s_tmp = scan_slice
            for _ in range(self.max_track_backward_slices):
                s_tmp = self.get_previous_slice(s_tmp)
                allowed_approach_slices.add(s_tmp)

            if current_slice not in allowed_approach_slices:
                self.approach_fail_count += 1
                self.state = "SCAN_GATE"
                self.clear_tracking_memory()
                self.reset_slice_scan()
                return self.get_slice_scan_command()

            if self.detection_candidate(gate_detection):
                self.locked_detection = gate_detection.copy()
                self.lost_frames = 0
                if self.detection_good_for_start(gate_detection):
                    self.approach_fail_count = 0
                    self.clear_tracking_memory()
                    self.locked_detection = gate_detection.copy()
                    self.tri_motion_sign = 1.0 if (self.current_gate_idx % 2 == 0) else -1.0
                    self.state = "TRACK_GATE"
                    self.log(f"Approach successful for gate {self.current_gate_idx}. Switching to TRACK_GATE.")
                    return self.center_on_gate_command(sensor_data, gate_detection, allow_forward=False)

                centered_enough = abs(gate_detection["norm_x"]) < 0.12 and abs(gate_detection["norm_y"]) < 0.14
                return self.center_on_gate_command(sensor_data, gate_detection, allow_forward=centered_enough)

            self.lost_frames += 1
            if self.lost_frames <= self.max_lost_frames and self.locked_detection is not None:
                centered_enough = abs(self.locked_detection["norm_x"]) < 0.12 and abs(self.locked_detection["norm_y"]) < 0.14
                return self.center_on_gate_command(sensor_data, self.locked_detection, allow_forward=centered_enough)

            self.approach_fail_count += 1
            self.log(
                f"Lost candidate during approach. Returning to farther outward scan "
                f"(fail {self.approach_fail_count}, outward_extra={self.get_scan_outward_extra():.2f})."
            )
            self.state = "SCAN_GATE"
            self.clear_tracking_memory()
            self.reset_slice_scan()
            return self.get_slice_scan_command()

        if self.state == "TRACK_GATE":
            allowed_track_slices = {scan_slice, expected_slice}
            s_tmp = scan_slice
            for _ in range(self.max_track_backward_slices):
                s_tmp = self.get_previous_slice(s_tmp)
                allowed_track_slices.add(s_tmp)

            if current_slice not in allowed_track_slices and gate_detection is None:
                self.log("Drifted out of the gate scan region while tracking and lost visual contact. Pulling back to scan pose.")
                self.approach_fail_count += 1
                self.state = "SCAN_GATE"
                self.reset_slice_scan()
                return self.get_slice_scan_command()

            det_use, is_fresh = self.get_detection_to_use(gate_detection)
            if det_use is None:
                self.log("Lost gate while tracking. Returning to scan.")
                self.approach_fail_count += 1
                self.state = "SCAN_GATE"
                self.clear_tracking_memory()
                self.reset_slice_scan()
                return self.get_slice_scan_command()

            rough_centered = (
                abs(det_use["norm_x"]) < self.obs_collect_tol_x
                and abs(det_use["norm_y"]) < self.obs_collect_tol_y
            )
            tangent_nudge = 0.0

            if rough_centered:
                obs = self.build_observation(sensor_data, det_use, camera_data)
                if is_fresh:
                    self.maybe_store_observation(obs)
                H, mean_dist = self.triangulate_from_buffer()
                if H is not None:
                    tri_slice = self.get_slice_idx(H[0], H[1])
                    prev_expected = self.get_previous_slice(expected_slice)
                    self.log(
                        f"Gate {self.current_gate_idx}: triangulated point = ({H[0]:.2f}, {H[1]:.2f}, {H[2]:.2f}), "
                        f"triangulated slice = {tri_slice}, expected slice = {expected_slice}"
                    )
                    if tri_slice in (prev_expected, expected_slice):
                        self.wrong_tri_count = 0
                        H[2] = float(np.clip(H[2], 0.78, 1.85))
                        self.saved_gates[self.current_gate_idx] = H.copy()

                        self.prepare_gate_pass(sensor_data, H, None)
                        self.state = "PASS_GATE"
                        self.log(f"Triangulated gate {self.current_gate_idx} at {H}, mean point-line distance = {mean_dist:.3f}")
                        return self.pass_gate_command(sensor_data)

                    self.wrong_tri_count += 1
                    self.obs_buffer = []
                    self.log(
                        f"Rejected triangulation for gate {self.current_gate_idx}: "
                        f"got slice {tri_slice}, expected {prev_expected} or {expected_slice} "
                        f"(bad count {self.wrong_tri_count}/{self.wrong_tri_limit})."
                    )

                    if self.wrong_tri_count >= self.wrong_tri_limit:
                        self.log("Wrong gate lock detected. Clearing visual lock and returning to scan.")
                        self.approach_fail_count += 1
                        self.state = "SCAN_GATE"
                        self.clear_tracking_memory()
                        self.reset_slice_scan()
                        return self.get_slice_scan_command()

                tangent_nudge = 0.05 * self.tri_motion_sign

            self.log(
                f"Tracking gate {self.current_gate_idx}: norm_x={det_use['norm_x']:.2f}, norm_y={det_use['norm_y']:.2f}, "
                f"fresh={is_fresh}, obs={len(self.obs_buffer)}"
            )
            return self.center_on_gate_command(sensor_data, det_use, allow_forward=False, tangent_nudge=tangent_nudge)

        if self.state == "PASS_GATE":
            return self.pass_gate_command(sensor_data)

        return self.ring_command(sensor_data)


_controller = MyAssignment()


def get_command(sensor_data, camera_data, dt):
    return _controller.compute_command(sensor_data, camera_data, dt)
