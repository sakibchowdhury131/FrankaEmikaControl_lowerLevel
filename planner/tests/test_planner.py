"""
planner/tests/test_planner.py
==============================
Offline unit tests — no robot, no Pinocchio, no URDF needed.

Run:  cd planner && python -m pytest tests/ -v
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from trajectory_planner import (
    N_JOINTS, Q_MIN, Q_MAX, V_MAX, A_MAX, J_MAX,
    compute_T_min, poly5_coeffs, sample_poly5,
    build_trajectory, build_via_trajectory,
    export_trajectory, TrajectoryResult,
    _validate,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Boundary conditions
# ─────────────────────────────────────────────────────────────────────────────

class TestBoundaryConditions:

    @pytest.fixture
    def single(self):
        q0, qf, T = -0.5, 1.1, 3.0
        return poly5_coeffs(q0, qf, T), q0, qf, T

    def test_start_position(self, single):
        c, q0, _, T = single
        pos, _, _ = sample_poly5(c, np.array([0.0]))
        assert abs(pos[0] - q0) < 1e-10

    def test_end_position(self, single):
        c, _, qf, T = single
        pos, _, _ = sample_poly5(c, np.array([T]))
        assert abs(pos[0] - qf) < 1e-10

    def test_start_velocity_zero(self, single):
        c, _, _, _ = single
        _, vel, _ = sample_poly5(c, np.array([0.0]))
        assert abs(vel[0]) < 1e-10

    def test_end_velocity_zero(self, single):
        c, _, _, T = single
        _, vel, _ = sample_poly5(c, np.array([T]))
        assert abs(vel[0]) < 1e-10

    def test_start_acceleration_zero(self, single):
        c, _, _, _ = single
        _, _, acc = sample_poly5(c, np.array([0.0]))
        assert abs(acc[0]) < 1e-10

    def test_end_acceleration_zero(self, single):
        c, _, _, T = single
        _, _, acc = sample_poly5(c, np.array([T]))
        assert abs(acc[0]) < 1e-10

    def test_zero_displacement_all_zero_except_a0(self):
        c = poly5_coeffs(1.5, 1.5, 2.0)
        assert abs(c[0] - 1.5) < 1e-12
        for ci in c[1:]:
            assert abs(ci) < 1e-12

    def test_velocity_symmetric_about_midpoint(self):
        c = poly5_coeffs(0.0, 1.0, 4.0)
        t = np.linspace(0, 4.0, 401)
        _, vel, _ = sample_poly5(c, t)
        np.testing.assert_allclose(vel, vel[::-1], atol=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
#  Kinematic limit enforcement
# ─────────────────────────────────────────────────────────────────────────────

class TestKinematicLimits:

    @pytest.fixture
    def traj(self):
        q0 = np.zeros(N_JOINTS)
        qf = np.array([0.5, 0.3, -0.2, 0.4, -0.1, 0.6, 0.2])
        T  = compute_T_min(qf - q0)
        t_arr, Q, Qd, Qdd = build_trajectory(q0, qf, T, hz=1000)
        return q0, qf, T, t_arr, Q, Qd, Qdd

    def test_velocity_within_limits(self, traj):
        *_, Qd, _ = traj
        assert np.all(np.max(np.abs(Qd), axis=0) <= V_MAX * 1.001)

    def test_acceleration_within_limits(self, traj):
        *_, Qdd = traj
        assert np.all(np.max(np.abs(Qdd), axis=0) <= A_MAX * 1.001)

    def test_jerk_within_limits(self, traj):
        *_, t_arr, _, _, Qdd = traj
        dt   = t_arr[1] - t_arr[0]
        jerk = np.diff(Qdd, axis=0) / dt
        assert np.all(np.max(np.abs(jerk), axis=0) <= J_MAX * 1.05)

    def test_start_and_end_positions(self, traj):
        q0, qf, _, _, Q, _, _ = traj
        np.testing.assert_allclose(Q[0],  q0, atol=1e-8)
        np.testing.assert_allclose(Q[-1], qf, atol=1e-8)

    def test_T_scales_with_displacement(self):
        q0 = np.zeros(N_JOINTS)
        T1 = compute_T_min(np.ones(N_JOINTS) * 0.1)
        T2 = compute_T_min(np.ones(N_JOINTS) * 1.0)
        assert T2 > T1

    def test_velocity_peak_near_midpoint(self):
        q0 = np.zeros(N_JOINTS)
        qf = np.ones(N_JOINTS)
        T  = compute_T_min(qf - q0)
        t_arr, _, Qd, _ = build_trajectory(q0, qf, T, hz=2000)
        for j in range(N_JOINTS):
            idx   = np.argmax(np.abs(Qd[:, j]))
            t_pek = t_arr[idx]
            assert abs(t_pek - T / 2) < 0.02 * T


# ─────────────────────────────────────────────────────────────────────────────
#  Analytic peak values
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyticPeaks:

    def _run(self, dq_j0):
        q0 = np.zeros(N_JOINTS)
        qf = np.zeros(N_JOINTS)
        qf[0] = dq_j0
        T = compute_T_min(qf - q0)
        t_arr, _, Qd, Qdd = build_trajectory(q0, qf, T, hz=5000)
        return T, np.max(np.abs(Qd[:, 0])), np.max(np.abs(Qdd[:, 0]))

    def test_velocity_peak_formula(self):
        dq = 0.8
        T, v_peak, _ = self._run(dq)
        expected = 15 * abs(dq) / (8 * T)
        assert abs(v_peak - expected) / expected < 0.005

    def test_acceleration_peak_formula(self):
        dq = 0.8
        T, _, a_peak = self._run(dq)
        expected = 10 * np.sqrt(3) * abs(dq) / (3 * T**2)
        assert abs(a_peak - expected) / expected < 0.01


# ─────────────────────────────────────────────────────────────────────────────
#  Via-point trajectory
# ─────────────────────────────────────────────────────────────────────────────

class TestViaTrajectory:

    def test_passes_through_via(self):
        q0  = np.zeros(N_JOINTS)
        via = np.ones(N_JOINTS) * 0.3
        qf  = np.ones(N_JOINTS) * 0.6
        T   = compute_T_min(qf - q0) * 2
        t_arr, Q, Qd, _ = build_via_trajectory(q0, via, qf, T, hz=1000)
        mid = len(t_arr) // 2
        np.testing.assert_allclose(Q[mid], via, atol=1e-6)

    def test_zero_velocity_at_via(self):
        q0  = np.zeros(N_JOINTS)
        via = np.ones(N_JOINTS) * 0.3
        qf  = np.ones(N_JOINTS) * 0.6
        T   = compute_T_min(qf - q0) * 2
        t_arr, _, Qd, _ = build_via_trajectory(q0, via, qf, T, hz=1000)
        # The via-point sits at t = T/2. Because we sample at exactly 1 kHz,
        # the sample nearest T/2 may be up to 0.5 ms away from the true
        # zero-velocity instant, giving a residual velocity of O(1e-5) rad/s.
        # Find the sample closest to T/2 and check it is near-zero.
        mid_t   = T / 2.0
        mid_idx = np.argmin(np.abs(t_arr - mid_t))
        np.testing.assert_allclose(Qd[mid_idx], np.zeros(N_JOINTS), atol=1e-4)

    def test_start_end_positions(self):
        q0  = np.zeros(N_JOINTS)
        via = np.ones(N_JOINTS) * 0.2
        qf  = np.ones(N_JOINTS) * 0.5
        T   = compute_T_min(qf - q0) * 2
        t_arr, Q, _, _ = build_via_trajectory(q0, via, qf, T, hz=1000)
        np.testing.assert_allclose(Q[0],  q0, atol=1e-8)
        np.testing.assert_allclose(Q[-1], qf, atol=1e-8)


# ─────────────────────────────────────────────────────────────────────────────
#  CSV export / reload
# ─────────────────────────────────────────────────────────────────────────────

class TestCSVExport:

    def test_csv_roundtrip(self, tmp_path):
        q0 = np.zeros(N_JOINTS)
        qf = np.array([0.3, -0.2, 0.1, -0.5, 0.2, 0.4, -0.1])
        T  = compute_T_min(qf - q0)
        t_arr, Q, Qd, Qdd = build_trajectory(q0, qf, T, hz=100)
        tau = np.random.randn(*Q.shape) * 5.0
        w   = np.ones(len(t_arr))
        sm  = np.ones(len(t_arr)) * 0.1

        result = TrajectoryResult(
            T=T, t_arr=t_arr, Q=Q, Qd=Qd, Qdd=Qdd,
            tau=tau, w=w, sigma_min=sm
        )

        path = str(tmp_path / "traj.csv")
        export_trajectory(result, path)

        data = np.loadtxt(path, delimiter=",", skiprows=1)
        assert data.shape == (len(t_arr), 29)
        np.testing.assert_allclose(data[:, 0],    t_arr, atol=1e-10)
        np.testing.assert_allclose(data[:, 1:8],  Q,     atol=1e-10)
        np.testing.assert_allclose(data[:, 8:15], Qd,    atol=1e-10)
        np.testing.assert_allclose(data[:, 22:],  tau,   atol=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
#  Input validation
# ─────────────────────────────────────────────────────────────────────────────

class TestValidation:

    def test_joint_out_of_range_raises(self):
        bad = np.array([10.0, 0, 0, 0, 0, 0, 0])
        with pytest.raises(ValueError, match="outside"):
            _validate(bad, "test")

    def test_wrong_number_of_joints_raises(self):
        with pytest.raises(ValueError, match="7 elements"):
            _validate(np.zeros(6), "test")

    def test_valid_config_passes(self):
        # Panda ready pose — joint 4 must be negative (range [-3.07, -0.07])
        ready = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        _validate(ready, "ready_pose")
        _validate(Q_MIN, "qmin")
        _validate(Q_MAX, "qmax")
