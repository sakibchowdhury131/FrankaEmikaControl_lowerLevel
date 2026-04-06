"""
tests/test_poly5_general.py
============================
Tests for the general 5th-order polynomial.
Run: python -m pytest tests/ -v
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from poly5_general import (
    poly5_coeffs_general,
    sample_poly5,
    verify_boundary_conditions,
    find_T_min,
    check_feasible,
)
from trajectory_builder import V_MAX, A_MAX, J_MAX, N_JOINTS


# ─────────────────────────────────────────────────────────────────────────────
#  Boundary condition tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBoundaryConditions:

    @pytest.mark.parametrize("q0,qf,v0,vf,a0,af,T,label", [
        ( 0.0,  1.0, 0.0,  0.0, 0.0,  0.0, 2.0, "rest-to-rest"),
        ( 0.0,  1.0, 0.0,  0.5, 0.0,  0.0, 2.0, "rest-to-non-rest"),
        ( 0.0,  1.0, 0.5,  0.0, 0.0,  0.0, 2.0, "non-rest-to-rest"),
        ( 0.0,  1.0, 0.3,  0.4, 0.1, -0.1, 3.0, "general"),
        (-0.5,  0.8, 0.0,  0.3, 0.0,  0.0, 1.5, "asymmetric"),
        ( 1.0, -0.5, 0.2, -0.2, 0.0,  0.0, 2.5, "negative displacement"),
        ( 0.0,  0.0, 0.3,  0.3, 0.0,  0.0, 1.0, "zero displacement nonzero vel"),
    ])
    def test_boundary_conditions_satisfied(self, q0, qf, v0, vf, a0, af, T, label):
        c = poly5_coeffs_general(q0, qf, v0, vf, a0, af, T)
        ok = verify_boundary_conditions(c, T, q0, qf, v0, vf, a0, af, tol=1e-8)
        assert ok, f"Boundary conditions failed for: {label}"

    def test_collapses_to_rest_to_rest(self):
        """General formula must equal rest-to-rest when v0=vf=a0=af=0."""
        q0, qf, T = -0.5, 1.2, 3.0
        dq = qf - q0
        c_gen = poly5_coeffs_general(q0, qf, 0, 0, 0, 0, T)
        # Known rest-to-rest coefficients
        c_rtr = np.array([q0, 0, 0,
                          10*dq/T**3, -15*dq/T**4, 6*dq/T**5])
        np.testing.assert_allclose(c_gen, c_rtr, atol=1e-10)

    def test_start_pos_correct(self):
        c = poly5_coeffs_general(0.5, 1.2, 0.3, -0.1, 0.0, 0.0, 2.0)
        pos, _, _ = sample_poly5(c, np.array([0.0]))
        assert abs(pos[0] - 0.5) < 1e-10

    def test_end_pos_correct(self):
        T = 2.0
        c = poly5_coeffs_general(0.5, 1.2, 0.3, -0.1, 0.0, 0.0, T)
        pos, _, _ = sample_poly5(c, np.array([T]))
        assert abs(pos[0] - 1.2) < 1e-10

    def test_start_vel_correct(self):
        c = poly5_coeffs_general(0.0, 1.0, 0.7, 0.3, 0.1, 0.0, 2.0)
        _, vel, _ = sample_poly5(c, np.array([0.0]))
        assert abs(vel[0] - 0.7) < 1e-10

    def test_end_vel_correct(self):
        T = 2.0
        c = poly5_coeffs_general(0.0, 1.0, 0.0, 0.5, 0.0, 0.0, T)
        _, vel, _ = sample_poly5(c, np.array([T]))
        assert abs(vel[0] - 0.5) < 1e-10

    def test_start_acc_correct(self):
        c = poly5_coeffs_general(0.0, 1.0, 0.0, 0.0, 0.4, 0.0, 2.0)
        _, _, acc = sample_poly5(c, np.array([0.0]))
        assert abs(acc[0] - 0.4) < 1e-10

    def test_end_acc_correct(self):
        T = 2.0
        c = poly5_coeffs_general(0.0, 1.0, 0.0, 0.0, 0.0, 0.2, T)
        _, _, acc = sample_poly5(c, np.array([T]))
        assert abs(acc[0] - 0.2) < 1e-10

    def test_nonzero_acc_both_ends(self):
        T = 3.0
        c = poly5_coeffs_general(0.0, 1.0, 0.3, 0.4, 0.1, -0.1, T)
        ok = verify_boundary_conditions(c, T, 0.0, 1.0, 0.3, 0.4, 0.1, -0.1)
        assert ok


# ─────────────────────────────────────────────────────────────────────────────
#  Bisection tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBisection:

    def test_rest_to_rest_feasible(self):
        T = find_T_min(0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                       V_MAX[0], A_MAX[0], J_MAX[0])
        assert check_feasible(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, T,
                              V_MAX[0], A_MAX[0], J_MAX[0])

    def test_rest_to_non_rest_feasible(self):
        vf = 0.8
        T = find_T_min(0.0, 1.0, 0.0, vf, 0.0, 0.0,
                       V_MAX[0], A_MAX[0], J_MAX[0])
        assert check_feasible(0.0, 1.0, 0.0, vf, 0.0, 0.0, T,
                              V_MAX[0], A_MAX[0], J_MAX[0])

    def test_non_rest_to_rest_feasible(self):
        v0 = 0.6
        T = find_T_min(0.0, 1.0, v0, 0.0, 0.0, 0.0,
                       V_MAX[0], A_MAX[0], J_MAX[0])
        assert check_feasible(0.0, 1.0, v0, 0.0, 0.0, 0.0, T,
                              V_MAX[0], A_MAX[0], J_MAX[0])

    def test_T_larger_for_larger_displacement(self):
        T_small = find_T_min(0.0, 0.5, 0.0, 0.0, 0.0, 0.0,
                            V_MAX[0], A_MAX[0], J_MAX[0])
        T_large = find_T_min(0.0, 2.0, 0.0, 0.0, 0.0, 0.0,
                            V_MAX[0], A_MAX[0], J_MAX[0])
        assert T_large >= T_small

    def test_velocity_within_limits(self):
        T = find_T_min(0.0, 1.0, 0.0, 0.5, 0.0, 0.0,
                       V_MAX[0], A_MAX[0], J_MAX[0])
        c = poly5_coeffs_general(0.0, 1.0, 0.0, 0.5, 0.0, 0.0, T)
        t = np.linspace(0, T, 5000)
        _, vel, _ = sample_poly5(c, t)
        assert np.max(np.abs(vel)) <= V_MAX[0] * 1.01

    def test_acceleration_within_limits(self):
        T = find_T_min(0.0, 1.0, 0.0, 0.5, 0.0, 0.0,
                       V_MAX[0], A_MAX[0], J_MAX[0])
        c = poly5_coeffs_general(0.0, 1.0, 0.0, 0.5, 0.0, 0.0, T)
        t = np.linspace(0, T, 5000)
        _, _, acc = sample_poly5(c, t)
        assert np.max(np.abs(acc)) <= A_MAX[0] * 1.01

    def test_infeasible_smaller_T_rejected(self):
        """T slightly below T_min should be infeasible."""
        T_min = find_T_min(0.0, 1.0, 0.0, 0.5, 0.0, 0.0,
                           V_MAX[0], A_MAX[0], J_MAX[0])
        # T_min already has 5% margin, so T_min/1.06 should be infeasible
        assert not check_feasible(0.0, 1.0, 0.0, 0.5, 0.0, 0.0,
                                  T_min / 1.06,
                                  V_MAX[0], A_MAX[0], J_MAX[0])
