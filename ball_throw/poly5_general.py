"""
poly5_general.py
================
Fifth-order polynomial trajectory generation for the general case:
  - rest-to-non-rest  (q0,0,0) → (qf, vf, af)
  - non-rest-to-rest  (q0,v0,a0) → (qf, 0, 0)
  - general           (q0,v0,a0) → (qf, vf, af)

Coefficients derived in closed form from the 6×6 boundary condition system.
T_min found via bisection search (monotone feasibility in T).

All single-joint functions. Multi-joint wrappers are in trajectory_builder.py.
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Coefficient computation
# ─────────────────────────────────────────────────────────────────────────────

def poly5_coeffs_general(q0: float, qf: float,
                          v0: float, vf: float,
                          a0: float, af: float,
                          T:  float) -> np.ndarray:
    """
    Closed-form coefficients [a0..a5] for the general 5th-order polynomial.

    Boundary conditions:
        q(0)=q0,  q(T)=qf
        q'(0)=v0, q'(T)=vf
        q''(0)=a0, q''(T)=af

    Derivation (via 3×3 reduced system after substituting t=0 BCs):

        a0_coeff = q0
        a1_coeff = v0
        a2_coeff = a0 / 2

        Δq = qf - q0 - v0·T - (a0/2)·T²   (residual displacement)

        c1 = Δq / T³
        c2 = (vf - v0 - a0·T) / T²
        c3 = (af - a0) / T

        u = 10·c1 - 4·c2 + (1/2)·c3   → a3
        v = -15·c1 + 7·c2 - c3         → a4·T  → a4 = v/T
        w = 6·c1 - 3·c2 + (1/2)·c3    → a5·T² → a5 = w/T²
    """
    # t=0 boundary conditions give first three coefficients directly
    a0c = q0
    a1c = v0
    a2c = a0 / 2.0

    # Residual displacement (what's left after initial v0, a0 have acted)
    dq = qf - q0 - v0 * T - 0.5 * a0 * T**2

    # Normalised RHS constants
    c1 = dq          / T**3
    c2 = (vf - v0 - a0 * T) / T**2
    c3 = (af - a0)   / T

    # Solve 3×3 system (see derivation in docs)
    u =  10.0 * c1 - 4.0 * c2 + 0.5 * c3
    v = -15.0 * c1 + 7.0 * c2 - 1.0 * c3
    w =   6.0 * c1 - 3.0 * c2 + 0.5 * c3

    a3c = u
    a4c = v / T
    a5c = w / T**2

    return np.array([a0c, a1c, a2c, a3c, a4c, a5c])


# ─────────────────────────────────────────────────────────────────────────────
#  Sampling
# ─────────────────────────────────────────────────────────────────────────────

def sample_poly5(c: np.ndarray,
                 t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate position, velocity, acceleration for coefficient vector c=[a0..a5].
    """
    a0, a1, a2, a3, a4, a5 = c
    pos =  a0 + a1*t   + a2*t**2  + a3*t**3    + a4*t**4    + a5*t**5
    vel =       a1     + 2*a2*t   + 3*a3*t**2   + 4*a4*t**3  + 5*a5*t**4
    acc =               2*a2      + 6*a3*t      + 12*a4*t**2  + 20*a5*t**3
    return pos, vel, acc


# ─────────────────────────────────────────────────────────────────────────────
#  Feasibility check and T_min bisection
# ─────────────────────────────────────────────────────────────────────────────

def check_feasible(q0: float, qf: float,
                   v0: float, vf: float,
                   a0: float, af: float,
                   T:  float,
                   v_max: float, a_max: float, j_max: float,
                   hz: int = 2000) -> bool:
    """
    Return True if the polynomial with duration T satisfies all limits.
    Samples at hz to find peaks — higher hz = more accurate but slower.
    """
    c = poly5_coeffs_general(q0, qf, v0, vf, a0, af, T)
    t = np.arange(0.0, T + 1.0 / hz, 1.0 / hz)
    _, vel, acc = sample_poly5(c, t)
    jerk = np.diff(acc) * hz   # finite difference jerk

    if np.max(np.abs(vel))  > v_max * 1.001: return False
    if np.max(np.abs(acc))  > a_max * 1.001: return False
    if np.max(np.abs(jerk)) > j_max * 1.001: return False
    return True


def find_T_min(q0: float, qf: float,
               v0: float, vf: float,
               a0: float, af: float,
               v_max: float, a_max: float, j_max: float,
               tol: float = 1e-3,
               hz:  int   = 2000) -> float:
    """
    Bisection search for minimum feasible trajectory duration T.

    The feasible set is [T_min, ∞) — a half-line — because all derivatives
    scale down as T increases (monotone property). Bisection therefore
    converges in O(log(1/tol)) iterations ≈ 14 for tol=1ms.

    Returns T_min with a 5% safety margin applied.
    """
    # ── Lower bound: physics floor ────────────────────────────────────────
    T_lo = max(
        1e-3,
        abs(vf - v0) / a_max,           # min time to change velocity
        abs(af - a0) / j_max,           # min time to change acceleration
    )

    # ── Upper bound: start from a rest-to-rest estimate and double ────────
    dq_raw = abs(qf - q0)
    T_rtr  = max(
        (15.0 / 8.0) * dq_raw / v_max,
        np.sqrt((10.0 * np.sqrt(3.0) / 3.0) * dq_raw / a_max),
        np.cbrt(60.0 * dq_raw / j_max),
        T_lo * 2,
        0.1,
    )
    T_hi = T_rtr * 3.0

    # Ensure T_hi is actually feasible (double until it is)
    max_doublings = 20
    for _ in range(max_doublings):
        if check_feasible(q0, qf, v0, vf, a0, af, T_hi,
                          v_max, a_max, j_max, hz):
            break
        T_hi *= 2.0
    else:
        raise RuntimeError(
            f"Cannot find feasible T within {T_hi:.1f} s. "
            "Check boundary velocities vs limits."
        )

    # ── Bisection ─────────────────────────────────────────────────────────
    while (T_hi - T_lo) > tol:
        T_mid = 0.5 * (T_lo + T_hi)
        if check_feasible(q0, qf, v0, vf, a0, af, T_mid,
                          v_max, a_max, j_max, hz):
            T_hi = T_mid
        else:
            T_lo = T_mid

    return T_hi * 1.05   # 5% safety margin


# ─────────────────────────────────────────────────────────────────────────────
#  Verification helper
# ─────────────────────────────────────────────────────────────────────────────

def verify_boundary_conditions(c: np.ndarray,
                                T: float,
                                q0: float, qf: float,
                                v0: float, vf: float,
                                a0: float, af: float,
                                tol: float = 1e-6) -> bool:
    """
    Numerically verify all 6 boundary conditions. Used in unit tests.
    """
    pos0, vel0, acc0 = sample_poly5(c, np.array([0.0]))
    posT, velT, accT = sample_poly5(c, np.array([T]))
    ok = True
    checks = [
        ("q(0)",  pos0[0], q0),
        ("q(T)",  posT[0], qf),
        ("dq(0)", vel0[0], v0),
        ("dq(T)", velT[0], vf),
        ("ddq(0)",acc0[0], a0),
        ("ddq(T)",accT[0], af),
    ]
    for name, got, want in checks:
        if abs(got - want) > tol:
            print(f"  FAIL {name}: got {got:.8f}, want {want:.8f}")
            ok = False
    return ok
