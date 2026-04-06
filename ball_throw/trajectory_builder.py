"""
trajectory_builder.py
======================
Builds the full two-phase ball-throw trajectory:

  Phase 1 (rest → non-rest):
    q_start (zero vel/acc)  →  q_release (dq_release, zero acc)

  Phase 2 (non-rest → rest):
    q_release (dq_release, zero acc)  →  q_stop (zero vel/acc)

For each phase, T is found by bisection to satisfy:
    - joint velocity limits
    - joint acceleration limits
    - joint jerk limits

After both phases are built, RNEA torques are computed and the
trajectory is checked against torque limits. If violated, T is
scaled up and re-planned.

Output CSV columns (29 total):
    t | q0..q6 | dq0..dq6 | ddq0..ddq6 | tau0..tau6
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from poly5_general import (
    poly5_coeffs_general,
    sample_poly5,
    find_T_min,
    verify_boundary_conditions,
)

try:
    import pinocchio as pin
    HAS_PINOCCHIO = True
except ImportError:
    HAS_PINOCCHIO = False

log = logging.getLogger(__name__)

N_JOINTS = 7

# ── Franka Panda hardware limits ──────────────────────────────────────────────
Q_MIN   = np.array([-2.8973, -1.7628, -2.8973, -3.0718,
                    -2.8973, -0.0175, -2.8973])
Q_MAX   = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,
                     2.8973,  3.7525,  2.8973])
V_MAX   = np.array([2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610])
A_MAX   = np.array([15.0,  7.5,  10.0,  12.5,  15.0,  15.0,  15.0])
J_MAX   = np.array([7500., 3750., 5000., 6250., 7500., 7500., 7500.])
TAU_MAX = np.array([87.0,  87.0,  87.0,  87.0,  12.0,  12.0,  12.0])


# ─────────────────────────────────────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ThrowConfig:
    """All user-facing parameters for the throwing motion."""
    q_start:    np.ndarray          # [7] starting config (rest)
    q_release:  np.ndarray          # [7] release config (from IK)
    dq_release: np.ndarray          # [7] release velocities (from IK)
    q_stop:     np.ndarray          # [7] stopping config (rest)

    # Optional: override T for each phase (None = use bisection)
    T1_override: Optional[float] = None
    T2_override: Optional[float] = None

    # Torque check with payload
    urdf_path:   str = "franka_panda/panda.urdf"
    payload_mass: float = 0.0         # kg
    payload_com:  np.ndarray = field(
        default_factory=lambda: np.zeros(3))

    control_hz:  int = 1000
    output_csv:  str = "throw_trajectory.csv"


@dataclass
class PhaseResult:
    """One phase of the trajectory."""
    label:  str
    T:      float
    t_arr:  np.ndarray   # [N]
    Q:      np.ndarray   # [N,7]
    Qd:     np.ndarray   # [N,7]
    Qdd:    np.ndarray   # [N,7]
    tau:    np.ndarray   # [N,7]  (zeros if no Pinocchio)


@dataclass
class ThrowTrajectory:
    """Complete two-phase trajectory."""
    phase1: PhaseResult
    phase2: PhaseResult
    t_arr:  np.ndarray   # [N_total]   concatenated
    Q:      np.ndarray   # [N_total,7]
    Qd:     np.ndarray   # [N_total,7]
    Qdd:    np.ndarray   # [N_total,7]
    tau:    np.ndarray   # [N_total,7]
    release_index: int   # index of release point in concatenated arrays


# ─────────────────────────────────────────────────────────────────────────────
#  Pinocchio helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_model(urdf_path: str, payload_mass: float,
               payload_com: np.ndarray):
    if not HAS_PINOCCHIO:
        return None, None

    urdf = Path(urdf_path)
    if not urdf.exists():
        log.warning("URDF not found at %s — torque checks disabled", urdf)
        return None, None

    model, cmodel, _ = pin.buildModelsFromUrdf(
        str(urdf), str(urdf.parent)
    )

    # Gripper DOF detection
    extra = model.nq - N_JOINTS
    model.arm_extra_dof = extra
    if extra > 0:
        log.info("URDF has %d extra DOF (gripper). Padding with zeros.", extra)

    # Payload
    if payload_mass > 0.0:
        jid = model.getJointId("panda_joint7")
        if jid < model.njoints:
            model.inertias[jid] += pin.Inertia(
                payload_mass, payload_com, np.eye(3) * 1e-5
            )
            log.info("Payload %.3f kg attached to joint 7", payload_mass)

    data = model.createData()
    return model, data


def pad_q(model, q7: np.ndarray) -> np.ndarray:
    extra = getattr(model, "arm_extra_dof", 0)
    return np.concatenate([q7, np.zeros(extra)]) if extra > 0 else q7


def compute_rnea(model, data,
                 Q: np.ndarray, Qd: np.ndarray,
                 Qdd: np.ndarray) -> np.ndarray:
    """
    Run RNEA at every sample. Returns tau [N, 7].
    If no model: returns zeros.
    """
    N = Q.shape[0]
    tau = np.zeros((N, N_JOINTS))
    if model is None:
        log.warning("No Pinocchio model — torques set to zero")
        return tau

    for i in range(N):
        tau_full = pin.rnea(
            model, data,
            pad_q(model, Q[i]),
            pad_q(model, Qd[i]),
            pad_q(model, Qdd[i]),
        )
        tau[i] = tau_full[:N_JOINTS]
    return tau


# ─────────────────────────────────────────────────────────────────────────────
#  Single-phase builder
# ─────────────────────────────────────────────────────────────────────────────

def build_phase(label: str,
                q0:  np.ndarray, qf:  np.ndarray,
                v0:  np.ndarray, vf:  np.ndarray,
                a0:  np.ndarray, af:  np.ndarray,
                model, data,
                hz:      int   = 1000,
                T_override: Optional[float] = None,
                torque_scale: float = 1.0) -> PhaseResult:
    """
    Build one phase of the trajectory.

    Finds T_min per joint via bisection, takes max across joints,
    samples at hz, computes RNEA torques, checks against limits.
    Scales T up if torque limits are violated.
    """
    log.info("=== Building %s ===", label)

    # ── Step 1: Find T_min from kinematic limits ──────────────────────────
    if T_override is not None:
        T = T_override
        log.info("Using T override: %.3f s", T)
    else:
        T_per_joint = []
        for j in range(N_JOINTS):
            T_j = find_T_min(
                q0[j], qf[j],
                v0[j], vf[j],
                a0[j], af[j],
                V_MAX[j], A_MAX[j], J_MAX[j],
            )
            T_per_joint.append(T_j)
            log.debug("  Joint %d: T_min = %.4f s", j+1, T_j)

        T = max(T_per_joint)
        log.info("T_min (kinematic) = %.4f s  (binding joint: %d)",
                 T, int(np.argmax(T_per_joint)) + 1)

    # ── Step 2: Sample at hz ──────────────────────────────────────────────
    for torque_iter in range(30):
        N_samp = int(round(T * hz)) + 1
        t_arr  = np.linspace(0.0, T, N_samp)
        N = len(t_arr)
        Q   = np.zeros((N, N_JOINTS))
        Qd  = np.zeros((N, N_JOINTS))
        Qdd = np.zeros((N, N_JOINTS))

        for j in range(N_JOINTS):
            c = poly5_coeffs_general(
                q0[j], qf[j],
                v0[j], vf[j],
                a0[j], af[j], T
            )
            Q[:, j], Qd[:, j], Qdd[:, j] = sample_poly5(c, t_arr)

        # ── Step 3: RNEA torque check ─────────────────────────────────────
        tau = compute_rnea(model, data, Q, Qd, Qdd)
        peak_tau = np.max(np.abs(tau), axis=0)

        violations = np.where(peak_tau > TAU_MAX * torque_scale)[0]
        if len(violations) == 0:
            log.info("Torque OK (max: %s Nm)",
                     np.round(peak_tau, 2))
            break

        log.warning("Torque violation (iter %d) — peaks: %s Nm "
                    "(limits: %s). Increasing T by 20%%.",
                    torque_iter, np.round(peak_tau, 2), TAU_MAX)
        T *= 1.2
    else:
        raise RuntimeError(
            f"{label}: Cannot satisfy torque limits within 30 iterations. "
            "Check payload parameters."
        )

    log.info("%s: T = %.4f s, %d samples", label, T, N)

    return PhaseResult(
        label=label, T=T, t_arr=t_arr,
        Q=Q, Qd=Qd, Qdd=Qdd, tau=tau,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Full two-phase trajectory
# ─────────────────────────────────────────────────────────────────────────────

def build_throw_trajectory(cfg: ThrowConfig) -> ThrowTrajectory:
    """
    Build the complete ball-throw trajectory.

    Phase 1: rest-to-non-rest
        q_start (v=0, a=0)  →  q_release (dq_release, a=0)

    Phase 2: non-rest-to-rest
        q_release (dq_release, a=0)  →  q_stop (v=0, a=0)

    The acceleration at the release point is set to zero (a=0).
    This is the simplest safe choice — the arm is neither speeding up
    nor slowing down at the moment of release, which keeps the throw
    velocity clean and predictable.
    """
    zeros7 = np.zeros(N_JOINTS)

    model, data = load_model(
        cfg.urdf_path, cfg.payload_mass, cfg.payload_com
    )

    # ── Phase 1: rest → release ───────────────────────────────────────────
    p1 = build_phase(
        label   = "Phase 1 (rest → release)",
        q0      = cfg.q_start,
        qf      = cfg.q_release,
        v0      = zeros7,
        vf      = cfg.dq_release,
        a0      = zeros7,
        af      = zeros7,        # zero acc at release
        model   = model,
        data    = data,
        hz      = cfg.control_hz,
        T_override = cfg.T1_override,
    )

    # ── Phase 2: release → stop ───────────────────────────────────────────
    p2 = build_phase(
        label   = "Phase 2 (release → stop)",
        q0      = cfg.q_release,
        qf      = cfg.q_stop,
        v0      = cfg.dq_release,
        vf      = zeros7,
        a0      = zeros7,        # zero acc at release (matches phase 1 end)
        af      = zeros7,
        model   = model,
        data    = data,
        hz      = cfg.control_hz,
        T_override = cfg.T2_override,
    )

    # ── Concatenate (remove duplicate release sample) ─────────────────────
    release_idx = len(p1.t_arr) - 1

    t_arr = np.concatenate([p1.t_arr, p2.t_arr[1:] + p1.T])
    Q     = np.concatenate([p1.Q,   p2.Q[1:]])
    Qd    = np.concatenate([p1.Qd,  p2.Qd[1:]])
    Qdd   = np.concatenate([p1.Qdd, p2.Qdd[1:]])
    tau   = np.concatenate([p1.tau, p2.tau[1:]])

    log.info("Total trajectory: %.4f s, %d samples (release at index %d, t=%.4f s)",
             t_arr[-1], len(t_arr), release_idx, t_arr[release_idx])

    return ThrowTrajectory(
        phase1=p1, phase2=p2,
        t_arr=t_arr, Q=Q, Qd=Qd, Qdd=Qdd, tau=tau,
        release_index=release_idx,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  CSV export
# ─────────────────────────────────────────────────────────────────────────────

def export_csv(traj: ThrowTrajectory, path: str) -> None:
    """
    Write trajectory CSV.

    Columns (29):
        t | q0..q6 | dq0..dq6 | ddq0..ddq6 | tau0..tau6

    An extra comment line marks the release index so the executor
    can know exactly when the ball leaves the hand.
    """
    data = np.column_stack([
        traj.t_arr,
        traj.Q,
        traj.Qd,
        traj.Qdd,
        traj.tau,
    ])

    header = (
        "t,"
        + ",".join(f"q{j}"   for j in range(N_JOINTS)) + ","
        + ",".join(f"dq{j}"  for j in range(N_JOINTS)) + ","
        + ",".join(f"ddq{j}" for j in range(N_JOINTS)) + ","
        + ",".join(f"tau{j}" for j in range(N_JOINTS))
        + f"\n# release_index={traj.release_index}"
        + f"  release_time={traj.t_arr[traj.release_index]:.6f}"
    )

    np.savetxt(path, data, delimiter=",",
               header=header, comments="")

    log.info("CSV saved: %s  (%d rows × %d cols)",
             path, data.shape[0], data.shape[1])
    log.info("Release at row %d (t = %.4f s)",
             traj.release_index,
             traj.t_arr[traj.release_index])
