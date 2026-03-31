"""
planner/trajectory_planner.py
==============================
Offline trajectory planner for Franka Emika Panda.

Steps performed here (all offline, no robot connection):
  2. Compute polynomial coefficients
  3. Sample trajectory at 1 kHz
  4. Check torques with payload via RNEA (Pinocchio) — increase T if violated
  5. Check manipulability / sigma_min — perturb via-points if near-singular
  6. Check self-collision (Pinocchio)
  7. Export trajectory.csv for the C++ executor

Usage:
    python main_plan.py                     # uses defaults in main_plan.py
    python main_plan.py --mock              # skip Pinocchio (no URDF needed)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import pinocchio as pin
    HAS_PINOCCHIO = True
except ImportError:
    HAS_PINOCCHIO = False
    print("[WARN] Pinocchio not found — dynamics/collision checks disabled")

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Franka Panda hardware constants
# ─────────────────────────────────────────────────────────────────────────────

N_JOINTS = 7

Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_MAX = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])

V_MAX   = np.array([2.175,  2.175,  2.175,  2.175,  2.610,  2.610,  2.610])
A_MAX   = np.array([15.0,   7.5,   10.0,   12.5,   15.0,   15.0,   15.0])
J_MAX   = np.array([7500., 3750., 5000., 6250., 7500., 7500., 7500.])
TAU_MAX = np.array([87.0,  87.0,  87.0,  87.0,  12.0,  12.0,  12.0])

# Singularity thresholds (offline)
SIGMA_MIN_THRESHOLD = 0.04   # [m/rad]

# T search
T_SAFETY_MARGIN      = 1.10
T_INCREMENT          = 0.2   # [s]
T_MAX                = 60.0  # [s]

# Via-point perturbation
VIA_PERTURB_MAG      = 0.05  # [rad]
VIA_PERTURB_MAX_ITER = 8


# ─────────────────────────────────────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PayloadParams:
    mass:    float       = 0.0
    com:     np.ndarray  = field(default_factory=lambda: np.zeros(3))
    inertia: np.ndarray  = field(default_factory=lambda: np.eye(3) * 1e-5)


@dataclass
class PlannerConfig:
    q_start:    np.ndarray
    q_end:      np.ndarray
    payload:    PayloadParams = field(default_factory=PayloadParams)
    urdf_path:  str           = "franka_panda/panda.urdf"
    control_hz: int           = 1000
    output_csv: str           = "trajectory.csv"


@dataclass
class TrajectoryResult:
    T:          float
    t_arr:      np.ndarray   # [N]
    Q:          np.ndarray   # [N, 7]  positions
    Qd:         np.ndarray   # [N, 7]  velocities
    Qdd:        np.ndarray   # [N, 7]  accelerations
    tau:        np.ndarray   # [N, 7]  RNEA torques
    w:          np.ndarray   # [N]     manipulability
    sigma_min:  np.ndarray   # [N]     min singular value of Jacobian


# ─────────────────────────────────────────────────────────────────────────────
#  Polynomial mathematics
# ─────────────────────────────────────────────────────────────────────────────

def compute_T_min(dq: np.ndarray) -> float:
    """
    Minimum duration satisfying all kinematic limits across all 7 joints.

    Analytic peak values for rest-to-rest 5th-order polynomial:
        |q̇|_peak  = 15|Δq| / (8T)
        |q̈|_peak  = 10√3·|Δq| / (3T²)
        |q⃛|_peak  = 60|Δq| / T³
    """
    abs_dq = np.abs(dq)
    T_vel  = (15.0 / 8.0) * abs_dq / V_MAX
    T_acc  = np.sqrt((10.0 * np.sqrt(3.0) / 3.0) * abs_dq / A_MAX)
    T_jerk = np.cbrt(60.0 * abs_dq / J_MAX)
    T_min  = float(np.max([T_vel, T_acc, T_jerk])) * T_SAFETY_MARGIN
    return max(T_min, 0.1)


def poly5_coeffs(q0: float, qf: float, T: float) -> np.ndarray:
    """Closed-form coefficients [a0..a5] for rest-to-rest single joint."""
    dq = qf - q0
    return np.array([
        q0,
        0.0,
        0.0,
        10.0 * dq / T**3,
        -15.0 * dq / T**4,
          6.0 * dq / T**5,
    ])


def sample_poly5(c: np.ndarray, t: np.ndarray):
    """Evaluate (pos, vel, acc) for one joint given coefficient vector c."""
    a0, a1, a2, a3, a4, a5 = c
    pos = a0 + a1*t   + a2*t**2  + a3*t**3   + a4*t**4   + a5*t**5
    vel =      a1     + 2*a2*t   + 3*a3*t**2  + 4*a4*t**3  + 5*a5*t**4
    acc =               2*a2     + 6*a3*t     + 12*a4*t**2  + 20*a5*t**3
    return pos, vel, acc


def build_trajectory(q0: np.ndarray, qf: np.ndarray,
                     T: float, hz: int = 1000):
    """Sample all 7 joints. Returns (t_arr, Q, Qd, Qdd), each shape [N, 7]."""
    t_arr = np.arange(0.0, T + 1.0/hz, 1.0/hz)
    N = len(t_arr)
    Q = np.zeros((N, N_JOINTS))
    Qd  = np.zeros((N, N_JOINTS))
    Qdd = np.zeros((N, N_JOINTS))
    for j in range(N_JOINTS):
        c = poly5_coeffs(q0[j], qf[j], T)
        Q[:, j], Qd[:, j], Qdd[:, j] = sample_poly5(c, t_arr)
    return t_arr, Q, Qd, Qdd


def build_via_trajectory(q0: np.ndarray, via: np.ndarray,
                          qf: np.ndarray, T: float, hz: int = 1000):
    """
    Two rest-to-rest segments: q0→via (T/2) and via→qf (T/2).
    Zero velocity enforced at via-point — conservative but safe.
    """
    T2 = T / 2.0
    t1, Q1, Qd1, Qdd1 = build_trajectory(q0,  via, T2, hz)
    t2, Q2, Qd2, Qdd2 = build_trajectory(via, qf,  T2, hz)
    t_arr = np.concatenate([t1, t2[1:] + T2])
    Q     = np.concatenate([Q1,   Q2[1:]])
    Qd    = np.concatenate([Qd1,  Qd2[1:]])
    Qdd   = np.concatenate([Qdd1, Qdd2[1:]])
    return t_arr, Q, Qd, Qdd


# ─────────────────────────────────────────────────────────────────────────────
#  Pinocchio model
# ─────────────────────────────────────────────────────────────────────────────

def load_panda_model(urdf_path: str, payload: PayloadParams):
    """
    Load URDF and merge payload inertia into joint 7.

    Some URDF sources (e.g. PyBullet's panda.urdf) include the gripper
    fingers, giving model.nq = 9 instead of 7.  We detect this and store
    the extra DOF count so every Pinocchio call can pad the 7-element arm
    arrays to the full model size.  The finger joints are held fixed at 0.
    """
    if not HAS_PINOCCHIO:
        return (None,) * 5

    urdf = Path(urdf_path)
    if not urdf.exists():
        log.warning("URDF not found at %s", urdf)
        return (None,) * 5

    model, cmodel, vmodel = pin.buildModelsFromUrdf(
        str(urdf), str(urdf.parent)
    )

    # Detect extra DOF (e.g. gripper fingers).
    # model.nq is the full configuration-space dimension.
    extra_dof = model.nq - N_JOINTS
    if extra_dof > 0:
        log.info(
            "URDF has %d extra DOF beyond the 7 arm joints (e.g. gripper). "
            "Finger joints will be held fixed at zero.", extra_dof
        )
    # Store on the model object so helpers can read it without a global.
    model.arm_extra_dof = extra_dof

    if payload.mass > 0.0:
        jid = model.getJointId("panda_joint7")
        if jid < model.njoints:
            model.inertias[jid] += pin.Inertia(
                payload.mass, payload.com, payload.inertia
            )
            log.info("Payload %.3f kg attached to joint 7", payload.mass)

    data      = model.createData()
    geom_data = pin.GeometryData(cmodel) if cmodel is not None else None
    return model, cmodel, vmodel, data, geom_data


def _pad(model, q7: np.ndarray) -> np.ndarray:
    """
    Pad a 7-element arm configuration to model.nq by appending zeros
    for any extra DOF (gripper fingers etc.).
    """
    extra = getattr(model, "arm_extra_dof", 0)
    if extra == 0:
        return q7
    return np.concatenate([q7, np.zeros(extra)])


def _pad_vel(model, v7: np.ndarray) -> np.ndarray:
    """
    Pad a 7-element arm velocity/acceleration to model.nv.
    For the Panda (all revolute joints) nv == nq.
    """
    extra = getattr(model, "arm_extra_dof", 0)
    if extra == 0:
        return v7
    return np.concatenate([v7, np.zeros(extra)])


# ─────────────────────────────────────────────────────────────────────────────
#  Step 4 — Torque check (RNEA)
# ─────────────────────────────────────────────────────────────────────────────

def check_torques(model, data, Q, Qd, Qdd):
    """
    Returns (ok, tau_traj [N,7], violation_indices).
    tau_traj contains the full RNEA output including gravity,
    sliced to the 7 arm joints only.
    """
    if model is None:
        log.warning("No Pinocchio model — torque check skipped")
        return True, np.zeros_like(Q), []

    N = Q.shape[0]
    tau_traj   = np.zeros((N, N_JOINTS))
    violations = []

    for i in range(N):
        tau_full = pin.rnea(
            model, data,
            _pad(model, Q[i]),
            _pad_vel(model, Qd[i]),
            _pad_vel(model, Qdd[i]),
        )
        tau = tau_full[:N_JOINTS]   # drop finger torques
        tau_traj[i] = tau
        if np.any(np.abs(tau) > TAU_MAX):
            violations.append(i)

    ok = len(violations) == 0
    if not ok:
        peak = np.max(np.abs(tau_traj[violations]), axis=0)
        log.warning("Torque violation — peaks: %s Nm", np.round(peak, 2))
    else:
        log.info("Torque OK (max: %s Nm)",
                 np.round(np.max(np.abs(tau_traj), axis=0), 2))
    return ok, tau_traj, violations


# ─────────────────────────────────────────────────────────────────────────────
#  Step 5 — Singularity check and via-point perturbation
# ─────────────────────────────────────────────────────────────────────────────

def _ee_jacobian(model, data, q7: np.ndarray) -> np.ndarray:
    """
    Return the 6×7 end-effector Jacobian for the 7 arm joints.
    Pads q to model.nq if the URDF includes gripper joints, then
    slices the resulting 6×nq Jacobian back to the first 7 columns.
    """
    q_full = _pad(model, q7)
    pin.computeJointJacobians(model, data, q_full)
    pin.framesForwardKinematics(model, data, q_full)

    # Try panda_hand first; fall back to panda_link8 if not present
    for frame_name in ("panda_hand", "panda_link8"):
        fid = model.getFrameId(frame_name)
        if fid < model.nframes:
            break

    J_full = pin.getFrameJacobian(
        model, data, fid,
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )
    return J_full[:, :N_JOINTS]   # 6×7 — arm columns only


def check_singularity(model, data, Q):
    """
    Returns (ok, w_arr [N], smin_arr [N], singular_indices).
    """
    if model is None:
        log.warning("No Pinocchio model — singularity check skipped")
        N = Q.shape[0]
        return True, np.ones(N), np.ones(N), []

    N = Q.shape[0]
    w_arr    = np.zeros(N)
    smin_arr = np.zeros(N)
    singular = []

    for i in range(N):
        J   = _ee_jacobian(model, data, Q[i])
        svs = np.linalg.svd(J, compute_uv=False)
        smin_arr[i] = svs[-1]
        w_arr[i]    = float(np.sqrt(max(0.0, np.linalg.det(J @ J.T))))
        if smin_arr[i] < SIGMA_MIN_THRESHOLD:
            singular.append(i)

    ok = len(singular) == 0
    if not ok:
        log.warning("Singularity risk at %d samples (min sigma=%.4f)",
                    len(singular), float(np.min(smin_arr)))
    else:
        log.info("Singularity OK (min sigma=%.4f)", float(np.min(smin_arr)))
    return ok, w_arr, smin_arr, singular


def perturb_via_point(model, data,
                      q0: np.ndarray, qf: np.ndarray,
                      singular_indices: list,
                      Q: np.ndarray,
                      iteration: int) -> Optional[np.ndarray]:
    """
    Find a via-point near the worst singular sample by nudging along
    the Jacobian nullspace direction that best increases manipulability.
    """
    if model is None:
        return None

    mag = VIA_PERTURB_MAG * (1.5 ** iteration)

    # Find sample with smallest sigma_min
    worst_i = singular_indices[
        int(np.argmin([
            np.linalg.svd(_ee_jacobian(model, data, Q[k]),
                          compute_uv=False)[-1]
            for k in singular_indices
        ]))
    ]
    q_sing = Q[worst_i].copy()
    J      = _ee_jacobian(model, data, q_sing)
    _, s, Vt = np.linalg.svd(J)

    # Use right singular vectors corresponding to smallest singular values
    # as the nullspace basis to perturb along
    null_vecs = Vt[s < SIGMA_MIN_THRESHOLD * 2] if np.any(
        s < SIGMA_MIN_THRESHOLD * 2) else Vt[-1:]

    best_w, best_via = -1.0, q_sing.copy()
    for v in null_vecs:
        for sign in (+1.0, -1.0):
            q_cand = np.clip(q_sing + sign * mag * v, Q_MIN, Q_MAX)
            J_c    = _ee_jacobian(model, data, q_cand)
            w_c    = float(np.sqrt(max(0.0, np.linalg.det(J_c @ J_c.T))))
            if w_c > best_w:
                best_w, best_via = w_c, q_cand

    new_sigma = np.linalg.svd(
        _ee_jacobian(model, data, best_via), compute_uv=False
    )[-1]
    log.info("Via-point iter %d: new sigma_min=%.4f (perturb mag=%.4f rad)",
             iteration, new_sigma, mag)
    return best_via


# ─────────────────────────────────────────────────────────────────────────────
#  Step 6 — Self-collision check
# ─────────────────────────────────────────────────────────────────────────────

def check_self_collision(model, cmodel, geom_data, Q):
    """Returns (ok, collision_indices)."""
    if model is None or cmodel is None or geom_data is None:
        log.warning("Collision model unavailable — check skipped")
        return True, []

    data       = model.createData()
    collisions = []
    for i in range(Q.shape[0]):
        q_full = _pad(model, Q[i])
        pin.updateGeometryPlacements(model, data, cmodel, geom_data, q_full)
        pin.computeCollisions(cmodel, geom_data, stop_at_first_collision=True)
        if any(geom_data.collisionResults[k].isCollision()
               for k in range(len(cmodel.collisionPairs))):
            collisions.append(i)

    ok = len(collisions) == 0
    if not ok:
        log.error("Self-collision at %d samples!", len(collisions))
    else:
        log.info("Self-collision check passed")
    return ok, collisions


# ─────────────────────────────────────────────────────────────────────────────
#  Step 7 — CSV export for C++ executor
# ─────────────────────────────────────────────────────────────────────────────

def export_trajectory(result: TrajectoryResult, path: str) -> None:
    """
    Write trajectory.csv.

    Columns (29 total):
        t | q0..q6 | dq0..dq6 | ddq0..ddq6 | tau0..tau6

    The C++ executor reads this file column by column.
    """
    data = np.column_stack([
        result.t_arr,
        result.Q,
        result.Qd,
        result.Qdd,
        result.tau,
    ])
    header = (
        "t,"
        + ",".join(f"q{j}"   for j in range(N_JOINTS)) + ","
        + ",".join(f"dq{j}"  for j in range(N_JOINTS)) + ","
        + ",".join(f"ddq{j}" for j in range(N_JOINTS)) + ","
        + ",".join(f"tau{j}" for j in range(N_JOINTS))
    )
    np.savetxt(path, data, delimiter=",", header=header, comments="")
    log.info("Trajectory saved to %s  (%d rows × %d cols)",
             path, data.shape[0], data.shape[1])


# ─────────────────────────────────────────────────────────────────────────────
#  Main planning pipeline
# ─────────────────────────────────────────────────────────────────────────────

def plan(cfg: PlannerConfig) -> TrajectoryResult:
    """
    Run the full offline planning pipeline (steps 2–7).
    Raises RuntimeError if a safe trajectory cannot be found.
    """
    q0 = np.asarray(cfg.q_start, dtype=float)
    qf = np.asarray(cfg.q_end,   dtype=float)
    _validate(q0, "q_start")
    _validate(qf, "q_end")

    model, cmodel, vmodel, data, geom_data = load_panda_model(
        cfg.urdf_path, cfg.payload
    )

    # ── Step 2: initial T from kinematic limits ───────────────────────────
    T = compute_T_min(qf - q0)
    log.info("T_min (kinematic) = %.3f s", T)

    via: Optional[np.ndarray] = None
    w_arr    = np.ones(1)
    smin_arr = np.ones(1)

    for outer in range(200):
        log.info("Iteration %d — T = %.3f s", outer, T)

        # ── Step 3: sample ────────────────────────────────────────────────
        if via is None:
            t_arr, Q, Qd, Qdd = build_trajectory(q0, qf, T, cfg.control_hz)
        else:
            t_arr, Q, Qd, Qdd = build_via_trajectory(q0, via, qf, T,
                                                       cfg.control_hz)

        # ── Step 4: torque check ──────────────────────────────────────────
        tok, tau_traj, _ = check_torques(model, data, Q, Qd, Qdd)
        if not tok:
            T = min(T + T_INCREMENT, T_MAX)
            if T >= T_MAX:
                raise RuntimeError(
                    f"No dynamically feasible T found within {T_MAX} s. "
                    "Check payload parameters or reconsider the path."
                )
            continue   # retry with larger T

        # ── Step 5: singularity check (inner loop) ────────────────────────
        resolved = False
        for via_iter in range(VIA_PERTURB_MAX_ITER + 1):

            if via is None:
                t_arr, Q, Qd, Qdd = build_trajectory(q0, qf, T, cfg.control_hz)
            else:
                t_arr, Q, Qd, Qdd = build_via_trajectory(q0, via, qf, T,
                                                           cfg.control_hz)

            sok, w_arr, smin_arr, sing_idx = check_singularity(model, data, Q)

            if sok:
                resolved = True
                break

            if via_iter == VIA_PERTURB_MAX_ITER:
                raise RuntimeError(
                    "Singularity could not be resolved after "
                    f"{VIA_PERTURB_MAX_ITER} perturbations. "
                    "Change start/end configuration."
                )

            via = perturb_via_point(model, data, q0, qf,
                                    sing_idx, Q, via_iter)
            if via is None:
                raise RuntimeError("Via-point perturbation failed.")

            # After adding via-point, re-check torques too
            t2, Q2, Qd2, Qdd2 = build_via_trajectory(q0, via, qf, T,
                                                       cfg.control_hz)
            tok2, _, _ = check_torques(model, data, Q2, Qd2, Qdd2)
            if not tok2:
                T = min(T + T_INCREMENT, T_MAX)
                break   # back to outer loop with larger T

        if not resolved:
            continue

        # ── Step 6: self-collision ────────────────────────────────────────
        cok, _ = check_self_collision(model, cmodel, geom_data, Q)
        if not cok:
            raise RuntimeError(
                "Self-collision detected. "
                "Modify configurations or add manual via-points."
            )

        # ── All checks passed ─────────────────────────────────────────────
        log.info("Planning SUCCESS — T=%.3f s, %d samples", T, len(t_arr))
        result = TrajectoryResult(
            T=T, t_arr=t_arr,
            Q=Q, Qd=Qd, Qdd=Qdd,
            tau=tau_traj,
            w=w_arr, sigma_min=smin_arr,
        )
        export_trajectory(result, cfg.output_csv)
        return result

    raise RuntimeError("Planning failed — maximum iterations reached.")


def _validate(q: np.ndarray, name: str) -> None:
    if len(q) != N_JOINTS:
        raise ValueError(f"{name} must have {N_JOINTS} elements, got {len(q)}")
    for j in range(N_JOINTS):
        if not (Q_MIN[j] <= q[j] <= Q_MAX[j]):
            raise ValueError(
                f"{name}[{j}]={q[j]:.4f} outside "
                f"[{Q_MIN[j]:.4f}, {Q_MAX[j]:.4f}]"
            )
