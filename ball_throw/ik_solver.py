"""
ik_solver.py
============
Inverse kinematics for Franka Panda using Pinocchio.

  - Position IK : iterative Jacobian pseudo-inverse (CLIK)
                  maps end-effector position → joint angles q
  - Velocity IK : single linear solve
                  maps end-effector velocity → joint velocities dq
                  dq = J†(q) · v_ee   (least-norm solution)

For ball throwing we need:
  1. q_release  — joint config at the moment of release
  2. dq_release — joint velocities at release (to achieve v_ee_release)
"""

import numpy as np
import logging

log = logging.getLogger(__name__)

try:
    import pinocchio as pin
    HAS_PINOCCHIO = True
except ImportError:
    HAS_PINOCCHIO = False
    log.warning("Pinocchio not found — IK will use mock values")


# ─────────────────────────────────────────────────────────────────────────────
#  Franka joint limits (for IK clamping)
# ─────────────────────────────────────────────────────────────────────────────

Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718,
                  -2.8973, -0.0175, -2.8973])
Q_MAX = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,
                   2.8973,  3.7525,  2.8973])
V_MAX = np.array([2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610])


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: get Pinocchio frame IDs
# ─────────────────────────────────────────────────────────────────────────────

def get_ee_frame_id(model) -> int:
    """Return frame ID for panda_hand, falling back to panda_link8."""
    for name in ("panda_hand", "panda_link8"):
        fid = model.getFrameId(name)
        if fid < model.nframes:
            return fid
    raise RuntimeError("Could not find end-effector frame in URDF")


def get_ee_jacobian(model, data, q_full: np.ndarray) -> np.ndarray:
    """
    6×7 end-effector Jacobian for the 7 arm joints.
    Pads q_full to model.nq if gripper joints are present.
    """
    pin.computeJointJacobians(model, data, q_full)
    pin.framesForwardKinematics(model, data, q_full)
    fid = get_ee_frame_id(model)
    J_full = pin.getFrameJacobian(
        model, data, fid,
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )
    return J_full[:, :7]   # 6×7 arm columns only


def pad_q(model, q7: np.ndarray) -> np.ndarray:
    """Pad 7-element arm config to model.nq (adds finger zeros if needed)."""
    extra = model.nq - 7
    if extra <= 0:
        return q7
    return np.concatenate([q7, np.zeros(extra)])


def get_ee_pose(model, data, q7: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (position [3], rotation_matrix [3×3]) of end-effector.
    """
    q_full = pad_q(model, q7)
    pin.forwardKinematics(model, data, q_full)
    pin.framesForwardKinematics(model, data, q_full)
    fid  = get_ee_frame_id(model)
    pose = data.oMf[fid]
    return np.array(pose.translation), np.array(pose.rotation)


# ─────────────────────────────────────────────────────────────────────────────
#  Position IK — iterative CLIK
# ─────────────────────────────────────────────────────────────────────────────

def ik_position(model, data,
                p_target:     np.ndarray,
                q_init:       np.ndarray,
                max_iter:     int   = 200,
                tol_pos:      float = 1e-4,
                damp:         float = 1e-6,
                step_size:    float = 0.5) -> np.ndarray:
    """
    Compute joint angles q such that the end-effector reaches p_target.

    Uses damped least-squares Jacobian pseudo-inverse (CLIK):
        dq = J^T (J J^T + λI)^{-1} · e_pos
        q  ← q + α · dq    (clamped to joint limits)

    Only position (3-DOF) is controlled. Orientation is free.
    For ball throwing this is appropriate — the ball leaves from a
    specific point in space regardless of wrist orientation.

    Parameters
    ----------
    model, data   : Pinocchio model/data
    p_target      : [3] desired end-effector position in world frame
    q_init        : [7] initial joint configuration for the iteration
    max_iter      : maximum CLIK iterations
    tol_pos       : convergence tolerance [m]
    damp          : damping factor λ for numerical stability
    step_size     : α — fraction of full step to take each iteration

    Returns
    -------
    q_sol : [7] joint configuration achieving p_target
    """
    if not HAS_PINOCCHIO:
        log.warning("IK mock: returning q_init")
        return q_init.copy()

    q = q_init.copy()

    for i in range(max_iter):
        # Current end-effector position
        p_cur, _ = get_ee_pose(model, data, q)
        e = p_target - p_cur           # position error [3]

        if np.linalg.norm(e) < tol_pos:
            log.info("IK converged in %d iterations (err=%.6f m)", i, np.linalg.norm(e))
            return q

        # 6×7 Jacobian → take only position rows (top 3)
        J_full = get_ee_jacobian(model, data, pad_q(model, q))
        Jp = J_full[:3, :]             # 3×7 position Jacobian

        # Damped pseudo-inverse: Jp^T (Jp Jp^T + λI)^{-1}
        JpJpT = Jp @ Jp.T              # 3×3
        JpJpT += damp * np.eye(3)
        dq = Jp.T @ np.linalg.solve(JpJpT, e)   # 7-element step

        # Update and clamp to joint limits
        q = np.clip(q + step_size * dq, Q_MIN, Q_MAX)

    # Did not converge — return best solution with warning
    p_final, _ = get_ee_pose(model, data, q)
    final_err = np.linalg.norm(p_target - p_final)
    log.warning("IK did not fully converge after %d iterations "
                "(residual=%.4f m). Using best solution.", max_iter, final_err)
    return q


# ─────────────────────────────────────────────────────────────────────────────
#  Velocity IK — single pseudo-inverse solve
# ─────────────────────────────────────────────────────────────────────────────

def ik_velocity(model, data,
                q:          np.ndarray,
                v_ee:       np.ndarray,
                damp:       float = 1e-6) -> np.ndarray:
    """
    Compute joint velocities dq to achieve end-effector velocity v_ee.

    Uses the least-norm solution:
        dq = J†(q) · v_ee
           = J^T (J J^T + λI)^{-1} · v_ee

    v_ee is the 3D linear velocity of the end-effector [m/s].
    Only linear velocity is controlled (3-DOF, consistent with position IK).

    Clamps the result to joint velocity limits.

    Parameters
    ----------
    model, data : Pinocchio model/data
    q           : [7] current joint configuration
    v_ee        : [3] desired end-effector linear velocity [m/s]
    damp        : damping factor

    Returns
    -------
    dq : [7] joint velocities [rad/s]
    """
    if not HAS_PINOCCHIO:
        log.warning("Velocity IK mock: returning zeros")
        return np.zeros(7)

    J_full = get_ee_jacobian(model, data, pad_q(model, q))
    Jp     = J_full[:3, :]   # 3×7 position Jacobian

    # Damped pseudo-inverse
    JpJpT  = Jp @ Jp.T + damp * np.eye(3)
    dq     = Jp.T @ np.linalg.solve(JpJpT, v_ee)

    # Check and warn if velocity limits are exceeded
    ratio = np.abs(dq) / V_MAX
    if np.any(ratio > 1.0):
        worst_j = int(np.argmax(ratio))
        log.warning(
            "Velocity IK: joint %d at %.1f%% of limit (%.3f / %.3f rad/s). "
            "Consider reducing end-effector speed.",
            worst_j + 1, ratio[worst_j] * 100,
            dq[worst_j], V_MAX[worst_j]
        )

    return dq


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience: full IK for a release state
# ─────────────────────────────────────────────────────────────────────────────

def solve_release_state(model, data,
                        p_release:  np.ndarray,
                        v_ee_release: np.ndarray,
                        q_init:     np.ndarray,
                        ik_tol:     float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    """
    Given a desired end-effector release point and velocity,
    return (q_release, dq_release) in joint space.

    Parameters
    ----------
    p_release      : [3] end-effector position at ball release [m]
    v_ee_release   : [3] end-effector linear velocity at release [m/s]
    q_init         : [7] initial guess for IK
    ik_tol         : position IK convergence tolerance [m]

    Returns
    -------
    q_release  : [7] joint angles at release
    dq_release : [7] joint velocities at release
    """
    log.info("Solving position IK for release point %s", np.round(p_release, 4))
    q_release = ik_position(model, data, p_release, q_init, tol_pos=ik_tol)

    # Verify IK quality
    p_achieved, _ = get_ee_pose(model, data, q_release)
    pos_err = np.linalg.norm(p_release - p_achieved)
    log.info("IK position error: %.4f m", pos_err)

    log.info("Solving velocity IK for v_ee = %s m/s",
             np.round(v_ee_release, 4))
    dq_release = ik_velocity(model, data, q_release, v_ee_release)

    # Verify velocity
    J   = get_ee_jacobian(model, data, pad_q(model, q_release))
    v_achieved = J[:3, :] @ dq_release
    vel_err = np.linalg.norm(v_ee_release - v_achieved)
    log.info("Velocity IK error: %.4f m/s", vel_err)
    log.info("q_release  = %s rad", np.round(q_release,  4))
    log.info("dq_release = %s rad/s", np.round(dq_release, 4))

    # ── Feasibility check — fail fast before planning hangs ───────────────
    if not check_velocity_feasibility(dq_release, "dq_release"):
        raise ValueError(
            "Velocity IK produced joint velocities that exceed hardware limits. "
            "Reduce V_EE_RELEASE or change P_RELEASE. "
            "See the diagnosis printed above."
        )

    return q_release, dq_release


# ─────────────────────────────────────────────────────────────────────────────
#  Model loader convenience function (called from main_throw.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_model_for_ik(urdf_path: str):
    """
    Load the Pinocchio model for IK.
    Returns (model, data).
    Raises ImportError if Pinocchio is not installed.
    Raises FileNotFoundError if the URDF does not exist.
    """
    if not HAS_PINOCCHIO:
        raise ImportError(
            "Pinocchio is required for full IK. "
            "Install with:  pip install pin --break-system-packages\n"
            "Or run in mock mode:  python main_throw.py --mock"
        )

    from pathlib import Path
    urdf = Path(urdf_path)
    if not urdf.exists():
        raise FileNotFoundError(
            f"URDF not found: {urdf}\n"
            "Make sure pybullet is installed and the URDF path is correct.\n"
            "Quick fix:\n"
            "  pip install pybullet --break-system-packages\n"
            "  python -c \"import pybullet_data,os; "
            "print(os.path.join(pybullet_data.getDataPath(),'franka_panda/panda.urdf'))\""
        )

    model, cmodel, vmodel = pin.buildModelsFromUrdf(
        str(urdf), str(urdf.parent)
    )

    # Detect and store extra DOF (gripper fingers in PyBullet URDF)
    extra = model.nq - 7
    model.arm_extra_dof = extra
    if extra > 0:
        log.info("URDF has %d extra DOF (gripper). Will pad with zeros.", extra)

    data = model.createData()
    log.info("Model loaded: %s  (nq=%d, njoints=%d)",
             urdf.name, model.nq, model.njoints)
    return model, data


# ─────────────────────────────────────────────────────────────────────────────
#  Velocity feasibility check
# ─────────────────────────────────────────────────────────────────────────────

def check_velocity_feasibility(dq: np.ndarray,
                                label: str = "dq_release") -> bool:
    """
    Check whether a joint velocity vector is within hardware limits.
    Prints a clear diagnosis if not.
    Returns True if feasible, False otherwise.
    """
    ratio    = np.abs(dq) / V_MAX
    violated = np.where(ratio > 1.0)[0]

    if len(violated) == 0:
        log.info("%s feasibility: OK  (max %.1f%% of limit)",
                 label, float(np.max(ratio)) * 100)
        return True

    print()
    print("=" * 60)
    print("  VELOCITY IK INFEASIBLE — cannot plan trajectory")
    print("=" * 60)
    print(f"  The requested end-effector velocity requires joint")
    print(f"  velocities that exceed hardware limits:\n")
    for j in violated:
        print(f"    Joint {j+1}: {dq[j]:+.3f} rad/s  "
              f"(limit ±{V_MAX[j]:.3f},  {ratio[j]*100:.0f}% of limit)")
    print()
    print("  Fix options (choose one):")
    print("  1. Reduce V_EE_RELEASE magnitude in main_throw.py")
    print(f"     Current: {np.linalg.norm(dq):.3f} m/s total")
    print(f"     Try:     scale down by factor {float(np.max(ratio)):.1f}×")
    print()
    print("  2. Change P_RELEASE to a configuration where the arm")
    print("     has better manipulability for this throw direction.")
    print()
    print("  3. Change throw direction in V_EE_RELEASE to one the")
    print("     arm can achieve at this configuration.")
    print("=" * 60)
    print()
    return False