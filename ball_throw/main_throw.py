"""
main_throw.py
=============
Entry point for ball-throw trajectory generation.

Edit the USER CONFIGURATION section below, then run:

    python main_throw.py                 # full run with Pinocchio
    python main_throw.py --mock          # skip IK + Pinocchio (test only)
    python main_throw.py --verify        # run verification checks

Output: throw_trajectory.csv  (load in C++ executor or plot_trajectory.py)
"""

import argparse
import logging
import sys
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
#  USER CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

# Starting configuration — robot at rest here before the throw
Q_START = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

# Stopping configuration — robot comes to rest here after the throw
# Can be same as Q_START or a safe parking pose
Q_STOP  = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

# ── Release state (task space) ────────────────────────────────────────────────
# End-effector position at the moment of ball release [m, world frame]
P_RELEASE = np.array([0.3, 0.0, 0.6])

# End-effector LINEAR velocity at release [m/s, world frame]
# Direction and magnitude determine where the ball goes.
#
# IMPORTANT: keep magnitude below ~1.5 m/s for most configurations.
# The Panda's joint velocity limits (2.175 rad/s for joints 1-4)
# constrain achievable end-effector speed. Too fast causes the
# velocity IK to exceed joint limits → planning hangs.
#
# Rule of thumb: start at 0.5 m/s, increase until you hit warnings.
# Run  python3 verify_release.py --offline  to check IK accuracy.
V_EE_RELEASE = np.array([0.348, -0.3, 0.824])  # 0.725 m/s — 90% of J4 limit

# ── IK initial guess ──────────────────────────────────────────────────────────
# Joint config close to the release pose — helps IK converge
Q_RELEASE_INIT = np.array([0.3, 0.1, 0.0, -1.8, 0.0, 2.0, 0.8])

# ── Payload (ball + gripper) ──────────────────────────────────────────────────
PAYLOAD_MASS = 0.15          # kg (ball ~150g)
PAYLOAD_COM  = np.array([0.0, 0.0, 0.08])   # m, in flange frame

# ── File paths ────────────────────────────────────────────────────────────────
import pybullet_data, os
URDF_PATH  = os.path.join(pybullet_data.getDataPath(),
                           "franka_panda/panda.urdf")
OUTPUT_CSV = "throw_trajectory.csv"

# ═════════════════════════════════════════════════════════════════════════════


def run_full(mock: bool = False) -> None:
    from trajectory_builder import ThrowConfig, build_throw_trajectory, export_csv

    if mock:
        # ── Mock mode: skip IK, use hardcoded release state ───────────────
        log.info("Running in MOCK mode (no Pinocchio / IK)")
        q_release  = Q_RELEASE_INIT.copy()
        dq_release = np.array([0.5, -0.3, 0.2, -0.4, 0.1, 0.6, 0.1])

        cfg = ThrowConfig(
            q_start    = Q_START,
            q_release  = q_release,
            dq_release = dq_release,
            q_stop     = Q_STOP,
            urdf_path  = "MOCK",
            payload_mass = PAYLOAD_MASS,
            payload_com  = PAYLOAD_COM,
            control_hz = 1000,
            output_csv = OUTPUT_CSV,
        )
    else:
        # ── Full mode: IK + Pinocchio torque checks ───────────────────────
        try:
            import pinocchio as pin
        except ImportError:
            log.error("Pinocchio not found. Run with --mock for testing.")
            sys.exit(1)

        from ik_solver import solve_release_state, load_model_for_ik

        log.info("Loading robot model...")
        model, data = load_model_for_ik(URDF_PATH)

        log.info("Solving IK for release state...")
        q_release, dq_release = solve_release_state(
            model, data,
            p_release      = P_RELEASE,
            v_ee_release   = V_EE_RELEASE,
            q_init         = Q_RELEASE_INIT,
        )

        cfg = ThrowConfig(
            q_start      = Q_START,
            q_release    = q_release,
            dq_release   = dq_release,
            q_stop       = Q_STOP,
            urdf_path    = URDF_PATH,
            payload_mass = PAYLOAD_MASS,
            payload_com  = PAYLOAD_COM,
            control_hz   = 1000,
            output_csv   = OUTPUT_CSV,
        )

    # ── Plan trajectory ───────────────────────────────────────────────────
    log.info("Planning trajectory...")
    traj = build_throw_trajectory(cfg)

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  Throw trajectory summary")
    print("="*55)
    print(f"  Phase 1 (rest → release) : {traj.phase1.T:.3f} s")
    print(f"  Phase 2 (release → stop) : {traj.phase2.T:.3f} s")
    print(f"  Total duration           : {traj.t_arr[-1]:.3f} s")
    print(f"  Total samples (1 kHz)    : {len(traj.t_arr)}")
    print(f"  Release at index         : {traj.release_index}")
    print(f"  Release time             : {traj.t_arr[traj.release_index]:.4f} s")
    print(f"  Max |tau| phase 1        : {np.max(np.abs(traj.phase1.tau), axis=0).round(2)} Nm")
    print(f"  Max |tau| phase 2        : {np.max(np.abs(traj.phase2.tau), axis=0).round(2)} Nm")
    print(f"  Release q                : {traj.Q[traj.release_index].round(4)} rad")
    print(f"  Release dq               : {traj.Qd[traj.release_index].round(4)} rad/s")
    print("="*55)

    # ── Export CSV ────────────────────────────────────────────────────────
    export_csv(traj, OUTPUT_CSV)

    # ── Save metadata for plot_throw_trajectory.py ────────────────────────
    meta_path = OUTPUT_CSV.replace(".csv", "_meta.npy")
    np.save(meta_path, {
        "Q_START":        cfg.q_start,
        "Q_RELEASE":      cfg.q_release,
        "Q_STOP":         cfg.q_stop,
        "DQ_RELEASE":     cfg.dq_release,
        "release_index":  traj.release_index,
        "T1":             traj.phase1.T,
        "T2":             traj.phase2.T,
        # Task-space goals — used by verify_release.py
        "P_RELEASE":      np.array(P_RELEASE)    if not mock else np.full(3, np.nan),
        "V_EE_RELEASE":   np.array(V_EE_RELEASE) if not mock else np.full(3, np.nan),
    })
    log.info("Metadata saved: %s", meta_path)

    print(f"\n  CSV saved : {OUTPUT_CSV}")
    print(f"  Meta saved: {meta_path}")
    print("  Run the C++ executor:")
    print(f"    ./panda_executor {OUTPUT_CSV} 192.168.1.10")
    print()
    print("  Plot the trajectory:")
    print(f"    python plot_throw_trajectory.py --csv {OUTPUT_CSV} --meta {meta_path}\n")


def run_verify() -> None:
    """Quick verification that polynomial math is correct."""
    from poly5_general import (
        poly5_coeffs_general, sample_poly5,
        verify_boundary_conditions, find_T_min,
    )

    print("Verifying boundary conditions...")
    test_cases = [
        # (q0, qf, v0, vf, a0, af, T, label)
        (0.0,  1.0, 0.0,  0.0,  0.0,  0.0,  2.0, "rest-to-rest"),
        (0.0,  1.0, 0.0,  0.5,  0.0,  0.0,  2.0, "rest-to-non-rest"),
        (0.0,  1.0, 0.5,  0.0,  0.0,  0.0,  2.0, "non-rest-to-rest"),
        (0.0,  1.0, 0.3,  0.4,  0.1, -0.1,  3.0, "general"),
        (-0.5, 0.8, 0.0,  0.3,  0.0,  0.0,  1.5, "asymmetric"),
    ]

    all_ok = True
    for q0, qf, v0, vf, a0, af, T, label in test_cases:
        c = poly5_coeffs_general(q0, qf, v0, vf, a0, af, T)
        ok = verify_boundary_conditions(c, T, q0, qf, v0, vf, a0, af)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}")
        if not ok:
            all_ok = False

    print("\nVerifying bisection T_min...")
    T_found = find_T_min(
        q0=0.0, qf=1.0, v0=0.0, vf=0.5,
        a0=0.0, af=0.0,
        v_max=2.175, a_max=10.0, j_max=5000.0,
    )
    print(f"  T_min for rest→non-rest (Δq=1.0, vf=0.5): {T_found:.4f} s")

    print("\nAll checks:", "PASSED" if all_ok else "SOME FAILED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock",   action="store_true",
                        help="Skip IK and Pinocchio (polynomial only)")
    parser.add_argument("--verify", action="store_true",
                        help="Run boundary condition verification")
    args = parser.parse_args()

    if args.verify:
        run_verify()
    else:
        run_full(mock=args.mock)