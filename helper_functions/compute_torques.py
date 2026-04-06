"""
helper_functions/compute_torques.py
=====================================
Compute joint torques via RNEA (Pinocchio) for a given trajectory CSV and
write them back into the tau0..tau6 columns.

Input CSV columns (at minimum):
    t | q0..q6 | dq0..dq6 | ddq0..ddq6

Output CSV columns (29 total):
    t | q0..q6 | dq0..dq6 | ddq0..ddq6 | tau0..tau6

Usage:
    cd helper_functions/
    python compute_torques.py --input ../trajectory.csv --output ../trajectory_with_tau.csv

    # With a payload attached at the flange:
    python compute_torques.py --input traj.csv --output traj_tau.csv \
        --payload-mass 0.3 --payload-com "0 0 0.06"

    # Overwrite the input file in-place:
    python compute_torques.py --input traj.csv --inplace
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# ── Pinocchio ─────────────────────────────────────────────────────────────────
try:
    import pinocchio as pin
    HAS_PINOCCHIO = True
except ImportError:
    HAS_PINOCCHIO = False
    print("[ERROR] Pinocchio is not installed. Install it with:")
    print("        conda install pinocchio -c conda-forge")
    print("   or:  pip install pin")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

N_JOINTS = 7

# Default URDF location relative to this file (mirrors planner layout)
_HERE = Path(__file__).resolve().parent
DEFAULT_URDF = _HERE.parent / "planner" / "franka_panda" / "panda.urdf"

# Panda hardware torque limits (for a summary printout only — not enforced)
TAU_MAX = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])


# ─────────────────────────────────────────────────────────────────────────────
#  Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(urdf_path: Path,
               payload_mass: float = 0.0,
               payload_com: np.ndarray = None,
               payload_inertia: np.ndarray = None):
    """
    Load Panda URDF into a Pinocchio model and optionally attach a payload
    to joint 7 (the flange).

    Returns (model, data).
    """
    if not urdf_path.exists():
        print(f"[ERROR] URDF not found: {urdf_path}")
        sys.exit(1)

    model, _, _ = pin.buildModelsFromUrdf(str(urdf_path), str(urdf_path.parent))

    # Detect extra DOF beyond the 7 arm joints (e.g. gripper fingers in some URDFs)
    model.arm_extra_dof = model.nq - N_JOINTS
    if model.arm_extra_dof > 0:
        print(f"[INFO] URDF has {model.arm_extra_dof} extra DOF (e.g. gripper) — "
              "held fixed at 0.")

    if payload_mass > 0.0:
        if payload_com is None:
            payload_com = np.zeros(3)
        if payload_inertia is None:
            payload_inertia = np.eye(3) * 1e-5

        jid = model.getJointId("panda_joint7")
        if jid < model.njoints:
            model.inertias[jid] += pin.Inertia(
                payload_mass, payload_com, payload_inertia
            )
            print(f"[INFO] Payload {payload_mass:.3f} kg attached to joint 7, "
                  f"CoM = {payload_com}")

    data = model.createData()
    return model, data


# ─────────────────────────────────────────────────────────────────────────────
#  CSV helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(path: Path):
    """
    Load the trajectory CSV.  Returns (header_list, data_array).

    Accepts files that:
      - have the 22-column variant (no tau columns yet), or
      - have the full 29-column variant (tau columns will be overwritten).
    """
    with open(path) as f:
        header_line = f.readline().strip()

    headers = [h.strip() for h in header_line.split(",")]
    data = np.loadtxt(path, delimiter=",", skiprows=1)

    n_cols = data.shape[1]
    if n_cols not in (22, 29):
        print(f"[WARN] Unexpected column count ({n_cols}). "
              "Expected 22 (no tau) or 29 (with tau).")

    return headers, data


def save_csv(path: Path, headers: list, data: np.ndarray) -> None:
    header_str = ",".join(headers)
    np.savetxt(str(path), data, delimiter=",",
               header=header_str, comments="", fmt="%.18e")
    print(f"[INFO] Saved {data.shape[0]} rows × {data.shape[1]} cols → {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Core: RNEA over the full trajectory
# ─────────────────────────────────────────────────────────────────────────────

def compute_rnea_torques(model, data, Q: np.ndarray,
                         Qd: np.ndarray, Qdd: np.ndarray) -> np.ndarray:
    """
    Run RNEA for every row in Q/Qd/Qdd.

    Parameters
    ----------
    Q, Qd, Qdd : ndarray, shape (N, 7)

    Returns
    -------
    tau : ndarray, shape (N, 7)
    """
    extra = getattr(model, "arm_extra_dof", 0)
    N = Q.shape[0]
    tau = np.zeros((N, N_JOINTS))

    for i in range(N):
        q_full   = np.concatenate([Q[i],   np.zeros(extra)]) if extra else Q[i]
        qd_full  = np.concatenate([Qd[i],  np.zeros(extra)]) if extra else Qd[i]
        qdd_full = np.concatenate([Qdd[i], np.zeros(extra)]) if extra else Qdd[i]

        tau_full = pin.rnea(model, data, q_full, qd_full, qdd_full)
        tau[i]   = tau_full[:N_JOINTS]

    return tau


# ─────────────────────────────────────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fill tau0..tau6 columns in a Panda trajectory CSV using RNEA."
    )
    parser.add_argument("--input",  required=True,
                        help="Input CSV (must have t, q0..q6, dq0..dq6, ddq0..ddq6)")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: <input>_tau.csv)")
    parser.add_argument("--inplace", action="store_true",
                        help="Overwrite the input file in-place")
    parser.add_argument("--urdf",   default=str(DEFAULT_URDF),
                        help=f"Path to panda.urdf  (default: {DEFAULT_URDF})")
    parser.add_argument("--payload-mass", type=float, default=0.0,
                        help="Payload mass in kg  (default: 0)")
    parser.add_argument("--payload-com",  default="0 0 0",
                        help='Payload CoM in flange frame, metres  (default: "0 0 0")')
    parser.add_argument("--payload-inertia", default=None,
                        help='Diagonal of payload inertia tensor kg·m², '
                             'e.g. "5e-4 5e-4 3e-4"  (default: near-zero)')
    args = parser.parse_args()

    in_path  = Path(args.input)
    if args.inplace:
        out_path = in_path
    elif args.output:
        out_path = Path(args.output)
    else:
        out_path = in_path.with_stem(in_path.stem + "_tau")

    # Parse payload args
    com = np.array([float(x) for x in args.payload_com.split()])
    if args.payload_inertia:
        diag = np.array([float(x) for x in args.payload_inertia.split()])
        inertia = np.diag(diag)
    else:
        inertia = None

    # ── Load model ────────────────────────────────────────────────────────────
    model, data = load_model(
        Path(args.urdf),
        payload_mass=args.payload_mass,
        payload_com=com,
        payload_inertia=inertia,
    )
    print(f"[INFO] Loaded URDF: {args.urdf}")

    # ── Load trajectory ───────────────────────────────────────────────────────
    headers, traj = load_csv(in_path)
    N = traj.shape[0]
    print(f"[INFO] Trajectory: {N} samples, {traj.shape[1]} columns")

    # Column layout: t(1) | q(7) | dq(7) | ddq(7) | tau(7, optional)
    t_col  = traj[:, 0]
    Q      = traj[:, 1:8]
    Qd     = traj[:, 8:15]
    Qdd    = traj[:, 15:22]

    # ── RNEA ──────────────────────────────────────────────────────────────────
    print(f"[INFO] Running RNEA on {N} samples …")
    tau = compute_rnea_torques(model, data, Q, Qd, Qdd)

    # Summary
    tau_peak = np.max(np.abs(tau), axis=0)
    print(f"[INFO] Peak |tau| per joint (Nm): {np.round(tau_peak, 3)}")
    violations = np.where(tau_peak > TAU_MAX)[0]
    if violations.size:
        print(f"[WARN] Joints {violations.tolist()} exceed hardware limits "
              f"({TAU_MAX[violations].tolist()} Nm). Check trajectory feasibility.")
    else:
        print("[INFO] All torques within hardware limits.")

    # ── Build output CSV ──────────────────────────────────────────────────────
    tau_headers = [f"tau{j}" for j in range(N_JOINTS)]
    if traj.shape[1] == 29:
        # Overwrite existing tau columns
        out_data = np.column_stack([traj[:, :22], tau])
        out_headers = headers[:22] + tau_headers
    else:
        # Append tau columns
        out_data = np.column_stack([traj, tau])
        base_headers = headers if len(headers) == 22 else (
            ["t"]
            + [f"q{j}"   for j in range(N_JOINTS)]
            + [f"dq{j}"  for j in range(N_JOINTS)]
            + [f"ddq{j}" for j in range(N_JOINTS)]
        )
        out_headers = base_headers + tau_headers

    save_csv(out_path, out_headers, out_data)


if __name__ == "__main__":
    main()
