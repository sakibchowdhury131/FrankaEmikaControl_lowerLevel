"""
planner/main_plan.py
====================
Entry point for the offline planning stage.

Edit Q_START, Q_END, PAYLOAD, URDF_PATH below, then run:

    cd planner/
    python main_plan.py               # full planning with Pinocchio
    python main_plan.py --mock        # skip Pinocchio (no URDF needed)

Output:
    ../trajectory.csv   <-- loaded by the C++ executor

CSV columns (29 total):
    t | q0..q6 | dq0..dq6 | ddq0..ddq6 | tau0..tau6
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from trajectory_planner import PlannerConfig, PayloadParams, plan

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
#  USER CONFIGURATION  — edit these values
# ═════════════════════════════════════════════════════════════════════════════

Q_START = np.array([0.0, -0.785, 0.0, -1.856, 0.0, 2.100, 0.785])
Q_END   = np.array([0.0,  0.200, 0.3, -1.800, 0.2, 2.100, 1.200])

PAYLOAD = PayloadParams(
    mass    = 0.3,                           # kg
    com     = np.array([0.0, 0.0, 0.06]),    # m, in flange frame
    inertia = np.diag([5e-4, 5e-4, 3e-4]),  # kg·m²
)

URDF_PATH  = "franka_panda/panda.urdf"   # path to Panda URDF
OUTPUT_CSV = "../trajectory.csv"         # picked up by C++ executor

# ═════════════════════════════════════════════════════════════════════════════


def main(mock: bool = False) -> None:
    log.info("═══════════════════════════════════════════")
    log.info("  Panda Offline Trajectory Planner")
    log.info("═══════════════════════════════════════════")

    cfg = PlannerConfig(
        q_start    = Q_START,
        q_end      = Q_END,
        payload    = PAYLOAD,
        urdf_path  = "MOCK" if mock else URDF_PATH,
        control_hz = 1000,
        output_csv = OUTPUT_CSV,
    )

    try:
        result = plan(cfg)
    except RuntimeError as e:
        log.error("Planning FAILED: %s", e)
        sys.exit(1)

    log.info("─── Trajectory summary ─────────────────────")
    log.info("  Duration   : %.3f s", result.T)
    log.info("  Samples    : %d  (%.0f Hz)", len(result.t_arr),
             1.0 / (result.t_arr[1] - result.t_arr[0]))
    log.info("  Max |tau|  : %s Nm",
             np.round(np.max(np.abs(result.tau), axis=0), 2))
    log.info("  Min σ_min  : %.4f m/rad", float(np.min(result.sigma_min)))
    log.info("  Min manip. : %.4f",       float(np.min(result.w)))
    log.info("  CSV saved  : %s", OUTPUT_CSV)
    log.info("────────────────────────────────────────────")
    log.info("Now build and run the C++ executor:")
    log.info("  cd ../executor && mkdir build && cd build")
    log.info("  cmake .. && make")
    log.info("  ./panda_executor ../trajectory.csv 192.168.1.1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true",
                        help="Skip Pinocchio checks (offline polynomial only)")
    args = parser.parse_args()
    main(args.mock)
