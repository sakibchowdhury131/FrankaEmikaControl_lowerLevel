/**
 * panda_executor.cpp  (v7 — final)
 * ==================================
 * Pure torque control with PD correction + clean end-of-trajectory ramp-down.
 *
 * CONTROL LAW:
 *   tau_cmd = tau_ff(t) + Kp*(q_des(t) - q) + Kd*(dq_des(t) - dq)
 *
 * SEQUENCE:
 *   1. move_to_start()     — position control, smooth move + 1.5s settle
 *   2. execute_trajectory() — pure torque control (robot.control(torque_cb))
 *        a. Ramp phase   (kRampMs):    hold q_start, tau_ff ramps 0 → traj[0].tau
 *        b. Track phase  (N samples):  tau_ff = traj[ti].tau, PD corrects drift
 *        c. Ramp-down    (kRampDownMs): tau_ff ramps traj.back().tau → gravity
 *           Robot is stationary at end, so MotionFinished is clean.
 *
 * RUNTIME MONITORS (inside torque callback):
 *   - External torque contact  (tau_ext_hat_filtered > kTauExtMax)
 *   - Singularity              (sigma_min of end-effector Jacobian < kSigmaMinStop)
 *   - Joint velocity hard limit (during tracking phase only)
 *   - Robot mode reflex / user-stop
 *   - SIGINT (Ctrl-C)
 */

#include <array>
#include <atomic>
#include <cmath>
#include <csignal>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/SVD>

#include <franka/duration.h>
#include <franka/exception.h>
#include <franka/model.h>
#include <franka/robot.h>
#include <franka/robot_state.h>

// ─────────────────────────────────────────────────────────────────────────────
//  Parameters — edit these to tune behaviour
// ─────────────────────────────────────────────────────────────────────────────

// Position-control settle time after reaching q_start [s]
static constexpr double kSettleSec     = 1.5;

// Torque ramp-up: tau_ff goes 0 → traj[0].tau over this many ms
static constexpr int    kRampMs        = 500;

// Torque ramp-down: tau_ff goes traj.back().tau → gravity over this many ms
// Eliminates "robot still moving" warning at end of trajectory
static constexpr int    kRampDownMs    = 150;

// Maximum torque change between consecutive 1ms steps [Nm/ms]
static constexpr double kMaxTorqueRate = 0.5;

// PD gains — corrects for PyBullet inertia inaccuracy during tracking
static constexpr std::array<double,7> kKp = {50,50,50,50,20,20,20}; // [Nm/rad]
static constexpr std::array<double,7> kKd = { 7, 7, 7, 7, 3, 3, 3}; // [Nm·s/rad]

// E-stop thresholds
static constexpr double kTauExtMax     = 10.0;  // [Nm]  contact detection
static constexpr double kSigmaMinStop  =  0.02; // [m/rad] singularity E-stop
static constexpr double kSigmaMinWarn  =  0.04; // [m/rad] singularity warning

// Velocity hard limit during tracking: 97% of hardware limits [rad/s]
// Hardware: joints 1-4 = 2.175 rad/s, joints 5-7 = 2.61 rad/s
static constexpr std::array<double,7> kVelHard = {
    2.11, 2.11, 2.11, 2.11, 2.53, 2.53, 2.53
};

static constexpr double kMoveToStartSpd = 0.12;  // fraction of max speed
static constexpr int    kStatusInterval = 1000;   // print every N steps


// ─────────────────────────────────────────────────────────────────────────────
//  Trajectory point
// ─────────────────────────────────────────────────────────────────────────────

struct TrajPoint {
    double t;
    std::array<double,7> q, dq, ddq, tau;
};


// ─────────────────────────────────────────────────────────────────────────────
//  CSV loader
//  Expected columns (29 total):
//    t | q0..q6 | dq0..dq6 | ddq0..ddq6 | tau0..tau6
// ─────────────────────────────────────────────────────────────────────────────

std::vector<TrajPoint> load_trajectory(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open trajectory file: " + path);

    std::vector<TrajPoint> traj;
    std::string line;
    std::getline(f, line);  // skip header row

    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string tok;
        std::vector<double> v;
        while (std::getline(ss, tok, ','))
            v.push_back(std::stod(tok));

        if (v.size() != 29)
            throw std::runtime_error(
                "CSV row has " + std::to_string(v.size()) +
                " columns, expected 29"
            );

        TrajPoint p;
        p.t = v[0];
        for (int j = 0; j < 7; ++j) {
            p.q[j]   = v[1  + j];
            p.dq[j]  = v[8  + j];
            p.ddq[j] = v[15 + j];
            p.tau[j] = v[22 + j];
        }
        traj.push_back(p);
    }

    if (traj.empty())
        throw std::runtime_error("Trajectory file is empty: " + path);

    std::cout << "[LOAD] " << traj.size() << " points  (duration="
              << traj.back().t << "s)\n";
    return traj;
}


// ─────────────────────────────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────────────────────────────

double sigma_min_val(franka::Model& model, const franka::RobotState& state) {
    auto J_arr = model.zeroJacobian(franka::Frame::kEndEffector, state);
    Eigen::Map<const Eigen::Matrix<double,6,7,Eigen::ColMajor>> J(J_arr.data());
    Eigen::JacobiSVD<Eigen::Matrix<double,6,7>> svd(J);
    return svd.singularValues().minCoeff();
}

std::array<double,7> rate_limit(const std::array<double,7>& des,
                                 const std::array<double,7>& prev,
                                 double maxd) {
    std::array<double,7> out;
    for (int j = 0; j < 7; ++j) {
        double d = std::max(-maxd, std::min(maxd, des[j] - prev[j]));
        out[j] = prev[j] + d;
    }
    return out;
}

std::atomic<bool> g_stop{false};
void sigint_handler(int) { g_stop.store(true); }


// ─────────────────────────────────────────────────────────────────────────────
//  Step 1 — Move to start configuration (position control)
// ─────────────────────────────────────────────────────────────────────────────

void move_to_start(franka::Robot& robot,
                   const std::array<double,7>& q_goal) {
    std::cout << "[INIT] Moving to start configuration...\n";
    auto q0 = robot.readOnce().q;

    // Duration: slowest joint at kMoveToStartSpd fraction of max speed
    double t_move = 0.0;
    for (int j = 0; j < 7; ++j)
        t_move = std::max(t_move,
            std::abs(q_goal[j] - q0[j]) / (kMoveToStartSpd * 2.175));
    t_move = std::max(t_move, 2.0);   // at least 2 s
    double t_total = t_move + kSettleSec;
    double t = 0.0;

    robot.control([&](const franka::RobotState&,
                       franka::Duration period) -> franka::JointPositions {
        t += period.toSec();

        if (t < t_move) {
            // Smooth-step interpolation
            double a = t / t_move;
            double s = a * a * (3.0 - 2.0 * a);
            std::array<double,7> cmd;
            for (int j = 0; j < 7; ++j)
                cmd[j] = q0[j] + s * (q_goal[j] - q0[j]);
            return franka::JointPositions(cmd);
        }

        // Settle phase — hold q_goal; libfranka handles gravity exactly
        if (t >= t_total)
            return franka::MotionFinished(franka::JointPositions(q_goal));
        return franka::JointPositions(q_goal);
    });

    std::cout << "[INIT] Settled at start configuration.\n";
}


// ─────────────────────────────────────────────────────────────────────────────
//  Execution stats
// ─────────────────────────────────────────────────────────────────────────────

struct Stats {
    size_t steps      = 0;
    double max_tau_ext = 0.0;
    double min_sigma   = 1e9;
    bool   estop       = false;
    std::string estop_reason;
};


// ─────────────────────────────────────────────────────────────────────────────
//  Step 2 — Execute trajectory (pure torque control)
// ─────────────────────────────────────────────────────────────────────────────

void execute_trajectory(franka::Robot& robot,
                        franka::Model& model,
                        const std::vector<TrajPoint>& traj,
                        Stats& stats) {
    size_t idx = 0;
    std::array<double,7> tau_ff_prev = {0, 0, 0, 0, 0, 0, 0};

    robot.control([&](const franka::RobotState& state,
                       franka::Duration) -> franka::Torques {

        const std::array<double,7> zero = {0, 0, 0, 0, 0, 0, 0};

        // ── Safety checks (every step) ────────────────────────────────────

        if (g_stop.load()) {
            stats.estop = true;
            stats.estop_reason = "SIGINT";
            return franka::MotionFinished(franka::Torques(zero));
        }

        if (state.robot_mode == franka::RobotMode::kReflex ||
            state.robot_mode == franka::RobotMode::kUserStopped) {
            stats.estop = true;
            stats.estop_reason = "Robot reflex / user-stop";
            return franka::MotionFinished(franka::Torques(zero));
        }

        // External torque — contact detection
        Eigen::Map<const Eigen::Matrix<double,7,1>> tau_ext(
            state.tau_ext_hat_filtered.data());
        double te = tau_ext.cwiseAbs().maxCoeff();
        stats.max_tau_ext = std::max(stats.max_tau_ext, te);
        if (te > kTauExtMax) {
            stats.estop = true;
            stats.estop_reason =
                "Contact: tau_ext=" + std::to_string(te) + " Nm";
            std::cerr << "[E-STOP] " << stats.estop_reason << "\n";
            return franka::MotionFinished(franka::Torques(zero));
        }

        // Singularity
        double sm = sigma_min_val(model, state);
        stats.min_sigma = std::min(stats.min_sigma, sm);
        if (sm < kSigmaMinStop) {
            stats.estop = true;
            stats.estop_reason =
                "Singularity: sigma_min=" + std::to_string(sm);
            std::cerr << "[E-STOP] " << stats.estop_reason << "\n";
            return franka::MotionFinished(franka::Torques(zero));
        }
        if (sm < kSigmaMinWarn)
            std::cerr << "[WARN] sigma_min=" << sm << "\n";

        // ── Determine desired state and feedforward ───────────────────────

        std::array<double,7> q_des, dq_des, tau_ff_des;

        if (idx < (size_t)kRampMs) {
            // ── Phase A: ramp-up ──────────────────────────────────────────
            // Hold q_start, ramp tau_ff from 0 → traj[0].tau
            // PD term keeps robot at q_start while feedforward builds up
            double alpha = double(idx) / double(kRampMs);
            q_des  = traj[0].q;
            dq_des = zero;
            for (int j = 0; j < 7; ++j)
                tau_ff_des[j] = alpha * traj[0].tau[j];

        } else {
            size_t ti = idx - kRampMs;

            if (ti < traj.size()) {
                // ── Phase B: trajectory tracking ──────────────────────────

                // Velocity watchdog (tracking only — ramp phase is stationary)
                for (int j = 0; j < 7; ++j) {
                    if (std::abs(state.dq[j]) > kVelHard[j]) {
                        stats.estop = true;
                        stats.estop_reason =
                            "Vel limit j" + std::to_string(j) +
                            "=" + std::to_string(state.dq[j]) + " rad/s";
                        std::cerr << "[E-STOP] " << stats.estop_reason << "\n";
                        return franka::MotionFinished(franka::Torques(zero));
                    }
                }

                q_des      = traj[ti].q;
                dq_des     = traj[ti].dq;
                tau_ff_des = traj[ti].tau;
                ++stats.steps;

                if (stats.steps % (size_t)kStatusInterval == 0)
                    std::cout << "[EXEC] t=" << traj[ti].t
                              << "s  sigma=" << sm
                              << "  tau_ext=" << te << " Nm\n";

            } else {
                // ── Phase C: ramp-down ────────────────────────────────────
                // After last trajectory sample, ramp tau_ff smoothly from
                // traj.back().tau → model.gravity(state) over kRampDownMs.
                // Robot is near-stationary so PD keeps it there.
                // When ramp completes, MotionFinished is called with the
                // robot already at rest — no "still moving" warning.
                size_t rd_idx = ti - traj.size();

                if (rd_idx >= (size_t)kRampDownMs) {
                    // Ramp-down complete — robot is at rest, finish cleanly
                    auto grav = model.gravity(state);
                    auto cmd  = rate_limit(grav, tau_ff_prev, kMaxTorqueRate);
                    tau_ff_prev = cmd;
                    return franka::MotionFinished(franka::Torques(cmd));
                }

                double alpha = double(rd_idx) / double(kRampDownMs);
                auto grav    = model.gravity(state);
                q_des  = traj.back().q;
                dq_des = zero;
                for (int j = 0; j < 7; ++j)
                    tau_ff_des[j] = (1.0 - alpha) * traj.back().tau[j]
                                  + alpha          * grav[j];
            }
        }

        // ── Rate-limit feedforward ────────────────────────────────────────
        std::array<double,7> tau_ff =
            rate_limit(tau_ff_des, tau_ff_prev, kMaxTorqueRate);
        tau_ff_prev = tau_ff;

        // ── Full torque command: feedforward + PD correction ──────────────
        // tau = tau_ff(t) + Kp*(q_des - q) + Kd*(dq_des - dq)
        std::array<double,7> tau_cmd;
        for (int j = 0; j < 7; ++j)
            tau_cmd[j] = tau_ff[j]
                       + kKp[j] * (q_des[j]  - state.q[j])
                       + kKd[j] * (dq_des[j] - state.dq[j]);

        ++idx;
        return franka::Torques(tau_cmd);
    });
}


// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <trajectory.csv> <robot_ip>\n";
        return 1;
    }

    // Load trajectory
    std::vector<TrajPoint> traj;
    try {
        traj = load_trajectory(argv[1]);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }

    // Connect to robot
    std::cout << "[NET] Connecting to " << argv[2] << "...\n";
    franka::Robot robot(argv[2]);
    franka::Model model = robot.loadModel();

    // Raise hardware reflex thresholds so our software monitors fire first,
    // giving cleaner error messages
    robot.setCollisionBehavior(
        {{100, 100, 100, 100, 100, 100, 100}},
        {{100, 100, 100, 100, 100, 100, 100}},
        {{  50,  50,  50,  50,  50,  50}},
        {{  50,  50,  50,  50,  50,  50}}
    );

    std::signal(SIGINT, sigint_handler);
    try {
        robot.automaticErrorRecovery();
        move_to_start(robot, traj.front().q);
    } catch (const franka::Exception& e) {
        std::cerr << "[ERROR] Move to start failed: " << e.what() << "\n";
        return 1;
    }

    std::cout << "\n[READY] " << traj.size() << " samples, "
              << traj.back().t << "s\n"
              << "  Control      : pure torque + PD correction\n"
              << "  Ramp-up      : " << kRampMs     << " ms\n"
              << "  Ramp-down    : " << kRampDownMs << " ms\n"
              << "  Kp           : [50,50,50,50,20,20,20] Nm/rad\n"
              << "  Kd           : [7,7,7,7,3,3,3] Nm·s/rad\n"
              << "  Total wall   : "
              << (kRampMs + kRampDownMs) / 1000.0 + traj.back().t << "s\n\n"
              << "  Press ENTER to execute, Ctrl-C to abort.\n";
    std::cin.get();

    if (g_stop.load()) {
        std::cout << "[ABORT] Aborted by user.\n";
        return 0;
    }

    Stats stats;
    std::cout << "[EXEC] Starting torque control...\n";
    try {
        execute_trajectory(robot, model, traj, stats);
    } catch (const franka::Exception& e) {
        std::string msg = e.what();
        // MotionFinished internally causes an exception in some libfranka
        // versions — this is expected and not a real error
        if (msg.find("MotionFinished") == std::string::npos)
            std::cerr << "[ERROR] " << msg << "\n";
    }

    std::cout << "\n[DONE] Steps    : " << stats.steps
              << " / " << traj.size() << "\n"
              << "       tau_ext  : " << stats.max_tau_ext << " Nm\n"
              << "       sigma_min: " << stats.min_sigma   << "\n";

    if (stats.estop) {
        std::cerr << "       E-STOP  : " << stats.estop_reason << "\n";
        return 1;
    }

    std::cout << "       Status  : SUCCESS\n";
    return 0;
}
