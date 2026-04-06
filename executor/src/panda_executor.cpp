/**
 * panda_executor.cpp  (v8 — adds actual trajectory logging)
 * ===========================================================
 * Same as v7 plus: writes actual_trajectory.csv during execution.
 *
 * actual_trajectory.csv columns (36 total):
 *   t | q0..q6 | dq0..dq6 | tau_ext0..tau_ext6 | tau_cmd0..tau_cmd6 | sigma_min | tau_ext_max
 *
 * This is read by plot_trajectory.py to compare planned vs actual.
 */

#include <array>
#include <atomic>
#include <cmath>
#include <csignal>
#include <fstream>
#include <iomanip>
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

// ── Parameters ────────────────────────────────────────────────────────────────
static constexpr double kSettleSec     = 1.5;
static constexpr int    kRampMs        = 500;
static constexpr int    kRampDownMs    = 500;
static constexpr double kMaxTorqueRate = 0.5;
// Joints 2 & 4 (indices 1, 3) carry the highest inertial loads — give them
// stronger feedback to resist dynamic coupling from each other.
static constexpr std::array<double,7> kKp = {100,100,100,100,50,50,50};
static constexpr std::array<double,7> kKd = { 14, 14, 14, 14, 7, 7, 7};
static constexpr double kTauExtMax     = 10.0;
static constexpr double kSigmaMinStop  =  0.002;
static constexpr double kSigmaMinWarn  =  0.04;
static constexpr std::array<double,7> kVelHard = {
    2.11, 2.11, 2.11, 2.11, 2.53, 2.53, 2.53
};
static constexpr double kMoveToStartSpd = 0.12;
// Throw mode: stop trajectory playback at the peak-velocity midpoint (N/2)
// instead of running the full deceleration phase.  Set to true for ball throws.
static constexpr bool   kThrowMode      = false;

static constexpr int    kStatusInterval = 1000;

// ── Trajectory point ──────────────────────────────────────────────────────────
struct TrajPoint {
    double t;
    std::array<double,7> q, dq, ddq, tau;
};

// ── Actual data row (logged during execution) ─────────────────────────────────
struct ActualRow {
    double t;
    std::array<double,7> q;        // measured joint positions
    std::array<double,7> dq;       // measured joint velocities
    std::array<double,7> tau_ext;  // external torque estimate
    std::array<double,7> tau_cmd;  // torque command we sent
    double sigma_min;
    double tau_ext_max;
};

// ── CSV loaders / writers ─────────────────────────────────────────────────────
std::vector<TrajPoint> load_trajectory(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open: " + path);
    std::vector<TrajPoint> traj;
    std::string line;
    std::getline(f, line);
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line); std::string tok;
        std::vector<double> v;
        while (std::getline(ss, tok, ',')) v.push_back(std::stod(tok));
        if (v.size() != 29)
            throw std::runtime_error("CSV: expected 29 cols, got "
                                     + std::to_string(v.size()));
        TrajPoint p; p.t = v[0];
        for (int j=0;j<7;++j){ p.q[j]=v[1+j]; p.dq[j]=v[8+j];
                                p.ddq[j]=v[15+j]; p.tau[j]=v[22+j]; }
        traj.push_back(p);
    }
    if (traj.empty()) throw std::runtime_error("Empty: " + path);
    std::cout << "[LOAD] " << traj.size() << " points  (duration="
              << traj.back().t << "s)\n";
    return traj;
}

void write_actual_csv(const std::string& path,
                      const std::vector<ActualRow>& rows) {
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "[WARN] Could not write actual trajectory to " << path << "\n";
        return;
    }
    f << std::fixed << std::setprecision(8);

    // Header
    f << "t";
    for (int j=0;j<7;++j) f << ",q"   << j;
    for (int j=0;j<7;++j) f << ",dq"  << j;
    for (int j=0;j<7;++j) f << ",tau_ext" << j;
    for (int j=0;j<7;++j) f << ",tau_cmd" << j;
    f << ",sigma_min,tau_ext_max\n";

    for (const auto& r : rows) {
        f << r.t;
        for (int j=0;j<7;++j) f << "," << r.q[j];
        for (int j=0;j<7;++j) f << "," << r.dq[j];
        for (int j=0;j<7;++j) f << "," << r.tau_ext[j];
        for (int j=0;j<7;++j) f << "," << r.tau_cmd[j];
        f << "," << r.sigma_min << "," << r.tau_ext_max << "\n";
    }
    std::cout << "[LOG] Actual trajectory saved to " << path
              << "  (" << rows.size() << " rows)\n";
}

// ── Helpers ───────────────────────────────────────────────────────────────────
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
    for (int j=0;j<7;++j){
        double d = std::max(-maxd, std::min(maxd, des[j]-prev[j]));
        out[j] = prev[j] + d;
    }
    return out;
}

std::atomic<bool> g_stop{false};
void sigint_handler(int) { g_stop.store(true); }

// ── Move to start ─────────────────────────────────────────────────────────────
void move_to_start(franka::Robot& robot,
                   const std::array<double,7>& q_goal) {
    std::cout << "[INIT] Moving to start configuration...\n";
    auto q0 = robot.readOnce().q;
    double t_move = 0.0;
    for (int j=0;j<7;++j)
        t_move = std::max(t_move,
            std::abs(q_goal[j]-q0[j]) / (kMoveToStartSpd*2.175));
    t_move = std::max(t_move, 2.0);
    double t_total = t_move + kSettleSec;
    double t = 0.0;
    robot.control([&](const franka::RobotState&,
                       franka::Duration period) -> franka::JointPositions {
        t += period.toSec();
        if (t < t_move) {
            double a = t/t_move, s = a*a*(3.0-2.0*a);
            std::array<double,7> cmd;
            for (int j=0;j<7;++j) cmd[j]=q0[j]+s*(q_goal[j]-q0[j]);
            return franka::JointPositions(cmd);
        }
        if (t >= t_total)
            return franka::MotionFinished(franka::JointPositions(q_goal));
        return franka::JointPositions(q_goal);
    });
    std::cout << "[INIT] Settled at start configuration.\n";
}

// ── Stats ─────────────────────────────────────────────────────────────────────
struct Stats {
    size_t steps=0; double max_tau_ext=0, min_sigma=1e9;
    bool estop=false; std::string estop_reason;
};

// ── Execute trajectory ────────────────────────────────────────────────────────
void execute_trajectory(franka::Robot& robot,
                        franka::Model& model,
                        const std::vector<TrajPoint>& traj,
                        Stats& stats,
                        std::vector<ActualRow>& log) {
    // ── Phase-joint setup (Option 3 feedforward) ─────────────────────────────
    // Find the joint with the largest displacement; its actual position is used
    // as a phase variable to look up the trajectory index.  This keeps the
    // feedforward (q_des, dq_des, ddq_des) matched to where the robot actually
    // is, not where the wall-clock says it should be — eliminating the
    // inertial-coupling overshoot seen with pure time-indexed playback.
    int phase_joint = 0;
    {
        double max_d = 0.0;
        for (int j = 0; j < 7; ++j) {
            double d = std::abs(traj.back().q[j] - traj.front().q[j]);
            if (d > max_d) { max_d = d; phase_joint = j; }
        }
    }
    const double q_phase_start = traj.front().q[phase_joint];
    const double q_phase_range = traj.back().q[phase_joint] - q_phase_start;
    std::cout << "[EXEC] Phase joint: j" << phase_joint+1
              << "  range=" << q_phase_range << " rad\n";

    size_t idx = 0;
    size_t phase_ti_prev = 0;          // monotonic lower bound for phase lookup
    std::array<double,7> tau_ff_prev = {0,0,0,0,0,0,0};
    double t_exec = 0.0;   // wall time inside torque control

    robot.control([&](const franka::RobotState& state,
                       franka::Duration period) -> franka::Torques {

        const std::array<double,7> zero = {0,0,0,0,0,0,0};
        t_exec += period.toSec();

        // ── Safety checks ─────────────────────────────────────────────────
        if (g_stop.load()) {
            stats.estop=true; stats.estop_reason="SIGINT";
            return franka::MotionFinished(franka::Torques(zero));
        }
        if (state.robot_mode==franka::RobotMode::kReflex ||
            state.robot_mode==franka::RobotMode::kUserStopped) {
            stats.estop=true; stats.estop_reason="Reflex/user-stop";
            return franka::MotionFinished(franka::Torques(zero));
        }

        Eigen::Map<const Eigen::Matrix<double,7,1>> tau_ext_vec(
            state.tau_ext_hat_filtered.data());
        double te = tau_ext_vec.cwiseAbs().maxCoeff();
        stats.max_tau_ext = std::max(stats.max_tau_ext, te);
        if (te > kTauExtMax) {
            stats.estop=true;
            stats.estop_reason="Contact tau_ext="+std::to_string(te)+"Nm";
            std::cerr<<"[E-STOP] "<<stats.estop_reason<<"\n";
            return franka::MotionFinished(franka::Torques(zero));
        }

        double sm = sigma_min_val(model, state);
        stats.min_sigma = std::min(stats.min_sigma, sm);
        if (sm < kSigmaMinStop) {
            stats.estop=true;
            stats.estop_reason="Singularity sigma="+std::to_string(sm);
            std::cerr<<"[E-STOP] "<<stats.estop_reason<<"\n";
            return franka::MotionFinished(franka::Torques(zero));
        }
        if (sm < kSigmaMinWarn)
            std::cerr<<"[WARN] sigma_min="<<sm<<"\n";

        // ── Desired state ─────────────────────────────────────────────────
        std::array<double,7> q_des, dq_des, tau_ff_des;

        if (idx < (size_t)kRampMs) {
            double alpha = double(idx)/double(kRampMs);
            q_des  = traj[0].q; dq_des = zero;
            for (int j=0;j<7;++j)
                tau_ff_des[j] = alpha * traj[0].tau[j];

        } else {
            size_t ti = idx - kRampMs;
            // In throw mode, treat trajectory as finished at the midpoint
            // (peak velocity) so the arm never decelerates before release.
            const size_t traj_end = kThrowMode ? traj.size() / 2 : traj.size();
            if (ti < traj_end) {
                for (int j=0;j<7;++j) {
                    if (std::abs(state.dq[j]) > kVelHard[j]) {
                        stats.estop=true;
                        stats.estop_reason="Vel limit j"+std::to_string(j)
                            +"="+std::to_string(state.dq[j])+"rad/s";
                        std::cerr<<"[E-STOP] "<<stats.estop_reason<<"\n";
                        return franka::MotionFinished(franka::Torques(zero));
                    }
                }
                // ── Phase-indexed reference ───────────────────────────────
                // Map the dominant joint's actual position to a trajectory
                // index.  The result is clamped to [phase_ti_prev, traj_end-1]
                // so it never regresses (monotonic advance only).
                size_t phase_ti = ti;
                if (std::abs(q_phase_range) > 0.01) {
                    double phase = (state.q[phase_joint] - q_phase_start)
                                   / q_phase_range;
                    phase = std::max(0.0, std::min(1.0, phase));
                    size_t cand = static_cast<size_t>(
                        phase * double(traj.size() - 1));
                    phase_ti = std::max(phase_ti_prev,
                                        std::min(cand, traj_end - 1));
                }
                phase_ti_prev = phase_ti;

                q_des  = traj[phase_ti].q;
                dq_des = traj[phase_ti].dq;
                // Online RNEA: gravity + coriolis at actual state,
                // inertial term M·ddq uses phase-indexed desired accel.
                {
                    auto grav  = model.gravity(state);
                    auto cor   = model.coriolis(state);
                    auto M_arr = model.mass(state);
                    for (int j=0;j<7;++j) {
                        tau_ff_des[j] = grav[j] + cor[j];
                        for (int k=0;k<7;++k)
                            tau_ff_des[j] += M_arr[j + k*7] * traj[phase_ti].ddq[k];
                    }
                }
                ++stats.steps;
                if (stats.steps%(size_t)kStatusInterval==0)
                    std::cout<<"[EXEC] t="<<traj[ti].t
                             <<"s  sigma="<<sm
                             <<"  tau_ext="<<te<<" Nm\n";
            } else {
                // Ramp-down (either after full traj or throw midpoint)
                size_t rd = ti - traj_end;
                if (rd >= (size_t)kRampDownMs) {
                    auto grav = model.gravity(state);
                    auto cmd  = rate_limit(grav, tau_ff_prev, kMaxTorqueRate);
                    tau_ff_prev = cmd;
                    return franka::MotionFinished(franka::Torques(cmd));
                }
                double alpha = double(rd)/double(kRampDownMs);
                auto grav = model.gravity(state);
                // Hold the last active trajectory point as reference
                const size_t last_ti = traj_end - 1;
                q_des = traj[last_ti].q; dq_des = zero;
                for (int j=0;j<7;++j)
                    tau_ff_des[j]=(1.0-alpha)*traj[last_ti].tau[j]
                                 +alpha*grav[j];
            }
        }

        // ── Rate-limit and compute torque command ─────────────────────────
        auto tau_ff = rate_limit(tau_ff_des, tau_ff_prev, kMaxTorqueRate);
        tau_ff_prev = tau_ff;

        std::array<double,7> tau_cmd;
        for (int j=0;j<7;++j)
            tau_cmd[j] = tau_ff[j]
                       + kKp[j]*(q_des[j] -state.q[j])
                       + kKd[j]*(dq_des[j]-state.dq[j]);

        // ── Log actual state (tracking phase only) ────────────────────────
        const size_t traj_end_log = kThrowMode ? traj.size()/2 : traj.size();
        if (idx >= (size_t)kRampMs &&
            (idx - kRampMs) < traj_end_log) {
            ActualRow row;
            row.t          = traj[idx - kRampMs].t;
            row.q          = state.q;
            row.dq         = state.dq;
            row.tau_ext    = state.tau_ext_hat_filtered;
            row.tau_cmd    = tau_cmd;
            row.sigma_min  = sm;
            row.tau_ext_max = te;
            log.push_back(row);
        }

        ++idx;
        return franka::Torques(tau_cmd);
    });
}

// ── main ──────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr<<"Usage: "<<argv[0]
                 <<" <trajectory.csv> <robot_ip> [actual_out.csv]\n";
        return 1;
    }
    std::string actual_path = (argc >= 4)
        ? argv[3] : "actual_trajectory.csv";

    std::vector<TrajPoint> traj;
    try { traj = load_trajectory(argv[1]); }
    catch(const std::exception& e){
        std::cerr<<"[ERROR] "<<e.what()<<"\n"; return 1;
    }

    std::cout<<"[NET] Connecting to "<<argv[2]<<"...\n";
    franka::Robot robot(argv[2]);
    franka::Model model = robot.loadModel();

    robot.setCollisionBehavior(
        {{100,100,100,100,100,100,100}},
        {{100,100,100,100,100,100,100}},
        {{ 50, 50, 50, 50, 50, 50}},
        {{ 50, 50, 50, 50, 50, 50}}
    );

    std::signal(SIGINT, sigint_handler);
    try {
        robot.automaticErrorRecovery();
        move_to_start(robot, traj.front().q);
    } catch(const franka::Exception& e){
        std::cerr<<"[ERROR] "<<e.what()<<"\n"; return 1;
    }

    std::cout<<"\n[READY] "<<traj.size()<<" samples, "
             <<traj.back().t<<"s\n"
             <<"  Ramp-up   : "<<kRampMs<<" ms\n"
             <<"  Ramp-down : "<<kRampDownMs<<" ms\n"
             <<"  Log file  : "<<actual_path<<"\n\n"
             <<"  Press ENTER to execute, Ctrl-C to abort.\n";
    std::cin.get();
    if (g_stop.load()){std::cout<<"[ABORT]\n";return 0;}

    Stats stats;
    std::vector<ActualRow> log;
    log.reserve(traj.size());

    std::cout<<"[EXEC] Starting torque control...\n";
    try {
        execute_trajectory(robot, model, traj, stats, log);
    } catch(const franka::Exception& e){
        std::string msg=e.what();
        if (msg.find("MotionFinished")==std::string::npos)
            std::cerr<<"[ERROR] "<<msg<<"\n";
    }

    write_actual_csv(actual_path, log);

    std::cout<<"\n[DONE] Steps    : "<<stats.steps<<"/"<<traj.size()<<"\n"
             <<"       tau_ext  : "<<stats.max_tau_ext<<" Nm\n"
             <<"       sigma_min: "<<stats.min_sigma<<"\n";
    if (stats.estop){
        std::cerr<<"       E-STOP  : "<<stats.estop_reason<<"\n";
        return 1;
    }
    std::cout<<"       Status  : SUCCESS\n";
    return 0;
}
