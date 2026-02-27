#pragma once

#include "simulator.hpp"
#include <cmath>
#include <random>
#include <algorithm>
#include <vector>

// Vehicle parameters — mirrors src/params.py VehicleParams
struct VehicleParamsCpp {
    double mu      = 1.0489;
    double C_Sf    = 4.718;
    double C_Sr    = 5.4562;
    double lf      = 0.15875;
    double lr      = 0.17145;
    double h       = 0.074;
    double m       = 3.74;
    double I       = 0.04712;
    double s_min   = -0.4189;
    double s_max   = 0.4189;
    double sv_min  = -3.2;
    double sv_max  = 3.2;
    double v_switch = 7.319;
    double a_max   = 9.51;
    double v_min   = -5.0;
    double v_max   = 20.0;
    double width   = 0.31;
    double length  = 0.58;
};

// Precomputed occupancy map data (built in Python, passed to C++)
struct OccupancyMapData {
    std::vector<float> grid;   // flat row-major: grid[ix * ny + iy]
    double x_min;
    double y_min;
    double cell_size;
    int nx;
    int ny;

    // O(1) off-track lookup: returns 1.0 if off-track, 0.0 if on-track
    float query(double x, double y) const {
        int ix = static_cast<int>((x - x_min) / cell_size);
        int iy = static_cast<int>((y - y_min) / cell_size);
        if (ix < 0 || ix >= nx || iy < 0 || iy >= ny) return 1.0f;
        return grid[ix * ny + iy];
    }
};

// Cost function tuning parameters
struct CostParams {
    double speed_weight     = 100.0;
    double alpha_threshold  = 1.0;
    double v_ref            = 10.0;
    int    rollout_horizon  = 15;
    double discount_factor  = 0.99;
    double dt               = 0.04;
    bool   normalize_reward = false;
};

// F1TENTH simulator for MCTS rollouts.
// Implements the Simulator interface with F1TENTH single-track dynamics
// and the same cost function as the Python version.
class F1TenthSimulator : public Simulator {
  private:
    // Members ordered to match initializer list.
    VehicleParamsCpp vp_;
    OccupancyMapData occ_map_;
    CostParams cost_;
    // State: [x, y, delta, v, psi, omega, beta]
    std::vector<double> state_;
    int step_count_ = 0;
    std::vector<double> v_ref_window_;
    double v_ref_min_ = 1e9;  // min of v_ref_window, precomputed
    // Gaussian action prior from SVGP: [accel_mean, sv_mean], [accel_std, sv_std]
    std::vector<double> action_mean_;  // size 2
    std::vector<double> action_std_;   // size 2
    bool use_action_prior_ = false;
    std::mt19937 rng_;

    static constexpr double G = 9.81;
    static constexpr double TWO_PI = 2.0 * M_PI;

  public:
    F1TenthSimulator(const VehicleParamsCpp& vp,
                     const OccupancyMapData& occ_map,
                     const CostParams& cost,
                     unsigned seed = std::random_device{}())
        : vp_(vp), occ_map_(occ_map), cost_(cost),
          state_(7, 0.0), rng_(seed) {}

    // --- Simulator interface ---

    bool is_action_discrete() const override { return false; }
    bool is_state_discrete() const override { return false; }
    int rollout_horizon() const override { return cost_.rollout_horizon; }
    double discount() const override { return cost_.discount_factor; }

    void set_state(const std::vector<double>& state) override {
        state_ = state;
        step_count_ = 0;
    }

    void set_v_ref_window(const std::vector<double>& window) {
        v_ref_window_ = window;
        v_ref_min_ = window.empty()
            ? 1e9
            : *std::min_element(window.begin(), window.end());
    }

    void set_action_prior(const std::vector<double>& mean,
                          const std::vector<double>& std) {
        action_mean_ = mean;
        action_std_ = std;
        use_action_prior_ = (mean.size() == 2 && std.size() == 2);
    }

    void set_rollout_horizon(int h) override { cost_.rollout_horizon = h; }

    StepResult step(const std::vector<double>& action) override {
        double x     = state_[0];
        double y     = state_[1];
        double delta = state_[2];
        double v     = state_[3];
        double psi   = state_[4];
        double omega = state_[5];
        double beta  = state_[6];

        double accl = action[0];
        double sv   = action[1];

        // --- Steering constraint ---
        if ((delta <= vp_.s_min && sv <= 0.0) ||
            (delta >= vp_.s_max && sv >= 0.0)) {
            sv = 0.0;
        }
        sv = std::max(vp_.sv_min, std::min(vp_.sv_max, sv));

        // --- Acceleration constraint ---
        double pos_limit;
        if (v > vp_.v_switch) {
            pos_limit = vp_.a_max * vp_.v_switch / v;
        } else {
            pos_limit = vp_.a_max;
        }
        if ((v <= vp_.v_min && accl <= 0.0) ||
            (v >= vp_.v_max && accl >= 0.0)) {
            accl = 0.0;
        }
        accl = std::max(-vp_.a_max, std::min(pos_limit, accl));

        double L = vp_.lf + vp_.lr;

        double f_x, f_y, f_delta, f_v, f_psi, f_omega, f_beta;

        if (std::abs(v) < 0.5) {
            // Kinematic model
            f_x     = v * std::cos(psi);
            f_y     = v * std::sin(psi);
            f_delta = sv;
            f_v     = accl;
            f_psi   = v / L * std::tan(delta);
            f_omega = accl / L * std::tan(delta)
                    + v / (L * std::cos(delta) * std::cos(delta)) * sv;
            f_beta  = 0.0;
        } else {
            // Dynamic model
            f_x     = v * std::cos(beta + psi);
            f_y     = v * std::sin(beta + psi);
            f_delta = sv;
            f_v     = accl;
            f_psi   = omega;

            double v_safe = std::max(std::abs(v), 0.5);
            f_omega = (-vp_.mu * vp_.m / (v_safe * vp_.I * L)
                       * (vp_.lf * vp_.lf * vp_.C_Sf * (G * vp_.lr - accl * vp_.h)
                        + vp_.lr * vp_.lr * vp_.C_Sr * (G * vp_.lf + accl * vp_.h))
                       * omega
                       + vp_.mu * vp_.m / (vp_.I * L)
                       * (vp_.lr * vp_.C_Sr * (G * vp_.lf + accl * vp_.h)
                        - vp_.lf * vp_.C_Sf * (G * vp_.lr - accl * vp_.h))
                       * beta
                       + vp_.mu * vp_.m / (vp_.I * L)
                       * vp_.lf * vp_.C_Sf * (G * vp_.lr - accl * vp_.h)
                       * delta);

            f_beta = ((vp_.mu / (v_safe * v_safe * L)
                       * (vp_.C_Sr * (G * vp_.lf + accl * vp_.h) * vp_.lr
                        - vp_.C_Sf * (G * vp_.lr - accl * vp_.h) * vp_.lf)
                       - 1.0)
                      * omega
                      - vp_.mu / (v_safe * L)
                      * (vp_.C_Sr * (G * vp_.lf + accl * vp_.h)
                       + vp_.C_Sf * (G * vp_.lr - accl * vp_.h))
                      * beta
                      + vp_.mu / (v_safe * L)
                      * vp_.C_Sf * (G * vp_.lr - accl * vp_.h)
                      * delta);
        }

        // Euler integration
        double dt = cost_.dt;
        double x_n     = x     + dt * f_x;
        double y_n     = y     + dt * f_y;
        double delta_n = delta + dt * f_delta;
        double v_n     = v     + dt * f_v;
        double psi_n   = psi   + dt * f_psi;
        double omega_n = omega + dt * f_omega;
        double beta_n  = beta  + dt * f_beta;

        // Normalize heading to [0, 2*pi)
        psi_n = std::fmod(psi_n, TWO_PI);
        if (psi_n < 0.0) psi_n += TWO_PI;

        state_ = {x_n, y_n, delta_n, v_n, psi_n, omega_n, beta_n};

        int t = step_count_;
        step_count_++;

        // --- Cost computation ---

        float off_track = occ_map_.query(x_n, y_n);
        bool done_off_track = (off_track >= 0.5f);

        double v_ref_curv = cost_.v_ref;
        if (!v_ref_window_.empty()) {
            int idx = std::min(t, static_cast<int>(v_ref_window_.size()) - 1);
            v_ref_curv = v_ref_window_[idx];
        }

        double v_x = v_n * std::cos(beta_n);
        double v_y = v_n * std::sin(beta_n);
        double alpha_slip = -std::atan2(v_y, std::abs(v_x) + 1e-8);
        bool done_slip = (std::abs(alpha_slip) > cost_.alpha_threshold);

        double track_cost, speed_cost, slip_cost, speed_cap_cost_max, speed_cap_cost_min;

        if (cost_.normalize_reward) {
            // All components normalized to ~[0, 1]
            double v_max_sq = vp_.v_max * vp_.v_max;
            track_cost = static_cast<double>(off_track);                          // {0, 1}
            speed_cost = -(cost_.speed_weight * v_n * v_n / v_max_sq);            // [-sw, 0]
            slip_cost = (alpha_slip * alpha_slip)
                        / (cost_.alpha_threshold * cost_.alpha_threshold)
                      + (done_slip ? 1.0 : 0.0);                                 // [0, ~2]
            speed_cap_cost_max = (std::abs(v_n) > 12.0 ? 1.0 : 0.0);             // {0, 1}
            speed_cap_cost_min = (v_n < 1.5 ? 1.0 : 0.0);                        // {0, 1}
        } else {
            track_cost = static_cast<double>(off_track) * 1000000.0;
            speed_cost = -(cost_.speed_weight * v_n );
            slip_cost = alpha_slip * alpha_slip
                      + 10000.0 * (done_slip ? 1.0 : 0.0);
            speed_cap_cost_max = 10000.0 * (std::abs(v_n) > 12.0 ? 1.0 : 0.0);
            speed_cap_cost_min = 10000.0 * (v_n < 1.5 ? 1.0 : 0.0);
        }

        bool done = done_off_track;
        double reward = -(track_cost + speed_cost + slip_cost + speed_cap_cost_max );


        return {state_, reward, done};
    }

    std::vector<double> sample_action() override {
        double v = state_[3];
        // At low speed: uniform positive accel, GP prior for steering with 50% clamp
        if (v < 3.0) {
            std::uniform_real_distribution<double> accel_dist(0.0, vp_.a_max);
            double sv_limit = std::max(0.5, (v / 3.0)) * vp_.sv_max;
            double sv;
            if (use_action_prior_) {
                std::normal_distribution<double> steer_dist(
                    action_mean_[1], action_std_[1]);
                sv = std::max(-sv_limit, std::min(sv_limit, steer_dist(rng_)));
            } else {
                std::uniform_real_distribution<double> steer_dist(
                    -sv_limit, sv_limit);
                sv = steer_dist(rng_);
            }
            return {accel_dist(rng_), sv};
        }
        // Overspeed heuristic: if v > min(v_ref_window), only sample braking
        if (v > v_ref_min_) {
            std::uniform_real_distribution<double> accel_dist(-vp_.a_max, 0.0);
            double sv;
            if (use_action_prior_) {
                std::normal_distribution<double> steer_dist(
                    action_mean_[1], action_std_[1]);
                sv = std::max(vp_.sv_min,
                              std::min(vp_.sv_max, steer_dist(rng_)));
            } else {
                std::uniform_real_distribution<double> steer_dist(
                    vp_.sv_min, vp_.sv_max);
                sv = steer_dist(rng_);
            }
            return {accel_dist(rng_), sv};
        }
        if (use_action_prior_) {
            // Sample from Gaussian prior N(mean, std), clamp to action limits
            std::normal_distribution<double> accel_dist(action_mean_[0], action_std_[0]);
            std::normal_distribution<double> steer_dist(action_mean_[1], action_std_[1]);
            double accel = std::max(-vp_.a_max, std::min(vp_.a_max, accel_dist(rng_)));
            double sv = std::max(vp_.sv_min, std::min(vp_.sv_max, steer_dist(rng_)));
            return {accel, sv};
        }
        // Fallback: uniform sampling
        std::uniform_real_distribution<double> accel_dist(-vp_.a_max, vp_.a_max);
        std::uniform_real_distribution<double> steer_dist(vp_.sv_min, vp_.sv_max);
        return {accel_dist(rng_), steer_dist(rng_)};
    }

    std::unique_ptr<Simulator> clone() const override {
        auto copy = std::make_unique<F1TenthSimulator>(vp_, occ_map_, cost_);
        copy->state_ = state_;
        copy->step_count_ = step_count_;
        copy->v_ref_window_ = v_ref_window_;
        copy->v_ref_min_ = v_ref_min_;
        copy->action_mean_ = action_mean_;
        copy->action_std_ = action_std_;
        copy->use_action_prior_ = use_action_prior_;
        copy->rng_ = rng_;
        return copy;
    }
};
