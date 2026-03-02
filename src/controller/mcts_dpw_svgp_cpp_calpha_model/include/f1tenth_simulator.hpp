#pragma once

#include "simulator.hpp"
#include <cmath>
#include <random>
#include <algorithm>
#include <vector>
#include <memory>

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

// ---------------------------------------------------------------------------
// SVGP (Sparse Variational Gaussian Process) parameters for one GP.
// Holds all data needed for kernel evaluation and prediction.
// ---------------------------------------------------------------------------
struct SVGPParams {
    int M;          // number of inducing points (e.g. 50)
    int D_input;    // raw input dimension (6)
    int D_hidden;   // hidden/task dimension (2)
    int D_aug;      // D_input + D_hidden (8)

    std::vector<double> theta;    // (D_aug + 1): [lengthscale_0..D_aug-1, sigma]
    std::vector<double> alpha;    // (M): precomputed K_mm_inv @ mu_m, squeezed
    std::vector<double> Z_aug;    // (M * D_aug): augmented inducing points, row-major
    std::vector<double> xmean;    // (D_input): input scaler mean
    std::vector<double> xscale;   // (D_input): input scaler std
    double ymean;                 // output scaler mean (scalar)
    double yscale;                // output scaler std (scalar)
    std::vector<double> hp_mu;    // (D_hidden): task-specific hidden parameters

    // Construct from a flat double array packed in Python.
    // Layout: theta | alpha | Z_aug | xmean | xscale | ymean | yscale | hp_mu
    static SVGPParams from_flat(const double* data, int M_, int D_input_, int D_hidden_) {
        SVGPParams p;
        p.M = M_;
        p.D_input = D_input_;
        p.D_hidden = D_hidden_;
        p.D_aug = D_input_ + D_hidden_;

        int off = 0;
        p.theta.assign(data + off, data + off + p.D_aug + 1); off += p.D_aug + 1;
        p.alpha.assign(data + off, data + off + M_);           off += M_;
        p.Z_aug.assign(data + off, data + off + M_ * p.D_aug); off += M_ * p.D_aug;
        p.xmean.assign(data + off, data + off + D_input_);     off += D_input_;
        p.xscale.assign(data + off, data + off + D_input_);    off += D_input_;
        p.ymean = data[off++];
        p.yscale = data[off++];
        p.hp_mu.assign(data + off, data + off + D_hidden_);    off += D_hidden_;
        return p;
    }
};

// ---------------------------------------------------------------------------
// SVGP single-sample prediction: normalize → augment → RBF ARD kernel → predict → denormalize
// ---------------------------------------------------------------------------
inline double svgp_predict_single(
    const double* gp_input,   // (D_input) raw GP input: [delta, v, omega, beta, accl, sv]
    const SVGPParams& gp
) {
    // Stack-allocate augmented input (max D_aug = 8)
    double x_aug[8];

    // 1. Normalize raw inputs
    for (int d = 0; d < gp.D_input; ++d) {
        x_aug[d] = (gp_input[d] - gp.xmean[d]) / gp.xscale[d];
    }
    // 2. Append task-specific hidden parameters
    for (int d = 0; d < gp.D_hidden; ++d) {
        x_aug[gp.D_input + d] = gp.hp_mu[d];
    }

    // 3. RBF ARD kernel evaluation + dot product with alpha (fused)
    const double sigma = gp.theta[gp.D_aug];   // last element of theta
    const double sigma_sq = sigma * sigma;
    double f_mean = 0.0;

    for (int j = 0; j < gp.M; ++j) {
        double dist = 0.0;
        const int base = j * gp.D_aug;
        for (int d = 0; d < gp.D_aug; ++d) {
            double diff = (x_aug[d] - gp.Z_aug[base + d]) / gp.theta[d];
            dist += diff * diff;
        }
        double k_j = sigma_sq * std::exp(-0.5 * dist);
        f_mean += k_j * gp.alpha[j];
    }

    // 4. Denormalize
    return f_mean * gp.yscale + gp.ymean;
}

// ---------------------------------------------------------------------------
// F1TENTH simulator for MCTS rollouts.
// Uses Extended Kinematic (Ekin) base model + SVGP residual corrections
// for omega (yaw rate) and beta (slip angle).
// ---------------------------------------------------------------------------
class F1TenthSimulator : public Simulator {
  private:
    VehicleParamsCpp vp_;
    OccupancyMapData occ_map_;       // imaginary boundary (soft penalty, no done)
    OccupancyMapData occ_map_real_;  // real track boundary (hard penalty, done=true)
    CostParams cost_;

    // SVGP models (shared across clones for efficiency)
    std::shared_ptr<const SVGPParams> omega_svgp_;
    std::shared_ptr<const SVGPParams> beta_svgp_;
    double base_timestep_;
    int n_substeps_;

    // State: [x, y, delta, v, psi, omega, beta]
    std::vector<double> state_;
    int step_count_ = 0;
    std::vector<double> v_ref_window_;
    double v_ref_min_ = 1e9;  // min of v_ref_window, precomputed

    // Gaussian action prior from SVGP: [accel_mean, sv_mean], [accel_std, sv_std]
    std::vector<double> action_mean_;  // size 2
    std::vector<double> action_std_;   // size 2
    bool use_action_prior_ = false;

    // Centerline for heading lookup: cx_[i], cy_[i], cheading_[i]
    std::vector<double> cx_;
    std::vector<double> cy_;
    std::vector<double> cheading_;

    std::mt19937 rng_;

    static constexpr double TWO_PI = 2.0 * M_PI;

  public:
    F1TenthSimulator(const VehicleParamsCpp& vp,
                     const OccupancyMapData& occ_map,
                     const OccupancyMapData& occ_map_real,
                     const CostParams& cost,
                     std::shared_ptr<const SVGPParams> omega_svgp,
                     std::shared_ptr<const SVGPParams> beta_svgp,
                     double base_timestep,
                     unsigned seed = std::random_device{}())
        : vp_(vp), occ_map_(occ_map), occ_map_real_(occ_map_real), cost_(cost),
          omega_svgp_(std::move(omega_svgp)),
          beta_svgp_(std::move(beta_svgp)),
          base_timestep_(base_timestep),
          n_substeps_(std::max(1, static_cast<int>(std::round(cost.dt / base_timestep)))),
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

    void set_centerline(const std::vector<double>& cx,
                        const std::vector<double>& cy,
                        const std::vector<double>& heading) {
        cx_ = cx;
        cy_ = cy;
        cheading_ = heading;
    }

    // Returns index of nearest centerline point (used for backward-progress check)
    int nearest_index(double x, double y) const {
        double best_d2 = 1e18;
        int best_idx = 0;
        int n = static_cast<int>(cx_.size());
        for (int i = 0; i < n; i++) {
            double dx = x - cx_[i];
            double dy = y - cy_[i];
            double d2 = dx * dx + dy * dy;
            if (d2 < best_d2) {
                best_d2 = d2;
                best_idx = i;
            }
        }
        return best_idx;
    }

    double nearest_heading(double x, double y) const {
        return cheading_[nearest_index(x, y)];
    }

    StepResult step(const std::vector<double>& action) override {
        double accl = action[0];
        double sv   = action[1];

        // --- Constrain actions ONCE (using initial state) ---
        double delta_init = state_[2];
        double v_init     = state_[3];

        // Steering constraint
        if ((delta_init <= vp_.s_min && sv <= 0.0) ||
            (delta_init >= vp_.s_max && sv >= 0.0)) {
            sv = 0.0;
        }
        sv = std::max(vp_.sv_min, std::min(vp_.sv_max, sv));

        // Acceleration constraint
        double pos_limit;
        if (v_init > vp_.v_switch) {
            pos_limit = vp_.a_max * vp_.v_switch / (v_init + 1e-8);
        } else {
            pos_limit = vp_.a_max;
        }
        if ((v_init <= vp_.v_min && accl <= 0.0) ||
            (v_init >= vp_.v_max && accl >= 0.0)) {
            accl = 0.0;
        }
        accl = std::max(-vp_.a_max, std::min(pos_limit, accl));

        // --- Substep loop: Ekin base + SVGP residuals ---
        const double L = vp_.lf + vp_.lr;
        const double sub_dt = (n_substeps_ > 1) ? base_timestep_ : cost_.dt;

        for (int sub = 0; sub < n_substeps_; ++sub) {
            double x     = state_[0];
            double y     = state_[1];
            double delta = state_[2];
            double v     = state_[3];
            double psi   = state_[4];
            double omega = state_[5];
            double beta  = state_[6];

            // Extended Kinematic model derivatives
            double x_dot     = v * std::cos(beta + psi);
            double y_dot     = v * std::sin(beta + psi);
            double v_dot     = accl;
            double psi_dot   = omega;
            double omega_dot = (sv * v + delta * accl) / L;
            double beta_dot  = (vp_.lf * (sv * v + delta * accl)) / L;

            // Euler integration (Ekin baseline)
            double next_x     = x     + sub_dt * x_dot;
            double next_y     = y     + sub_dt * y_dot;
            double next_delta = delta + sub_dt * sv;
            double next_v     = v     + sub_dt * v_dot;
            double next_psi   = psi   + sub_dt * psi_dot;
            double next_omega = omega + sub_dt * omega_dot;
            double next_beta  = beta  + sub_dt * beta_dot;

            // SVGP residual corrections (mean only)
            double gp_input[6] = {delta, v, omega, beta, accl, sv};
            double err_omega = svgp_predict_single(gp_input, *omega_svgp_);
            double err_beta  = svgp_predict_single(gp_input, *beta_svgp_);
            next_omega += err_omega;
            next_beta  += err_beta;

            state_ = {next_x, next_y, next_delta, next_v, next_psi, next_omega, next_beta};
        }

        // Normalize heading to [0, 2*pi)
        state_[4] = std::fmod(state_[4], TWO_PI);
        if (state_[4] < 0.0) state_[4] += TWO_PI;

        int t = step_count_;
        step_count_++;

        // --- Cost computation ---

        double x_n     = state_[0];
        double y_n     = state_[1];
        double v_n     = state_[3];
        double psi_n   = state_[4];
        double beta_n  = state_[6];

        float off_imag = occ_map_.query(x_n, y_n);       // imaginary boundary
        float off_real = occ_map_real_.query(x_n, y_n);   // real wall
        bool done_off_track = (off_real >= 0.5f);

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
            double v_max_sq = vp_.v_max * vp_.v_max;
            double real_cost = static_cast<double>(off_real);
            track_cost = real_cost;
            speed_cost = -(cost_.speed_weight * v_n * v_n / v_max_sq);
            slip_cost = (alpha_slip * alpha_slip)
                        / (cost_.alpha_threshold * cost_.alpha_threshold)
                      + (done_slip ? 1.0 : 0.0);
            speed_cap_cost_max = (std::abs(v_n) > 12.0 ? 1.0 : 0.0);
            speed_cap_cost_min = (v_n < 1.5 ? 1.0 : 0.0);
        } else {
            double real_cost = static_cast<double>(off_real) * 1000000.0;
            track_cost = real_cost;
            speed_cost = -(cost_.speed_weight * v_n);
            slip_cost = alpha_slip * alpha_slip
                      + 10000.0 * (done_slip ? 1.0 : 0.0);
            speed_cap_cost_max = 10000.0 * (std::abs(v_n) > 12.0 ? 1.0 : 0.0);
            speed_cap_cost_min = 10000.0 * (v_n < 1.5 ? 1.0 : 0.0);
        }

        // Facing-backward check: if car heading differs from track by > 90°, terminate.
        bool done_backward = false;
        if (!cx_.empty()) {
            double ref_heading = nearest_heading(x_n, y_n);
            double hdiff = std::abs(psi_n - ref_heading);
            if (hdiff > M_PI) hdiff = TWO_PI - hdiff;
            if (hdiff > M_PI / 2.0) {
                done_backward = true;
                track_cost += cost_.normalize_reward ? 1.0 : 1000000.0;
            }
        }

        bool done = done_off_track || done_backward;
        double reward = -(track_cost + speed_cost + slip_cost + speed_cap_cost_max);

        return {state_, reward, done};
    }

    std::vector<double> sample_action() override {
        double v = state_[3];
        // At low speed: uniform positive accel, GP prior for steering with 50% clamp
        if (v < 2.0) {
            std::uniform_real_distribution<double> accel_dist(0.0, vp_.a_max);
            double sv_limit = std::max(0.5, (v / 2.0)) * vp_.sv_max;
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
        // SVGP params are shared (read-only) — cheap shared_ptr copy
        auto copy = std::make_unique<F1TenthSimulator>(
            vp_, occ_map_, occ_map_real_, cost_,
            omega_svgp_, beta_svgp_, base_timestep_);
        copy->state_ = state_;
        copy->step_count_ = step_count_;
        copy->v_ref_window_ = v_ref_window_;
        copy->v_ref_min_ = v_ref_min_;
        copy->action_mean_ = action_mean_;
        copy->action_std_ = action_std_;
        copy->use_action_prior_ = use_action_prior_;
        copy->cx_ = cx_;
        copy->cy_ = cy_;
        copy->cheading_ = cheading_;
        copy->rng_ = rng_;
        return copy;
    }
};
