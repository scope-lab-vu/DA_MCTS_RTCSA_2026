#pragma once

#include "simulator.hpp"
#include <cmath>
#include <cstdint>
#include <random>
#include <algorithm>
#include <vector>
#include <memory>

// Fast exp() using single-precision expf. 2-3x faster than std::exp (double)
// via SSE, with ~7 digits of precision — more than sufficient for GP kernels.
inline double fast_exp(double x) {
    return static_cast<double>(expf(static_cast<float>(x)));
}

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
// Krasser SVGP parameters for one GP (single-output, ARD RBF kernel).
// Uses C_alpha as direct 7th input feature — no hidden/task parameters.
// ARD kernel: k(x,x') = sigma_f^2 * exp(-0.5 * sum_d((x_d - x'_d)^2 / l_d^2))
// ---------------------------------------------------------------------------
struct KrasserSVGPParams {
    int M;          // number of inducing points (e.g. 53)
    int D_input;    // input dimension (7: delta, v, omega, beta, accl, sv, c_alpha)

    std::vector<double> lengthscales;  // (D_input): per-dimension ARD lengthscales
    double sigma_f;                    // signal standard deviation

    std::vector<double> alpha;    // (M): precomputed K_mm_inv @ mu_m, squeezed
    std::vector<double> X_m;     // (M * D_input): inducing points in normalized space, row-major
    std::vector<double> xmean;    // (D_input): input scaler mean
    std::vector<double> xscale;   // (D_input): input scaler std
    double ymean;                 // output scaler mean (scalar)
    double yscale;                // output scaler std (scalar)

    // --- Precomputed for fast prediction ---
    std::vector<double> inv_xscale_times_inv_l;  // (D_input): 1/(xscale[d]*lengthscales[d])
    std::vector<double> xmean_scaled;             // (D_input): xmean[d] * inv_xscale_times_inv_l[d]
    std::vector<double> X_m_scaled;  // (M*D_input): X_m[j*D+d] / lengthscales[d]
    std::vector<double> alpha_scaled; // (M): sigma_f^2 * alpha[j] * yscale

    // Call after from_flat() to precompute fast-prediction arrays.
    void precompute() {
        double sigma_f_sq = sigma_f * sigma_f;

        inv_xscale_times_inv_l.resize(D_input);
        xmean_scaled.resize(D_input);
        for (int d = 0; d < D_input; ++d) {
            inv_xscale_times_inv_l[d] = 1.0 / (xscale[d] * lengthscales[d]);
            xmean_scaled[d] = xmean[d] * inv_xscale_times_inv_l[d];
        }

        X_m_scaled.resize(M * D_input);
        for (int j = 0; j < M; ++j) {
            const int base = j * D_input;
            for (int d = 0; d < D_input; ++d) {
                X_m_scaled[base + d] = X_m[base + d] / lengthscales[d];
            }
        }

        alpha_scaled.resize(M);
        for (int j = 0; j < M; ++j) {
            alpha_scaled[j] = sigma_f_sq * alpha[j] * yscale;
        }
    }

    // Construct from a flat double array packed in Python.
    // Layout: lengthscales(D_input) | sigma_f | alpha(M) | X_m(M*D_input) |
    //         xmean(D_input) | xscale(D_input) | ymean | yscale
    static KrasserSVGPParams from_flat(const double* data, int M_, int D_input_) {
        KrasserSVGPParams p;
        p.M = M_;
        p.D_input = D_input_;

        int off = 0;
        p.lengthscales.assign(data + off, data + off + D_input_); off += D_input_;
        p.sigma_f     = data[off++];
        p.alpha.assign(data + off, data + off + M_);           off += M_;
        p.X_m.assign(data + off, data + off + M_ * D_input_);  off += M_ * D_input_;
        p.xmean.assign(data + off, data + off + D_input_);     off += D_input_;
        p.xscale.assign(data + off, data + off + D_input_);    off += D_input_;
        p.ymean = data[off++];
        p.yscale = data[off++];
        p.precompute();
        return p;
    }
};

// ---------------------------------------------------------------------------
// Krasser SVGP single-point prediction:
//   normalize → ARD RBF kernel → predict → denormalize
// ---------------------------------------------------------------------------
inline double krasser_predict_single(
    const double* gp_input,   // (D_input=7) raw: [delta, v, omega, beta, accl, sv, c_alpha]
    const KrasserSVGPParams& gp
) {
    // 1. Normalize + scale by inverse lengthscale in one pass (no divisions)
    //    x_scaled[d] = (input[d] - xmean[d]) / (xscale[d] * lengthscales[d])
    //                = input[d] * inv_xscale_times_inv_l[d] - xmean_scaled[d]
    double x_scaled[7];
    for (int d = 0; d < gp.D_input; ++d) {
        x_scaled[d] = gp_input[d] * gp.inv_xscale_times_inv_l[d] - gp.xmean_scaled[d];
    }

    // 2. ARD RBF kernel + dot product with alpha_scaled (fused)
    //    alpha_scaled[j] = sigma_f^2 * alpha[j] * yscale  (absorbs denorm)
    //    X_m_scaled[j*D+d] = X_m[j*D+d] / lengthscales[d]
    double f_mean = 0.0;

    for (int j = 0; j < gp.M; ++j) {
        double sqdist = 0.0;
        const int base = j * gp.D_input;
        for (int d = 0; d < gp.D_input; ++d) {
            double diff = x_scaled[d] - gp.X_m_scaled[base + d];
            sqdist += diff * diff;
        }
        f_mean += gp.alpha_scaled[j] * fast_exp(-0.5 * sqdist);
    }

    // 3. Denormalize (yscale already absorbed into alpha_scaled)
    return f_mean + gp.ymean;
}

// ---------------------------------------------------------------------------
// F1TENTH simulator for MCTS rollouts.
// Uses Extended Kinematic (Ekin) base model + Krasser SVGP residual corrections
// for omega (yaw rate) and beta (slip angle).
// C_alpha is passed as a direct 7th input feature to the SVGP models.
// ---------------------------------------------------------------------------
class F1TenthSimulator : public Simulator {
  private:
    VehicleParamsCpp vp_;
    OccupancyMapData occ_map_;       // imaginary boundary (soft penalty, no done)
    OccupancyMapData occ_map_real_;  // real track boundary (hard penalty, done=true)
    CostParams cost_;

    // Krasser SVGP models (shared across clones for efficiency)
    std::shared_ptr<const KrasserSVGPParams> omega_svgp_;
    std::shared_ptr<const KrasserSVGPParams> beta_svgp_;
    double base_timestep_;
    int n_substeps_;

    // State: [x, y, delta, v, psi, omega, beta]
    State7 state_;
    int step_count_ = 0;
    std::vector<double> v_ref_window_;
    double v_ref_min_ = 1e9;  // min of v_ref_window, precomputed

    // Gaussian action prior from SVGP: [accel_mean, sv_mean], [accel_std, sv_std]
    Action2 action_mean_{};
    Action2 action_std_{};
    bool use_action_prior_ = false;
    bool is_curve_ = false;

    // Centerline for heading lookup: cx_[i], cy_[i], cheading_[i]
    std::vector<double> cx_;
    std::vector<double> cy_;
    std::vector<double> cheading_;

    // C_alpha (cornering stiffness) — passed as 7th GP input
    double c_alpha_value_ = 104.89;

    std::mt19937 rng_;

    static constexpr double TWO_PI = 2.0 * M_PI;

  public:
    F1TenthSimulator(const VehicleParamsCpp& vp,
                     const OccupancyMapData& occ_map,
                     const OccupancyMapData& occ_map_real,
                     const CostParams& cost,
                     std::shared_ptr<const KrasserSVGPParams> omega_svgp,
                     std::shared_ptr<const KrasserSVGPParams> beta_svgp,
                     double base_timestep,
                     unsigned seed = std::random_device{}())
        : vp_(vp), occ_map_(occ_map), occ_map_real_(occ_map_real), cost_(cost),
          omega_svgp_(std::move(omega_svgp)),
          beta_svgp_(std::move(beta_svgp)),
          base_timestep_(base_timestep),
          n_substeps_(std::max(1, static_cast<int>(std::round(cost.dt / base_timestep)))),
          state_{}, rng_(seed) {}

    // --- Simulator interface ---

    bool is_action_discrete() const override { return false; }
    bool is_state_discrete() const override { return false; }
    int rollout_horizon() const override { return cost_.rollout_horizon; }
    double discount() const override { return cost_.discount_factor; }

    void set_state(const State7& state) override {
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
        if (mean.size() == 2 && std.size() == 2) {
            action_mean_[0] = mean[0]; action_mean_[1] = mean[1];
            action_std_[0] = std[0];   action_std_[1] = std[1];
            use_action_prior_ = true;
        } else {
            use_action_prior_ = false;
        }
    }

    void set_rollout_horizon(int h) override { cost_.rollout_horizon = h; }

    void set_is_curve(bool is_curve) { is_curve_ = is_curve; }

    void set_centerline(const std::vector<double>& cx,
                        const std::vector<double>& cy,
                        const std::vector<double>& heading) {
        cx_ = cx;
        cy_ = cy;
        cheading_ = heading;
    }

    void set_c_alpha(double c_alpha) { c_alpha_value_ = c_alpha; }

    // Returns index of nearest centerline point (used for backward-progress check)
    int nearest_index(double x, double y) const {
        double best_d2 = 1e18;
        int best_idx = 0;
        int n = static_cast<int>(cx_.size());
        for (int i = 0; i < n; i++) {
            double dx = x - cx_[i];
            double dy = y - cy_[i];
            double d2 = dx * dx + dy * dy;
            if (d2 < best_d2) { best_d2 = d2; best_idx = i; }
        }
        return best_idx;
    }

    double nearest_heading(double x, double y) const {
        return cheading_[nearest_index(x, y)];
    }

    StepResult step(const Action2& action) override {
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

        // --- Substep loop: Ekin base + Krasser SVGP residuals ---
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

            // Krasser SVGP residual corrections (mean only)
            // 7D input: [delta, v, omega, beta, accl, sv, c_alpha]
            double gp_input[7] = {delta, v, omega, beta, accl, sv, c_alpha_value_};
            double err_omega = krasser_predict_single(gp_input, *omega_svgp_);
            double err_beta  = krasser_predict_single(gp_input, *beta_svgp_);
            next_omega += err_omega;
            next_beta  += err_beta;

            state_[0] = next_x;     state_[1] = next_y;      state_[2] = next_delta;
            state_[3] = next_v;     state_[4] = next_psi;    state_[5] = next_omega;
            state_[6] = next_beta;
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
            speed_cap_cost_max = (std::abs(v_n) > 10.0 ? 1.0 : 0.0);
            speed_cap_cost_min = (v_n < 1.5 ? 1.0 : 0.0);
        } else {
            double imag_cost = static_cast<double>(off_imag) * 100000.0;    // soft penalty
            double real_cost = static_cast<double>(off_real) * 1000000.0;  // hard penalty
            track_cost = real_cost + imag_cost;
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

    Action2 sample_action() override {
        double v = state_[3];
        // At low speed: uniform positive accel, GP prior for steering with 50% clamp
        if (v < 2.5) {
            std::uniform_real_distribution<double> accel_dist(0.0, vp_.a_max);
            double sv_limit = std::max(0.5, (v / 2.5)) * vp_.sv_max;
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
            return Action2{accel_dist(rng_), sv};
        }
        // Steering: GP prior when available, uniform fallback
        double sv;
        if (use_action_prior_) {
            std::normal_distribution<double> steer_dist(
                action_mean_[1], action_std_[1]);
            sv = std::max(vp_.sv_min, std::min(vp_.sv_max, steer_dist(rng_)));
        } else {
            std::uniform_real_distribution<double> steer_dist(
                vp_.sv_min, vp_.sv_max);
            sv = steer_dist(rng_);
        }

        // Acceleration:
        //   Straight: 100% GP prior
        //   Curve:    33% Aggressive + 33% Slow + 34% Feasible (GP prior)
        double accel;
        if (!is_curve_) {
            // Straight: GP prior only
            if (use_action_prior_) {
                std::normal_distribution<double> accel_dist(
                    action_mean_[0], action_std_[0]);
                accel = std::max(-vp_.a_max, std::min(vp_.a_max, accel_dist(rng_)));
            } else {
                std::uniform_real_distribution<double> accel_dist(-vp_.a_max, vp_.a_max);
                accel = accel_dist(rng_);
            }
        } else {
            // Curve: equal three-way split
            std::uniform_real_distribution<double> selector(0.0, 1.0);
            double roll = selector(rng_);
            if (roll < 0.35) {
                // Aggressive: Uniform(0, a_max)
                std::uniform_real_distribution<double> accel_dist(0.0, vp_.a_max);
                accel = accel_dist(rng_);
            } else if (roll < 0.45) {
                // Slow: Uniform(-a_max, 0)
                std::uniform_real_distribution<double> accel_dist(-vp_.a_max, 0.0);
                accel = accel_dist(rng_);
            } else {
                // Feasible: GP prior
                if (use_action_prior_) {
                    std::normal_distribution<double> accel_dist(
                        action_mean_[0], action_std_[0]);
                    accel = std::max(-vp_.a_max, std::min(vp_.a_max, accel_dist(rng_)));
                } else {
                    std::uniform_real_distribution<double> accel_dist(-vp_.a_max, vp_.a_max);
                    accel = accel_dist(rng_);
                }
            }
        }

        return Action2{accel, sv};
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
        copy->is_curve_ = is_curve_;
        copy->cx_ = cx_;
        copy->cy_ = cy_;
        copy->cheading_ = cheading_;
        copy->c_alpha_value_ = c_alpha_value_;
        copy->rng_ = rng_;
        return copy;
    }
};
