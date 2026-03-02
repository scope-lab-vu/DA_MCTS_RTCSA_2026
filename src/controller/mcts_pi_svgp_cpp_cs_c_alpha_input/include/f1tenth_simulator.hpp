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
// PI-SVGP parameters: multi-output GP predicting 9 Pacejka tire parameters.
// ---------------------------------------------------------------------------
struct PiSVGPParams {
    int M;            // number of inducing points (50)
    int D_input;      // raw input dimension (6)
    int D_hidden;     // hidden/task dimension (2)
    int D_aug;        // D_input + D_hidden (8)
    int num_outputs;  // number of GP outputs (9 Pacejka params)

    std::vector<double> theta;        // (D_aug + 1): kernel hyperparams
    std::vector<double> alpha;        // (M * num_outputs): K_mm_inv @ m, row-major
    std::vector<double> Z_aug;        // (M * D_aug): augmented inducing points
    std::vector<double> xmean;        // (D_input): input scaler mean
    std::vector<double> xscale;       // (D_input): input scaler std
    std::vector<double> hp_mu;        // (D_hidden): task-specific latent params
    std::vector<double> output_bias;  // (num_outputs): learnable output bias
    std::vector<double> K_mm_inv;     // (M * M): inverse kernel matrix, row-major
    std::vector<double> S_diag;       // (M * num_outputs): variational posterior variance
                                      //   S_diag[j * num_outputs + p] = exp(2 * m_logstd[j, p])

    // Construct from flat array packed in Python.
    // Layout: theta | alpha | Z_aug | xmean | xscale | hp_mu | output_bias | K_mm_inv | S_diag
    static PiSVGPParams from_flat(const double* data, int M_, int D_input_,
                                   int D_hidden_, int num_outputs_) {
        PiSVGPParams p;
        p.M = M_;
        p.D_input = D_input_;
        p.D_hidden = D_hidden_;
        p.D_aug = D_input_ + D_hidden_;
        p.num_outputs = num_outputs_;

        int off = 0;
        p.theta.assign(data + off, data + off + p.D_aug + 1);          off += p.D_aug + 1;
        p.alpha.assign(data + off, data + off + M_ * num_outputs_);    off += M_ * num_outputs_;
        p.Z_aug.assign(data + off, data + off + M_ * p.D_aug);        off += M_ * p.D_aug;
        p.xmean.assign(data + off, data + off + D_input_);            off += D_input_;
        p.xscale.assign(data + off, data + off + D_input_);           off += D_input_;
        p.hp_mu.assign(data + off, data + off + D_hidden_);           off += D_hidden_;
        p.output_bias.assign(data + off, data + off + num_outputs_);   off += num_outputs_;
        p.K_mm_inv.assign(data + off, data + off + M_ * M_);          off += M_ * M_;
        p.S_diag.assign(data + off, data + off + M_ * num_outputs_);  off += M_ * num_outputs_;
        return p;
    }
};

// ---------------------------------------------------------------------------
// Softplus: log(1 + exp(x))
// ---------------------------------------------------------------------------
inline double softplus(double x) {
    if (x > 20.0) return x;  // numerical stability
    return std::log(1.0 + std::exp(x));
}

// Pacejka positive-constraint indices: Bf=0, Cf=1, Df=2, Br=4, Cr=5, Dr=6
inline bool is_positive_pacejka(int idx) {
    return idx == 0 || idx == 1 || idx == 2 ||
           idx == 4 || idx == 5 || idx == 6;
}

// ---------------------------------------------------------------------------
// PI-SVGP multi-output prediction with posterior sampling:
//   normalize -> augment -> RBF ARD kernel -> mean_p, var_p -> sample ~ N(mean_p, var_p)
//
//   mean_p  = k*^T @ alpha_p                  (alpha = K_mm_inv @ m)
//   var_p   = sigma^2 - k*^T @ K_mm_inv @ k*
//             + k*^T @ K_mm_inv @ S_p @ K_mm_inv @ k*
//   sample  ~ N(mean_p, var_p)
// ---------------------------------------------------------------------------
inline void pi_svgp_predict(
    const double* gp_input,   // (D_input=6): [VX, VY, YAW_RATE, STEERING_FB, THROTTLE_CMD, STEERING_CMD]
    const PiSVGPParams& gp,
    double* pacejka_out,      // (num_outputs=9): sampled Pacejka parameters
    std::mt19937& rng         // RNG for posterior sampling
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

    // 3. Initialize raw GP mean to zero
    for (int p = 0; p < gp.num_outputs; ++p) {
        pacejka_out[p] = 0.0;
    }

    // 4. Compute k_star vector and accumulate mean (fused)
    const double sigma = gp.theta[gp.D_aug];
    const double sigma_sq = sigma * sigma;

    double k_star[50];  // stack-allocate (max M = 50)

    for (int j = 0; j < gp.M; ++j) {
        double dist = 0.0;
        const int z_base = j * gp.D_aug;
        for (int d = 0; d < gp.D_aug; ++d) {
            double diff = (x_aug[d] - gp.Z_aug[z_base + d]) / gp.theta[d];
            dist += diff * diff;
        }
        k_star[j] = sigma_sq * std::exp(-0.5 * dist);

        // Accumulate mean: k_star[j] * alpha[j, p]
        const int a_base = j * gp.num_outputs;
        for (int p = 0; p < gp.num_outputs; ++p) {
            pacejka_out[p] += k_star[j] * gp.alpha[a_base + p];
        }
    }

    // 5. Compute v = K_mm_inv @ k_star  (M-vector)
    double v[50];
    for (int i = 0; i < gp.M; ++i) {
        double sum = 0.0;
        const int row_base = i * gp.M;
        for (int j = 0; j < gp.M; ++j) {
            sum += gp.K_mm_inv[row_base + j] * k_star[j];
        }
        v[i] = sum;
    }

    // 6. Prior variance - explained variance (shared across outputs)
    //    var_base = sigma^2 - k*^T @ v
    double k_dot_v = 0.0;
    for (int j = 0; j < gp.M; ++j) {
        k_dot_v += k_star[j] * v[j];
    }
    double var_base = sigma_sq - k_dot_v;

    // 7. Per-output: compute var_p, then sample from N(mean_p, var_p)
    //    var_S_p = sum_j v[j]^2 * S_diag[j, p]   (diagonal posterior correction)
    //    var_p   = var_base + var_S_p
    for (int p = 0; p < gp.num_outputs; ++p) {
        double var_S = 0.0;
        for (int j = 0; j < gp.M; ++j) {
            var_S += v[j] * v[j] * gp.S_diag[j * gp.num_outputs + p];
        }
        double var_p = var_base + var_S;
        if (var_p < 1e-12) var_p = 1e-12;  // clamp for numerics

        double mean_p = pacejka_out[p];
        double std_p  = std::sqrt(var_p);
        std::normal_distribution<double> dist(mean_p, std_p);
        pacejka_out[p] = dist(rng);
    }

    // 8. Add output bias and apply Pacejka constraints (softplus on B, C, D)
    for (int p = 0; p < gp.num_outputs; ++p) {
        pacejka_out[p] += gp.output_bias[p];
        if (is_positive_pacejka(p)) {
            pacejka_out[p] = softplus(pacejka_out[p]);
        }
    }
}


// ---------------------------------------------------------------------------
// F1TENTH simulator for MCTS rollouts.
// Uses PI-SVGP: multi-output GP -> Pacejka tire physics.
// Includes dual occupancy maps, heading-backward check, GP action prior.
// ---------------------------------------------------------------------------
class F1TenthSimulator : public Simulator {
  private:
    VehicleParamsCpp vp_;
    OccupancyMapData occ_map_;       // imaginary boundary (soft penalty, no done)
    OccupancyMapData occ_map_real_;  // real track boundary (hard penalty, done=true)
    CostParams cost_;

    // PI-SVGP model (shared across clones — hp_mu is mutable for latent updates)
    std::shared_ptr<PiSVGPParams> pi_svgp_;
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
    // C_alpha (cornering stiffness) — passed directly as 7th GP input
    double c_alpha_value_ = 104.89;  // nominal high-friction value
    std::mt19937 rng_;

    static constexpr double TWO_PI = 2.0 * M_PI;

  public:
    F1TenthSimulator(const VehicleParamsCpp& vp,
                     const OccupancyMapData& occ_map,
                     const OccupancyMapData& occ_map_real,
                     const CostParams& cost,
                     std::shared_ptr<PiSVGPParams> pi_svgp,
                     double base_timestep,
                     unsigned seed = std::random_device{}())
        : vp_(vp), occ_map_(occ_map), occ_map_real_(occ_map_real), cost_(cost),
          pi_svgp_(std::move(pi_svgp)),
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

    void set_hp_mu(const std::vector<double>& new_hp_mu) {
        pi_svgp_->hp_mu = new_hp_mu;
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

    void set_c_alpha(double c_alpha) { c_alpha_value_ = c_alpha; }

    // Returns index of nearest centerline point
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

        // --- Substep loop: GP -> Pacejka physics -> kinematic update ---
        const double sub_dt = (n_substeps_ > 1) ? base_timestep_ : cost_.dt;

        for (int sub = 0; sub < n_substeps_; ++sub) {
            double x     = state_[0];
            double y     = state_[1];
            double delta = state_[2];
            double v     = state_[3];
            double psi   = state_[4];
            double omega = state_[5];
            double beta  = state_[6];

            // 1. Extract body-frame features for GP
            double VX          = v * std::cos(beta);
            double VY          = v * std::sin(beta);
            double YAW_RATE    = omega;
            double STEERING_FB = delta;
            double THROTTLE_CMD = accl * base_timestep_;
            double STEERING_CMD = sv * base_timestep_;

            double gp_input[7] = {VX, VY, YAW_RATE, STEERING_FB, THROTTLE_CMD, STEERING_CMD, c_alpha_value_};

            // 2. GP predict -> 9 constrained Pacejka parameters (sampled from posterior)
            double pacejka[9];
            pi_svgp_predict(gp_input, *pi_svgp_, pacejka, rng_);

            // Unpack Pacejka parameters
            double Bf  = pacejka[0];
            double Cf  = pacejka[1];
            double Df  = pacejka[2];
            double Ef  = pacejka[3];
            double Br  = pacejka[4];
            double Cr  = pacejka[5];
            double Dr  = pacejka[6];
            double Er  = pacejka[7];
            double Frx = pacejka[8];

            // 3. Pacejka tire physics
            double steering = STEERING_FB + STEERING_CMD;

            // Slip angles
            double alphaf = steering - std::atan2(vp_.lf * YAW_RATE + VY, std::abs(VX) + 1e-8);
            double alphar = std::atan2(vp_.lr * YAW_RATE - VY, std::abs(VX) + 1e-8);

            // Tire forces
            double Bf_alphaf = Bf * alphaf;
            double Br_alphar = Br * alphar;
            double Ffy = Df * std::sin(Cf * std::atan(Bf_alphaf - Ef * (Bf_alphaf - std::atan(Bf_alphaf))));
            double Fry = Dr * std::sin(Cr * std::atan(Br_alphar - Er * (Br_alphar - std::atan(Br_alphar))));

            // Vehicle dynamics derivatives
            double dVX       = (1.0 / vp_.m) * (Frx - Ffy * std::sin(steering)) + VY * YAW_RATE;
            double dVY       = (1.0 / vp_.m) * (Fry + Ffy * std::cos(steering)) - VX * YAW_RATE;
            double dYAW_RATE = (1.0 / vp_.I) * (Ffy * vp_.lf * std::cos(steering) - Fry * vp_.lr);

            // Euler integration -> next body-frame state
            double next_VX       = VX       + dVX * sub_dt;
            double next_VY       = VY       + dVY * sub_dt;
            double next_YAW_RATE = YAW_RATE + dYAW_RATE * sub_dt;

            // 4. Map back to vehicle state
            double omega_pred = next_YAW_RATE;
            double beta_pred  = std::atan2(next_VY, next_VX + 1e-8);

            // Kinematic updates
            double next_x     = x     + sub_dt * v * std::cos(beta + psi);
            double next_y     = y     + sub_dt * v * std::sin(beta + psi);
            double next_delta = delta + sub_dt * sv;
            double next_v     = v     + sub_dt * accl;
            double next_psi   = psi   + sub_dt * omega;

            state_ = {next_x, next_y, next_delta, next_v, next_psi, omega_pred, beta_pred};
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

        // Dual occupancy map queries
        float off_imag = occ_map_.query(x_n, y_n);       // imaginary boundary
        float off_real = occ_map_real_.query(x_n, y_n);   // real wall
        bool done_off_track = (off_real >= 0.5f);

        // Speed cost: v_ref from window
        double v_ref_t;
        if (!v_ref_window_.empty()) {
            int idx = std::min(t, static_cast<int>(v_ref_window_.size()) - 1);
            v_ref_t = v_ref_window_[idx];
        } else {
            v_ref_t = cost_.v_ref;
        }

        // Slip angle
        double v_x = v_n * std::cos(beta_n);
        double v_y = v_n * std::sin(beta_n);
        double alpha_slip = -std::atan2(v_y, std::abs(v_x) + 1e-8);
        bool done_slip = (std::abs(alpha_slip) > cost_.alpha_threshold);

        double track_cost, speed_cost, slip_cost, speed_cap_cost_max, speed_cap_cost_min;

        if (cost_.normalize_reward) {
            // All components normalized to ~[0, 1]
            double v_max_sq = vp_.v_max * vp_.v_max;
            // double imag_cost = static_cast<double>(off_imag);                     // {0, 1}
            double real_cost = static_cast<double>(off_real);                     // {0, 1}
            track_cost = real_cost;
            speed_cost = -(cost_.speed_weight * v_n * v_n / v_max_sq);            // [-sw, 0]
            slip_cost = (alpha_slip * alpha_slip)
                        / (cost_.alpha_threshold * cost_.alpha_threshold)
                      + (done_slip ? 1.0 : 0.0);                                 // [0, ~2]
            speed_cap_cost_max = (std::abs(v_n) > 12.0 ? 1.0 : 0.0);              // {0, 1}
            speed_cap_cost_min = (v_n < 1.5 ? 1.0 : 0.0);                        // {0, 1}
        } else {
            // double imag_cost = static_cast<double>(off_imag) * 10000.0;    // soft penalty
            double real_cost = static_cast<double>(off_real) * 1000000.0;  // hard penalty
            track_cost = real_cost;
            speed_cost = -(cost_.speed_weight * v_n );
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
        // PI-SVGP params are shared — all clones see latent updates via set_hp_mu()
        auto copy = std::make_unique<F1TenthSimulator>(
            vp_, occ_map_, occ_map_real_, cost_, pi_svgp_, base_timestep_);
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
        copy->c_alpha_value_ = c_alpha_value_;
        copy->rng_ = rng_;
        return copy;
    }
};
