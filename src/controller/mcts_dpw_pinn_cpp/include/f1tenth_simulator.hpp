#pragma once

#include "simulator.hpp"
#include <torch/script.h>
#include <cmath>
#include <random>
#include <algorithm>
#include <vector>
#include <string>
#include <utility>

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
};

// PINN configuration
struct PINNConfig {
    int horizon      = 10;    // Feature history window length
    int num_features = 6;     // [vx, vy, yaw_rate, steering_fb, throttle_cmd, steering_cmd]
    double base_timestep = 0.02;  // PINN was trained on this timestep
    int gru_num_layers   = 1;
    int gru_hidden_size  = 10;    // = horizon (as set in the model config)
};

// F1TENTH simulator for MCTS rollouts, using PINN dynamics.
// Replaces the analytical single-track model with a traced TorchScript PINN.
// Keeps the same cost function as the original.
class F1TenthSimulator : public Simulator {
  private:
    VehicleParamsCpp vp_;
    OccupancyMapData occ_map_;
    CostParams cost_;
    PINNConfig pinn_cfg_;

    // PINN model (TorchScript, CPU)
    torch::jit::Module pinn_model_;

    // Scaler for feature normalization: z = (x - mean) / scale
    std::vector<float> scaler_mean_;   // (num_features,)
    std::vector<float> scaler_scale_;  // (num_features,)

    // Feature history: sliding window [horizon][num_features]
    // Updated during step(), restored to root on set_state()
    std::vector<std::vector<float>> feature_history_;
    std::vector<std::vector<float>> root_feature_history_;

    // State: [x, y, delta, v, psi, omega, beta]
    std::vector<double> state_;
    int step_count_ = 0;
    std::vector<double> v_ref_window_;
    std::mt19937 rng_;

    int n_substeps_ = 1;

    static constexpr double TWO_PI = 2.0 * M_PI;

    // ── Feature extraction ──────────────────────────────────────────────────
    // Mirrors PINNModel._extract_features in pinn.py
    // state = [x, y, delta, v, psi, omega, beta], action = [accl, sv]
    // Returns [vx, vy, yaw_rate, steering_fb, throttle_cmd, steering_cmd]
    std::vector<float> extract_features(
        double v, double beta, double omega, double delta,
        double accl, double steer_vel, double ts
    ) const {
        float vx = static_cast<float>(v * std::cos(beta));
        float vy = static_cast<float>(v * std::sin(beta));
        float yaw_rate = static_cast<float>(omega);
        float steering_fb = static_cast<float>(delta);
        float throttle_cmd = static_cast<float>(accl * ts);
        float steering_cmd = static_cast<float>(steer_vel * ts);
        return {vx, vy, yaw_rate, steering_fb, throttle_cmd, steering_cmd};
    }

    // ── PINN inference → (omega_pred, beta_pred) ────────────────────────────
    // Mirrors PINNModel._predict_batch for batch=1
    std::pair<double, double> pinn_predict(
        double v, double beta_val, double omega, double delta,
        double accl, double steer_vel, double ts
    ) {
        int H = pinn_cfg_.horizon;
        int F = pinn_cfg_.num_features;

        // Extract new features from current state + constrained action
        auto new_feat = extract_features(v, beta_val, omega, delta, accl, steer_vel, ts);

        // Build input tensor: slide window (drop oldest, append new)
        // Shape: (1, H, F)
        auto opts = torch::TensorOptions().dtype(torch::kFloat32);
        auto input_raw  = torch::zeros({1, H, F}, opts);
        auto input_norm = torch::zeros({1, H, F}, opts);

        auto raw_acc  = input_raw.accessor<float, 3>();
        auto norm_acc = input_norm.accessor<float, 3>();

        // Copy history[1:] into positions [0..H-2]
        for (int t = 0; t < H - 1; t++) {
            for (int f = 0; f < F; f++) {
                float val = feature_history_[t + 1][f];
                raw_acc[0][t][f] = val;
                norm_acc[0][t][f] = (val - scaler_mean_[f]) / scaler_scale_[f];
            }
        }
        // Last timestep: new features
        for (int f = 0; f < F; f++) {
            raw_acc[0][H - 1][f] = new_feat[f];
            norm_acc[0][H - 1][f] = (new_feat[f] - scaler_mean_[f]) / scaler_scale_[f];
        }

        // Update feature history (slide window)
        for (int t = 0; t < H - 1; t++) {
            feature_history_[t] = feature_history_[t + 1];
        }
        feature_history_[H - 1] = new_feat;

        // Initialize GRU hidden state to zeros: (num_layers, batch, hidden_size)
        auto h0 = torch::zeros(
            {pinn_cfg_.gru_num_layers, 1, pinn_cfg_.gru_hidden_size}, opts);

        // Run PINN forward
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_raw);
        inputs.push_back(input_norm);
        inputs.push_back(h0);

        auto output_tuple = pinn_model_.forward(inputs).toTuple();
        auto output = output_tuple->elements()[0].toTensor();  // (1, 3)

        float vx_pred       = output[0][0].item<float>();
        float vy_pred       = output[0][1].item<float>();
        float yaw_rate_pred = output[0][2].item<float>();

        double omega_pred = static_cast<double>(yaw_rate_pred);
        double beta_pred  = std::atan2(
            static_cast<double>(vy_pred),
            static_cast<double>(vx_pred) + 1e-8
        );

        return {omega_pred, beta_pred};
    }

  public:
    F1TenthSimulator(const VehicleParamsCpp& vp,
                     const OccupancyMapData& occ_map,
                     const CostParams& cost,
                     const std::string& model_path,
                     const std::vector<float>& scaler_mean,
                     const std::vector<float>& scaler_scale,
                     unsigned seed = std::random_device{}())
        : vp_(vp), occ_map_(occ_map), cost_(cost),
          scaler_mean_(scaler_mean), scaler_scale_(scaler_scale),
          state_(7, 0.0), rng_(seed)
    {
        // Load TorchScript model onto CPU
        pinn_model_ = torch::jit::load(model_path);
        pinn_model_.eval();

        // Compute substeps: e.g. dt=0.04 / base=0.02 → 2 substeps
        n_substeps_ = std::max(
            1, static_cast<int>(std::round(cost_.dt / pinn_cfg_.base_timestep)));

        // Initialize feature history to zeros
        int H = pinn_cfg_.horizon;
        int F = pinn_cfg_.num_features;
        feature_history_.assign(H, std::vector<float>(F, 0.0f));
        root_feature_history_ = feature_history_;
    }

    // --- Simulator interface ---

    bool is_action_discrete() const override { return false; }
    bool is_state_discrete() const override { return false; }
    int rollout_horizon() const override { return cost_.rollout_horizon; }
    double discount() const override { return cost_.discount_factor; }

    void set_state(const std::vector<double>& state) override {
        state_ = state;
        step_count_ = 0;
        // Restore feature history to root snapshot
        feature_history_ = root_feature_history_;
    }

    // Called from Python to set the root feature history
    void set_feature_history(const std::vector<std::vector<float>>& history) {
        root_feature_history_ = history;
        feature_history_ = history;
    }

    void set_v_ref_window(const std::vector<double>& window) {
        v_ref_window_ = window;
    }

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

        // --- Steering constraint (same as original) ---
        if ((delta <= vp_.s_min && sv <= 0.0) ||
            (delta >= vp_.s_max && sv >= 0.0)) {
            sv = 0.0;
        }
        sv = std::max(vp_.sv_min, std::min(vp_.sv_max, sv));

        // --- Acceleration constraint (same as original) ---
        double pos_limit;
        if (v > vp_.v_switch) {
            pos_limit = vp_.a_max * vp_.v_switch / (v + 1e-8);
        } else {
            pos_limit = vp_.a_max;
        }
        if ((v <= vp_.v_min && accl <= 0.0) ||
            (v >= vp_.v_max && accl >= 0.0)) {
            accl = 0.0;
        }
        accl = std::max(-vp_.a_max, std::min(pos_limit, accl));

        // --- PINN dynamics with substeps ---
        bool use_chaining = n_substeps_ > 1;
        double base_dt = pinn_cfg_.base_timestep;
        // Feature timestep: base_dt when chaining, full dt otherwise
        double ts_feature = use_chaining ? base_dt : cost_.dt;

        if (!use_chaining) {
            // Single-step: one PINN call, integrate with full dt
            auto [omega_pred, beta_pred] = pinn_predict(
                v, beta, omega, delta, accl, sv, ts_feature);

            double dt = cost_.dt;
            x     += dt * v * std::cos(beta_pred + psi);
            y     += dt * v * std::sin(beta_pred + psi);
            delta += dt * sv;
            v     += dt * accl;
            psi   += dt * omega_pred;
            omega  = omega_pred;
            beta   = beta_pred;
        } else {
            // Chaining: n substeps of base_dt each
            for (int sub = 0; sub < n_substeps_; sub++) {
                auto [omega_pred, beta_pred] = pinn_predict(
                    v, beta, omega, delta, accl, sv, ts_feature);

                x     += base_dt * v * std::cos(beta_pred + psi);
                y     += base_dt * v * std::sin(beta_pred + psi);
                delta += base_dt * sv;
                v     += base_dt * accl;
                psi   += base_dt * omega_pred;
                omega  = omega_pred;
                beta   = beta_pred;
            }
        }

        // Normalize heading to [0, 2*pi)
        psi = std::fmod(psi, TWO_PI);
        if (psi < 0.0) psi += TWO_PI;

        state_ = {x, y, delta, v, psi, omega, beta};

        int t = step_count_;
        step_count_++;

        // --- Cost computation (same as original f1tenth_simulator.hpp) ---

        // Track cost: 0.9^t * off_track * 10000
        float off_track = occ_map_.query(x, y);
        double track_cost = std::pow(0.9, static_cast<double>(t))
                          * static_cast<double>(off_track) * 10000.0;

        // Speed cost: (v - v_ref_t)^2
        double v_ref_t;
        if (!v_ref_window_.empty() &&
            t < static_cast<int>(v_ref_window_.size())) {
            v_ref_t = v_ref_window_[t];
        } else {
            v_ref_t = cost_.v_ref;
        }
        double speed_cost = (v - v_ref_t) * (v - v_ref_t);

        // Slip cost: alpha^2 + 10000 * I(|alpha| > threshold)
        double v_x = v * std::cos(beta);
        double v_y = v * std::sin(beta);
        double alpha_slip = -std::atan2(v_y, std::abs(v_x) + 1e-8);
        double slip_cost = alpha_slip * alpha_slip
                         + 10000.0 * (std::abs(alpha_slip) > cost_.alpha_threshold
                                      ? 1.0 : 0.0);

        // Reward = negated cost (MCTS maximizes)
        double reward = -(track_cost + cost_.speed_weight * speed_cost + slip_cost);

        return {state_, reward, false};
    }

    std::vector<double> sample_action() override {
        std::uniform_real_distribution<double> accel_dist(-vp_.a_max, vp_.a_max);
        std::uniform_real_distribution<double> steer_dist(vp_.sv_min, vp_.sv_max);
        return {accel_dist(rng_), steer_dist(rng_)};
    }

    std::unique_ptr<Simulator> clone() const override {
        // Create a copy sharing the same PINN model (torch::jit::Module
        // is internally reference-counted, safe for read-only inference).
        auto copy = std::make_unique<F1TenthSimulator>(*this);
        return copy;
    }
};
