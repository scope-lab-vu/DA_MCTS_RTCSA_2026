#pragma once
#include <array>
#include <cmath>
#include <vector>

// Max hidden dimension (compile-time). Matches best_hp.json hidden_dim=32.
constexpr int MLP_MAX_H = 32;

// Tiny 2-hidden-layer MLP: 5 inputs → [H, H] → 3 outputs (softmax).
// Weights are passed in as row-major float32 vectors from Python.
// predict() uses only stack-allocated arrays — zero heap allocation in the hot path.
struct ProbMLP {
    int H = 0;

    // Row-major weight matrices and bias vectors (heap, but allocated once at load)
    std::vector<float> W0, b0;  // H×5, H
    std::vector<float> W1, b1;  // H×H, H
    std::vector<float> W2, b2;  // 3×H, 3

    // Input normaliser: x_norm = (x - mu) / sigma
    std::array<float, 5> mu{}, sigma{};

    bool ready = false;

    void load(const std::vector<float>& w0, const std::vector<float>& bias0,
              const std::vector<float>& w1, const std::vector<float>& bias1,
              const std::vector<float>& w2, const std::vector<float>& bias2,
              const std::vector<float>& mu_vec,
              const std::vector<float>& sigma_vec) {
        H = static_cast<int>(bias0.size());
        W0 = w0;  b0 = bias0;
        W1 = w1;  b1 = bias1;
        W2 = w2;  b2 = bias2;
        for (int i = 0; i < 5; ++i) {
            mu[i]    = mu_vec[i];
            sigma[i] = sigma_vec[i];
        }
        ready = true;
    }

    // Returns {p_agg, p_slow, p_feas}
    std::array<float, 3> predict(float n_iter, float t_remain,
                                  float v, float kappa, float depth) const {
        // Normalise inputs
        float x[5] = {
            (n_iter   - mu[0]) / sigma[0],
            (t_remain - mu[1]) / sigma[1],
            (v        - mu[2]) / sigma[2],
            (kappa    - mu[3]) / sigma[3],
            (depth    - mu[4]) / sigma[4],
        };

        // Stack-allocated activation buffers (no heap alloc)
        float h1[MLP_MAX_H], h2[MLP_MAX_H];

        // Layer 0: h1 = ReLU(W0 @ x + b0)
        for (int i = 0; i < H; ++i) {
            float acc = b0[i];
            for (int j = 0; j < 5; ++j) acc += W0[i * 5 + j] * x[j];
            h1[i] = acc > 0.0f ? acc : 0.0f;
        }

        // Layer 1: h2 = ReLU(W1 @ h1 + b1)
        for (int i = 0; i < H; ++i) {
            float acc = b1[i];
            for (int j = 0; j < H; ++j) acc += W1[i * H + j] * h1[j];
            h2[i] = acc > 0.0f ? acc : 0.0f;
        }

        // Layer 2: logits = W2 @ h2 + b2
        float logits[3] = {b2[0], b2[1], b2[2]};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < H; ++j)
                logits[i] += W2[i * H + j] * h2[j];

        // Softmax (numerically stable)
        float max_l = logits[0];
        for (int i = 1; i < 3; ++i) if (logits[i] > max_l) max_l = logits[i];
        float e[3], esum = 0.0f;
        for (int i = 0; i < 3; ++i) { e[i] = std::exp(logits[i] - max_l); esum += e[i]; }

        return {e[0] / esum, e[1] / esum, e[2] / esum};
    }
};
