#pragma once

#include "types.hpp"
#include <memory>

// Step result: (observation, reward, done)
struct StepResult {
    State7 observation;
    double reward;
    bool done;
};

// Step result with GP posterior variance for omega and beta residuals.
// omega_var and beta_var are in the original (denormalized) output scale.
struct StepWithVarianceResult {
    State7 mean_observation;  // deterministic (mean) transition — same as StepResult
    double omega_var;         // GP posterior variance for omega correction
    double beta_var;          // GP posterior variance for beta correction
    double reward;
    bool done;
};

// Abstract C++ simulator interface
class Simulator {
  public:
    virtual ~Simulator() = default;

    virtual bool is_action_discrete() const = 0;
    virtual bool is_state_discrete() const = 0;
    virtual int n_actions() const { return 0; }

    virtual StepResult step(const Action2& action) = 0;
    virtual Action2 sample_action() = 0;
    virtual void set_state(const State7& state) = 0;
    virtual std::unique_ptr<Simulator> clone() const = 0;

    // step_with_variance: deterministic step + GP posterior variance for sampling
    // Default falls back to step() with zero variance (deterministic simulators).
    virtual StepWithVarianceResult step_with_variance(const Action2& action) {
        auto r = step(action);
        return {r.observation, 0.0, 0.0, r.reward, r.done};
    }

    // Optional: rollout parameters (used by MCTS evaluate)
    virtual int rollout_horizon() const { return 100; }
    virtual double discount() const { return 1.0; }

    // Dynamically change rollout horizon (e.g. based on curvature)
    virtual void set_rollout_horizon(int /*h*/) {}
};
