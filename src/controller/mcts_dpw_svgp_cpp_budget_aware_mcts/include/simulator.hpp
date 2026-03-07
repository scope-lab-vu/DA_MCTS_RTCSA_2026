#pragma once

#include "types.hpp"
#include <memory>

// Step result: (observation, reward, done)
struct StepResult {
    State7 observation;
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

    // Optional: rollout parameters (used by MCTS evaluate)
    virtual int rollout_horizon() const { return 100; }
    virtual double discount() const { return 1.0; }

    // Dynamically change rollout horizon (e.g. based on curvature)
    virtual void set_rollout_horizon(int /*h*/) {}

    // Budget-aware action sampling weights (p_agg, p_slow, p_feas)
    virtual void set_action_weights(double /*p_agg*/, double /*p_slow*/, double /*p_feas*/) {}
};
