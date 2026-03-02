#pragma once

#include <memory>
#include <vector>

// Step result: (observation, reward, done)
struct StepResult {
    std::vector<double> observation;
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

    virtual StepResult step(const std::vector<double>& action) = 0;
    virtual std::vector<double> sample_action() = 0;
    virtual void set_state(const std::vector<double>& state) = 0;
    virtual std::unique_ptr<Simulator> clone() const = 0;

    // Optional: rollout parameters (used by MCTS evaluate)
    virtual int rollout_horizon() const { return 100; }
    virtual double discount() const { return 1.0; }

    // Dynamically change rollout horizon (e.g. based on curvature)
    virtual void set_rollout_horizon(int /*h*/) {}
};
