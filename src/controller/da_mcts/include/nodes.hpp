#pragma once

#include "types.hpp"
#include <map>
#include <memory>
#include <string>

struct RandomNode;

struct DecisionNode {
    State7 state;
    std::map<Action2, std::shared_ptr<RandomNode>> children;
    int visits = 0;
    double reward = 0.0;
    bool is_final = false;
    bool is_root = false;
    RandomNode* father = nullptr;

    DecisionNode() : state{} {}

    DecisionNode(State7 state, bool is_root = false,
                 bool is_final = false, RandomNode* father = nullptr)
        : state(state), is_final(is_final), is_root(is_root),
          father(father) {}

    // Get or create a child RandomNode for the given action
    std::shared_ptr<RandomNode> next_random_node(const Action2& action);
};

struct RandomNode {
    Action2 action;
    std::map<State7, std::shared_ptr<DecisionNode>> children;
    double cumulative_reward = 0.0;
    double reward = 0.0;
    int visits = 0;
    DecisionNode* father = nullptr;

    // GP variance cache — computed once on first expansion, reused for subsequent samples.
    // Stores the posterior variance of omega[5] and beta[6] at this (state, action) point.
    bool gp_computed = false;
    double omega_var = 0.0;
    double beta_var = 0.0;

    RandomNode() : action{} {}

    RandomNode(Action2 action, DecisionNode* father = nullptr)
        : action(action), father(father) {}

    void add_child(std::shared_ptr<DecisionNode> child) {
        children[child->state] = child;
    }
};

// Implementation (inline, header-only)
inline std::shared_ptr<RandomNode>
DecisionNode::next_random_node(const Action2& action) {
    auto it = children.find(action);
    if (it != children.end()) {
        return it->second;
    }
    auto node = std::make_shared<RandomNode>(action, this);
    children[action] = node;
    return node;
}
