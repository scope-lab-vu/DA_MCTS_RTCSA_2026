#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

struct RandomNode;

struct DecisionNode {
    std::vector<double> state;
    std::map<std::vector<double>, std::shared_ptr<RandomNode>> children;
    int visits = 0;
    double reward = 0.0;
    bool is_final = false;
    bool is_root = false;
    RandomNode* father = nullptr;

    DecisionNode() = default;

    DecisionNode(std::vector<double> state, bool is_root = false,
                 bool is_final = false, RandomNode* father = nullptr)
        : state(std::move(state)), is_final(is_final), is_root(is_root),
          father(father) {}

    // Get or create a child RandomNode for the given action
    std::shared_ptr<RandomNode> next_random_node(const std::vector<double>& action);
};

struct RandomNode {
    std::vector<double> action;
    std::map<std::vector<double>, std::shared_ptr<DecisionNode>> children;
    double cumulative_reward = 0.0;
    double reward = 0.0;
    int visits = 0;
    DecisionNode* father = nullptr;

    RandomNode() = default;

    RandomNode(std::vector<double> action, DecisionNode* father = nullptr)
        : action(std::move(action)), father(father) {}

    void add_child(std::shared_ptr<DecisionNode> child) {
        children[child->state] = child;
    }
};

// Implementation (inline, header-only)
inline std::shared_ptr<RandomNode>
DecisionNode::next_random_node(const std::vector<double>& action) {
    auto it = children.find(action);
    if (it != children.end()) {
        return it->second;
    }
    auto node = std::make_shared<RandomNode>(action, this);
    children[action] = node;
    return node;
}
