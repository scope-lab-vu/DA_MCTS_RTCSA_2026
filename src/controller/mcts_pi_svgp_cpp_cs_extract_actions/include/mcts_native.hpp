#pragma once

#include "nodes.hpp"
#include "simulator.hpp"
#include <cmath>
#include <limits>
#include <random>
#include <utility>

// Pure C++ MCTS — no Python calls in the hot path.
// Uses the Simulator interface directly.
// Adapted from the reference implementation with rollout_horizon/discount support.

class MCTSNative {
  protected:
    Simulator* sim_;  // non-owning, caller manages lifetime
    double K_;
    std::shared_ptr<DecisionNode> root_;
    std::mt19937 rng_;

  public:
    MCTSNative(Simulator* sim, const std::vector<double>& initial_obs, double K,
               unsigned seed = 0)
        : sim_(sim), K_(K), rng_(seed) {
        root_ = std::make_shared<DecisionNode>(initial_obs, true);
    }

    virtual ~MCTSNative() = default;

    void grow_tree() {
        // Reset simulator to root state
        sim_->set_state(root_->state);

        std::vector<std::pair<RandomNode*, DecisionNode*>> path;
        DecisionNode* dn = root_.get();

        while (!dn->is_final && dn->visits > 1) {
            auto action = select(dn);
            auto rnd_node = dn->next_random_node(action);
            RandomNode* rnd_raw = rnd_node.get();

            auto [new_dn, r] = select_outcome(rnd_raw);

            auto it = rnd_raw->children.find(new_dn->state);
            if (it == rnd_raw->children.end()) {
                new_dn->father = rnd_raw;
                rnd_raw->add_child(new_dn);
            } else {
                new_dn = it->second;
            }

            new_dn->reward = r;
            rnd_raw->reward = r;
            path.push_back({rnd_raw, new_dn.get()});
            dn = new_dn.get();
        }

        dn->visits += 1;
        double cumulative_reward = evaluate();

        // Backpropagate
        for (int i = static_cast<int>(path.size()) - 1; i >= 0; --i) {
            auto [rnd, dec] = path[i];
            cumulative_reward += rnd->reward;
            rnd->cumulative_reward += cumulative_reward;
            rnd->visits += 1;
            DecisionNode* parent_dn =
                (i > 0) ? path[i - 1].second : root_.get();
            parent_dn->visits += 1;
        }
    }

    virtual double evaluate() {
        int max_iter = sim_->rollout_horizon();
        double disc = sim_->discount();
        double R = 0.0;
        bool done = false;
        for (int i = 0; i < max_iter && !done; ++i) {
            auto a = sim_->sample_action();
            auto result = sim_->step(a);
            R += std::pow(disc, static_cast<double>(i)) * result.reward;
            done = result.done;
        }
        return R;
    }

    virtual std::pair<std::shared_ptr<DecisionNode>, double>
    select_outcome(RandomNode* random_node) {
        auto result = sim_->step(random_node->action);
        auto node = std::make_shared<DecisionNode>(
            std::move(result.observation), false, result.done, random_node);
        return {node, result.reward};
    }

    virtual std::vector<double> select(DecisionNode* x) {
        return ucb_select(x);
    }

    std::vector<double> ucb_select(DecisionNode* x) {
        std::vector<double> best_action;
        double best_score = -std::numeric_limits<double>::infinity();
        double log_visits = std::log(static_cast<double>(x->visits));

        for (auto& [action, child] : x->children) {
            double score;
            if (child->visits > 0) {
                score = child->cumulative_reward / child->visits +
                        K_ * std::sqrt(log_visits / child->visits);
            } else {
                score = std::numeric_limits<double>::infinity();
            }
            if (score > best_score) {
                best_score = score;
                best_action = action;
            }
        }
        return best_action;
    }

    std::vector<double> best_action_vec() {
        int max_visits = -1;
        std::vector<double> best;
        for (auto& [action, child] : root_->children) {
            if (child->visits > max_visits) {
                max_visits = child->visits;
                best = action;
            }
        }
        return best;
    }

    // Extract the best path from root to leaf by following most-visited
    // children at each level.  Returns a flat vector of actions:
    //   [a0_0, a0_1, a1_0, a1_1, ..., aN_0, aN_1]
    // where each action is 2-dim [accel, steer_vel].
    // Path may be shorter than rollout_horizon if a terminal node is reached.
    std::vector<double> best_path_flat() {
        std::vector<double> path;
        DecisionNode* dn = root_.get();

        while (!dn->children.empty() && !dn->is_final) {
            // Pick the most-visited action child (RandomNode)
            int max_visits = -1;
            RandomNode* best_rn = nullptr;
            for (auto& [action, rn] : dn->children) {
                if (rn->visits > max_visits) {
                    max_visits = rn->visits;
                    best_rn = rn.get();
                }
            }
            if (!best_rn) break;

            // Record this action
            for (double v : best_rn->action) {
                path.push_back(v);
            }

            // Pick the most-visited outcome child (DecisionNode)
            if (best_rn->children.empty()) break;
            int max_dn_visits = -1;
            DecisionNode* best_dn = nullptr;
            for (auto& [state, child_dn] : best_rn->children) {
                if (child_dn->visits > max_dn_visits) {
                    max_dn_visits = child_dn->visits;
                    best_dn = child_dn.get();
                }
            }
            if (!best_dn) break;
            dn = best_dn;
        }
        return path;
    }

    void learn(int Nsim) {
        for (int i = 0; i < Nsim; ++i) {
            grow_tree();
        }
        sim_->set_state(root_->state);
    }

    void forward(const std::vector<double>& action,
                 const std::vector<double>& new_state) {
        auto it = root_->children.find(action);
        if (it != root_->children.end()) {
            auto& rnd_node = it->second;
            if (rnd_node->children.size() > 1) {
                root_ = std::make_shared<DecisionNode>(new_state, true);
            } else if (!rnd_node->children.empty()) {
                auto& next = rnd_node->children.begin()->second;
                double dist = 0.0;
                for (size_t i = 0; i < next->state.size(); ++i) {
                    double d = next->state[i] - new_state[i];
                    dist += d * d;
                }
                if (std::sqrt(dist) > 1e-3) {
                    root_ = std::make_shared<DecisionNode>(new_state, true);
                } else {
                    next->father = nullptr;
                    next->is_root = true;
                    next->is_final = false;
                    root_ = next;
                }
            } else {
                root_ = std::make_shared<DecisionNode>(new_state, true);
            }
        } else {
            root_ = std::make_shared<DecisionNode>(new_state, true);
        }
    }
};

// -- SPW Native ---------------------------------------------------------------

class SPWNative : public MCTSNative {
  protected:
    double alpha_;

  public:
    SPWNative(Simulator* sim, const std::vector<double>& initial_obs,
              double alpha, double K, unsigned seed = 0)
        : MCTSNative(sim, initial_obs, K, seed), alpha_(alpha) {}

    std::vector<double> select(DecisionNode* x) override {
        if (std::pow(static_cast<double>(x->visits), alpha_) >=
            static_cast<double>(x->children.size())) {
            return sim_->sample_action();
        } else {
            return ucb_select(x);
        }
    }
};

// -- DPW Native ---------------------------------------------------------------

class DPWNative : public SPWNative {
  protected:
    double beta_;

  public:
    DPWNative(Simulator* sim, const std::vector<double>& initial_obs,
              double alpha, double beta, double K, unsigned seed = 0)
        : SPWNative(sim, initial_obs, alpha, K, seed), beta_(beta) {}

    std::pair<std::shared_ptr<DecisionNode>, double>
    select_outcome(RandomNode* random_node) override {
        if (std::pow(static_cast<double>(random_node->visits), beta_) >=
            static_cast<double>(random_node->children.size())) {
            auto result = sim_->step(random_node->action);
            auto node = std::make_shared<DecisionNode>(
                std::move(result.observation), false, result.done,
                random_node);
            return {node, result.reward};
        } else {
            // Sample from existing outcomes proportional to visits
            std::vector<double> weights;
            std::vector<std::shared_ptr<DecisionNode>> nodes;
            for (auto& [state, child] : random_node->children) {
                weights.push_back(static_cast<double>(child->visits));
                nodes.push_back(child);
            }
            double total = 0.0;
            for (double w : weights) total += w;

            std::uniform_real_distribution<double> dist(0.0, total);
            double r = dist(rng_);
            double cum = 0.0;
            size_t idx = 0;
            for (size_t i = 0; i < weights.size(); ++i) {
                cum += weights[i];
                if (r <= cum) {
                    idx = i;
                    break;
                }
            }
            auto& chosen = nodes[idx];
            return {chosen, chosen->reward};
        }
    }
};
