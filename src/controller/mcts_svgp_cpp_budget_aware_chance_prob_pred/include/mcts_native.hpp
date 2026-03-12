#pragma once

#include "nodes.hpp"
#include "simulator.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <random>
#include <string>
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
    int max_depth_seen_ = 0;

  public:
    MCTSNative(Simulator* sim, const State7& initial_obs, double K,
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
        if (static_cast<int>(path.size()) > max_depth_seen_)
            max_depth_seen_ = static_cast<int>(path.size());
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
            result.observation, false, result.done, random_node);
        return {node, result.reward};
    }

    virtual Action2 select(DecisionNode* x) {
        return ucb_select(x);
    }

    Action2 ucb_select(DecisionNode* x) {
        const Action2* best_action_ptr = nullptr;
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
                best_action_ptr = &action;
            }
        }
        return *best_action_ptr;
    }

    Action2 best_action_vec() {
        int max_visits = -1;
        Action2 best{};
        for (auto& [action, child] : root_->children) {
            if (child->visits > max_visits) {
                max_visits = child->visits;
                best = action;
            }
        }
        return best;
    }

    void learn(int Nsim) {
        for (int i = 0; i < Nsim; ++i) {
            grow_tree();
        }
        sim_->set_state(root_->state);
    }

    virtual int learn_timed(double budget_s) {
        auto start = std::chrono::steady_clock::now();
        auto deadline = start + std::chrono::duration<double>(budget_s);
        int count = 0;
        while (std::chrono::steady_clock::now() < deadline) {
            grow_tree();
            ++count;
        }
        sim_->set_state(root_->state);
        return count;
    }

    // Return root children: list of {action, visits, avg_reward}
    struct ActionInfo {
        Action2 action;
        int visits;
        double avg_reward;
    };

    std::vector<ActionInfo> get_root_actions() const {
        std::vector<ActionInfo> result;
        for (auto& [action, rnd] : root_->children) {
            double avg = (rnd->visits > 0)
                ? rnd->cumulative_reward / rnd->visits
                : 0.0;
            result.push_back({action, rnd->visits, avg});
        }
        return result;
    }

    // Walk the tree following most-visited actions/outcomes to extract
    // the best trajectory MCTS has found.
    struct TrajectoryStep {
        State7 state;
        Action2 action;
        double avg_reward;
        int visits;
    };

    std::vector<TrajectoryStep> get_best_trajectory(int max_depth = 50) const {
        std::vector<TrajectoryStep> traj;
        const DecisionNode* dn = root_.get();

        for (int d = 0; d < max_depth; ++d) {
            if (dn->is_final || dn->children.empty()) break;

            // Find most-visited action (RandomNode)
            int best_visits = -1;
            const RandomNode* best_rnd = nullptr;
            Action2 best_action{};
            for (auto& [action, rnd] : dn->children) {
                if (rnd->visits > best_visits) {
                    best_visits = rnd->visits;
                    best_rnd = rnd.get();
                    best_action = action;
                }
            }
            if (!best_rnd || best_rnd->children.empty()) break;

            double avg_r = (best_rnd->visits > 0)
                ? best_rnd->cumulative_reward / best_rnd->visits : 0.0;
            traj.push_back({dn->state, best_action, avg_r, best_visits});

            // Follow most-visited outcome (DecisionNode child)
            int best_out_visits = -1;
            const DecisionNode* best_outcome = nullptr;
            for (auto& [state, child] : best_rnd->children) {
                if (child->visits > best_out_visits) {
                    best_out_visits = child->visits;
                    best_outcome = child.get();
                }
            }
            if (!best_outcome) break;
            dn = best_outcome;
        }
        // Add final state
        if (dn && !traj.empty()) {
            traj.push_back({dn->state, Action2{}, 0.0, dn->visits});
        }
        return traj;
    }

    // Maximum depth seen across all grow_tree() calls since last reset.
    // O(1) — maintained as a running max inside grow_tree().
    int max_tree_depth() const { return max_depth_seen_; }

    void forward(const Action2& action,
                 const State7& new_state) {
        auto it = root_->children.find(action);
        if (it != root_->children.end()) {
            auto& rnd_node = it->second;
            if (rnd_node->children.size() > 1) {
                root_ = std::make_shared<DecisionNode>(new_state, true);
            } else if (!rnd_node->children.empty()) {
                auto& next = rnd_node->children.begin()->second;
                double dist = 0.0;
                for (size_t i = 0; i < STATE_DIM; ++i) {
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
    SPWNative(Simulator* sim, const State7& initial_obs,
              double alpha, double K, unsigned seed = 0)
        : MCTSNative(sim, initial_obs, K, seed), alpha_(alpha) {}

    Action2 select(DecisionNode* x) override {
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

    // Budget-aware MCTS parameters
    double W_low_ = 15.0;
    double W_high_ = 50.0;
    int N_w_ = 100;
    int N_r_ = 100;
    double last_tau_ = 0.0;

    // Context set from Python each step for logging
    double v_context_ = 0.0;
    double kappa_context_ = 0.0;

    // Per-episode weight log
    std::ofstream log_file_;

  public:
    DPWNative(Simulator* sim, const State7& initial_obs,
              double alpha, double beta, double K, unsigned seed = 0)
        : SPWNative(sim, initial_obs, alpha, K, seed), beta_(beta) {}

    void set_budget_params(double W_low, double W_high, int N_w, int N_r) {
        W_low_ = W_low;
        W_high_ = W_high;
        N_w_ = N_w;
        N_r_ = N_r;
    }

    // Called from Python before each learn_timed() to provide logging context.
    void set_context(double v, double kappa) {
        v_context_ = v;
        kappa_context_ = kappa;
    }

    // Open (or replace) the per-episode CSV log file.
    void open_log(const std::string& path) {
        if (log_file_.is_open()) log_file_.close();
        log_file_.open(path, std::ios::out | std::ios::trunc);
        if (log_file_.is_open())
            log_file_ << "n_iter,t_remain_s,v,kappa,p_agg,p_slow,p_feas,max_depth\n";
    }

    struct ActionWeights {
        double p_agg, p_slow, p_feas;
    };

    void write_log_row(int n, double t_remain, const ActionWeights& w) {
        if (!log_file_.is_open()) return;
        log_file_ << n << ","
                  << t_remain << ","
                  << v_context_ << ","
                  << kappa_context_ << ","
                  << w.p_agg << ","
                  << w.p_slow << ","
                  << w.p_feas << ","
                  << max_depth_seen_ << "\n";
        log_file_.flush();
    }

    ActionWeights compute_weights(double W_hat) {
        double tau = std::max(0.0, std::min(1.0,
            (W_hat - W_low_) / (W_high_ - W_low_)));
        last_tau_ = tau;
        double p_slow = 0.90 - 0.85 * tau;
        double p_feas = 0.05 + 0.55 * tau;
        double p_agg = 1.0 - p_slow - p_feas;
        return {p_agg, p_slow, p_feas};
    }

    // Budget-aware 3-phase learn_timed
    int learn_timed(double budget_s) override {
        auto start = std::chrono::steady_clock::now();
        auto deadline = start + std::chrono::duration<double>(budget_s);
        int n = 0;
        max_depth_seen_ = 0;  // reset per learn_timed() call

        // Phase 1: Warmup — equal sampling (1/3, 1/3, 1/3)
        sim_->set_action_weights(1.0/3.0, 1.0/3.0, 1.0/3.0);
        while (n < N_w_ && std::chrono::steady_clock::now() < deadline) {
            grow_tree();
            ++n;
        }

        if (n == 0 || std::chrono::steady_clock::now() >= deadline) {
            sim_->set_state(root_->state);
            return n;
        }

        // Phase 2: Estimate budget and adapt weights
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start).count();
        double t_iter = elapsed / n;
        double remaining = std::chrono::duration<double>(deadline - now).count();
        int N_total = n + static_cast<int>(remaining / t_iter);
        double W_hat = std::pow(static_cast<double>(N_total), alpha_);
        auto w = compute_weights(W_hat);
        sim_->set_action_weights(w.p_agg, w.p_slow, w.p_feas);
        write_log_row(n, remaining, w);  // Phase 2: initial weight estimate

        // Phase 3: Adaptive search with periodic re-calibration
        while (std::chrono::steady_clock::now() < deadline) {
            grow_tree();
            ++n;
            if (n % N_r_ == 0) {
                now = std::chrono::steady_clock::now();
                elapsed = std::chrono::duration<double>(now - start).count();
                t_iter = elapsed / n;
                remaining = std::chrono::duration<double>(deadline - now).count();
                N_total = n + static_cast<int>(remaining / t_iter);
                W_hat = std::pow(static_cast<double>(N_total), alpha_);
                w = compute_weights(W_hat);
                sim_->set_action_weights(w.p_agg, w.p_slow, w.p_feas);
                write_log_row(n, remaining, w);  // Phase 3: recalibration
            }
        }

        sim_->set_state(root_->state);
        return n;
    }

    double get_last_tau() const { return last_tau_; }

    std::pair<std::shared_ptr<DecisionNode>, double>
    select_outcome(RandomNode* random_node) override {
        if (std::pow(static_cast<double>(random_node->visits), beta_) >=
            static_cast<double>(random_node->children.size())) {

            if (!random_node->gp_computed) {
                // First expansion: compute mean transition + cache GP variance.
                // The mean state becomes the first (deterministic) child.
                auto result = sim_->step_with_variance(random_node->action);
                random_node->omega_var  = result.omega_var;
                random_node->beta_var   = result.beta_var;
                random_node->gp_computed = true;
                auto node = std::make_shared<DecisionNode>(
                    result.mean_observation, false, result.done, random_node);
                return {node, result.reward};
            } else {
                // Subsequent expansions: deterministic step (advances simulator),
                // then perturb omega[5] and beta[6] with cached GP std-dev to
                // generate a distinct stochastic outcome.
                auto result = sim_->step(random_node->action);
                State7 sampled = result.observation;
                std::normal_distribution<double> norm(0.0, 1.0);
                if (random_node->omega_var > 0.0)
                    sampled[5] += norm(rng_) * std::sqrt(random_node->omega_var);
                if (random_node->beta_var > 0.0)
                    sampled[6] += norm(rng_) * std::sqrt(random_node->beta_var);
                auto node = std::make_shared<DecisionNode>(
                    sampled, false, result.done, random_node);
                return {node, result.reward};
            }

        } else {
            // Reuse an existing outcome, sampled proportional to visit counts.
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
            // Sync simulator to the chosen outcome so evaluate() rolls out correctly.
            sim_->set_state(chosen->state);
            return {chosen, chosen->reward};
        }
    }
};
