"""
Batched DPW tree search with PINN evaluation.

Extends the standard DPW tree to batch leaf rollout evaluations through
the PINN model, dramatically reducing the number of sequential neural
network calls by evaluating B leaves in parallel per round.
"""

import numpy as np

from src.controller.mcts_dpw_py.nodes import DecisionNode
from src.controller.mcts_dpw_py.tree import DPW


class BatchedDPW(DPW):
    """DPW with batched PINN rollout evaluation.

    Instead of evaluating one leaf at a time (sequential random rollout),
    collects B leaves per round and evaluates all their rollouts in a single
    batched PINN forward pass per rollout step.

    :param alpha: action progressive widening exponent
    :param beta: outcome progressive widening exponent
    :param initial_obs: initial state numpy array
    :param env: F1TenthSelectionSimulator (used for selection dynamics)
    :param K: UCB exploration constant
    :param evaluator: BatchedPINNEvaluator instance
    :param batch_size: number of leaves to collect per evaluation round
    """

    def __init__(self, alpha, beta, initial_obs, env, K, evaluator, batch_size=8):
        super().__init__(alpha, beta, initial_obs, env, K)
        self.evaluator = evaluator
        self.batch_size = batch_size

    def learn(self, Nsim):
        """Run Nsim tree traversals with batched rollout evaluation."""
        remaining = Nsim
        while remaining > 0:
            B = min(self.batch_size, remaining)

            # Phase 1: Selection + Expansion (sequential tree traversals).
            leaf_data = []
            for _ in range(B):
                data = self._select_and_expand()
                leaf_data.append(data)

            # Phase 2: Batched Evaluation (GPU PINN rollouts).
            states = [d["state"] for d in leaf_data]
            histories = [d["feature_history"] for d in leaf_data]

            # Use v_ref_window from the env (set by controller each step).
            v_ref_window = self.env.v_ref_window

            values = self.evaluator.evaluate_batch(
                states, histories, v_ref_window,
            )

            # Phase 3: Backpropagation.
            for i, data in enumerate(leaf_data):
                self._backpropagate(data["node"], float(values[i]))

            remaining -= B

    def _select_and_expand(self):
        """Single tree traversal: selection → expansion.

        Returns dict with leaf node, state, and feature history for evaluation.
        Follows the same logic as DPW.grow_tree() (tree.py:45-60) but
        separates evaluation from the traversal.
        """
        decision_node = self.root
        internal_env = self.env.copy()

        # Selection: descend tree while node is visited and non-terminal.
        while (not decision_node.is_final) and decision_node.visits > 1:
            a = self.select(decision_node)
            new_random_node = decision_node.next_random_node(a, self._hash_action)
            (new_decision_node, r) = self.select_outcome(internal_env, new_random_node)
            new_decision_node = self.update_decision_node(
                new_decision_node, new_random_node, self._hash_space,
            )
            new_decision_node.reward = r
            new_random_node.reward = r
            decision_node = new_decision_node

        # Mark leaf as visited (expansion).
        decision_node.visits += 1

        return {
            "node": decision_node,
            "state": decision_node.state.copy(),
            "feature_history": internal_env._feature_history.clone(),
        }

    def _backpropagate(self, decision_node, cumulative_reward):
        """Backpropagate cumulative reward from leaf to root.

        Same logic as MCTS.grow_tree() backprop (tree.py:67-73).
        """
        while not decision_node.is_root:
            random_node = decision_node.father
            cumulative_reward += random_node.reward
            random_node.cumulative_reward += cumulative_reward
            random_node.visits += 1
            decision_node = random_node.father
            decision_node.visits += 1
