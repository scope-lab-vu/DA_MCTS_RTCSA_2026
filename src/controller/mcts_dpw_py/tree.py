"""
MCTS, SPW, and DPW tree search algorithms.

Adapted from reference implementation at:
    MCTS_DPW/mcts_no_gym/MCTS.py
    MCTS_DPW/mcts_no_gym/SPW.py
    MCTS_DPW/mcts_no_gym/DPW.py

Based on: Monte Carlo Tree Search for Continuous and Stochastic
Sequential Decision Making Problems, Courtoux.
"""

import numpy as np
import cloudpickle

from src.controller.mcts_dpw_py.nodes import DecisionNode, RandomNode


class MCTS:
    """Base MCTS with UCB selection for continuous state/action spaces.

    :param initial_obs: numpy array — initial state
    :param env: simulator implementing step(), sample_action(), copy()
    :param K: UCB exploration constant
    """

    def __init__(self, initial_obs, env, K):
        self.env = env
        self.K = K
        self.root = DecisionNode(state=initial_obs, is_root=True)
        # Continuous state and action — hash via tuple conversion.
        self._hash_action = lambda x: tuple(x)
        self._hash_space = lambda x: tuple(x)

    def update_decision_node(self, decision_node, random_node, hash_preprocess):
        """Return existing or register new decision node under random_node."""
        key = hash_preprocess(decision_node.state)
        if key not in random_node.children:
            decision_node.father = random_node
            random_node.add_children(decision_node, hash_preprocess)
        else:
            decision_node = random_node.children[key]
        return decision_node

    def grow_tree(self):
        """One iteration: selection → expansion → rollout → backprop."""
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

        # Expansion + rollout.
        decision_node.visits += 1
        cumulative_reward = self.evaluate(internal_env)

        # Backpropagation.
        while not decision_node.is_root:
            random_node = decision_node.father
            cumulative_reward += random_node.reward
            random_node.cumulative_reward += cumulative_reward
            random_node.visits += 1
            decision_node = random_node.father
            decision_node.visits += 1

    def evaluate(self, env):
        """Random rollout from current simulator state.

        :param env: simulator copy at the leaf node
        :return: cumulative discounted reward from rollout
        """
        max_iter = getattr(env, 'rollout_horizon', 100)
        discount = getattr(env, 'discount', 1.0)
        R = 0.0
        done = False
        t = 0
        while (not done) and (t < max_iter):
            a = env.sample_action()
            s, r, done, _ = env.step(a)
            R += (discount ** t) * r
            t += 1
        return R

    def select_outcome(self, env, random_node):
        """Step the simulator to get a new state from a random node.

        :return: (DecisionNode, reward)
        """
        new_state, r, done, _ = env.step(random_node.action)
        return DecisionNode(state=new_state, father=random_node, is_final=done), r

    def select(self, x):
        """UCB action selection (base class — discrete, not used for F1TENTH)."""
        def scoring(k):
            child = x.children[k]
            if child.visits > 0:
                return (child.cumulative_reward / child.visits
                        + self.K * np.sqrt(np.log(x.visits) / child.visits))
            return np.inf

        return max(x.children, key=scoring)

    def best_action(self):
        """Return the most-visited child action."""
        visits = [node.visits for node in self.root.children.values()]
        idx = np.argmax(visits)
        return list(self.root.children.values())[idx].action

    def learn(self, Nsim):
        """Run Nsim tree traversals."""
        for _ in range(Nsim):
            self.grow_tree()

    def forward(self, action, new_state):
        """Tree reuse: advance root to the child matching (action, new_state).

        For deterministic dynamics, reuses most of the tree structure.
        Raises RuntimeWarning if the action is not found or env is stochastic.
        """
        key = self._hash_action(action)
        if key in self.root.children:
            rnd_node = self.root.children[key]
            if len(rnd_node.children) > 1:
                self.root = DecisionNode(state=new_state, is_root=True)
            else:
                next_dn = list(rnd_node.children.values())[0]
                if np.linalg.norm(next_dn.state - new_state) > 1e-3:
                    raise RuntimeWarning("The env is probably stochastic")
                next_dn.father = None
                self.root.children.pop(key)
                self.root = next_dn
                self.root.is_root = True
                # Clear terminal flag — the node may have been marked
                # terminal during simulation, but the car is still running.
                self.root.is_final = False
        else:
            raise RuntimeWarning(
                f"Action {action} not in root children."
            )

    def act(self):
        """Alias for best_action()."""
        return self.best_action()

    def save(self, path=None):
        """Save tree to pickle file."""
        data = self._collect_data()
        if path is None:
            name = "".join(np.random.choice(
                list("abcdef0123456789"), size=8,
            ))
            path = f"./logs/{name}_"
        with open(path, "wb") as f:
            cloudpickle.dump(data, f)

    def _collect_data(self):
        return {"K": self.K, "root": self.root}


class SPW(MCTS):
    """Simple Progressive Widening — controls action branching.

    :param alpha: widening exponent — new action sampled when
        visits^alpha >= len(children)
    """

    def __init__(self, alpha, initial_obs, env, K):
        super().__init__(initial_obs, env, K)
        self.alpha = alpha

    def select(self, x):
        """Progressive widening action selection."""
        if x.visits ** self.alpha >= len(x.children):
            return self.env.sample_action()

        def scoring(k):
            child = x.children[k]
            if child.visits > 0:
                return (child.cumulative_reward / child.visits
                        + self.K * np.sqrt(np.log(x.visits) / child.visits))
            return np.inf

        return max(x.children, key=scoring)

    def _collect_data(self):
        data = super()._collect_data()
        data["alpha"] = self.alpha
        return data


class DPW(SPW):
    """Double Progressive Widening — controls both action and outcome branching.

    :param beta: outcome widening exponent — new outcome sampled when
        visits^beta >= len(children)
    """

    def __init__(self, alpha, beta, initial_obs, env, K):
        super().__init__(alpha, initial_obs, env, K)
        self.beta = beta

    def select_outcome(self, env, random_node):
        """Progressive widening on outcomes."""
        if random_node.visits ** self.beta >= len(random_node.children):
            new_state, r, done, _ = env.step(random_node.action)
            return DecisionNode(state=new_state, father=random_node, is_final=done), r

        # Reuse existing outcome proportional to visit count.
        unnorm_probs = [child.visits for child in random_node.children.values()]
        probs = np.array(unnorm_probs) / np.sum(unnorm_probs)
        chosen_state = np.random.choice(
            list(random_node.children.values()), p=probs,
        )
        return chosen_state, chosen_state.reward

    def _collect_data(self):
        data = super()._collect_data()
        data["beta"] = self.beta
        return data
