"""
Decision and Random nodes for MCTS tree.

Adapted from reference implementation at:
    MCTS_DPW/mcts_no_gym/Nodes.py
"""


class DecisionNode:
    """State node in the MCTS tree.

    :param state: numpy array defining the state
    :param father: RandomNode parent (None if root)
    :param is_root: whether this is the tree root
    :param is_final: whether this is a terminal state
    """

    def __init__(self, state=None, father=None, is_root=False, is_final=False):
        self.state = state
        self.children = {}
        self.is_final = is_final
        self.visits = 0
        self.reward = 0
        self.father = father
        self.is_root = is_root

    def add_children(self, random_node, hash_preprocess):
        """Add a RandomNode to children dict, keyed by hashed action."""
        self.children[hash_preprocess(random_node.action)] = random_node

    def next_random_node(self, action, hash_preprocess):
        """Get or create the RandomNode for the given action."""
        key = hash_preprocess(action)
        if key not in self.children:
            new_random_node = RandomNode(action, father=self)
            self.add_children(new_random_node, hash_preprocess)
        else:
            new_random_node = self.children[key]
        return new_random_node

    def __repr__(self):
        s = ""
        for k, v in self.__dict__.items():
            if k in ("children", "father"):
                continue
            s += f"{k}: {v}\n"
        return s


class RandomNode:
    """Action/chance node in the MCTS tree.

    :param action: numpy array of the action taken
    :param father: DecisionNode parent
    """

    def __init__(self, action, father=None):
        self.action = action
        self.children = {}
        self.cumulative_reward = 0
        self.visits = 0
        self.father = father

    def add_children(self, decision_node, hash_preprocess):
        """Add a DecisionNode to children dict, keyed by hashed state."""
        self.children[hash_preprocess(decision_node.state)] = decision_node

    def __repr__(self):
        mean_rew = round(self.cumulative_reward / (self.visits + 1), 2)
        return f"action: {self.action}\nmean_reward: {mean_rew}\nvisits: {self.visits}"
