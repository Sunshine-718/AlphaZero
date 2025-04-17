# cython: language_level=3
# distutils: language = c++
import numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cdef class TreeNode:
    cdef object parent
    cdef dict children
    cdef int n_visits
    cdef double Q
    cdef double u
    cdef double prior
    cdef double noise
    cdef bint deterministic

    def __cinit__(self, parent, double prior, double dirichlet_noise=0):
        self.parent = parent
        self.children = {}
        self.n_visits = 0
        self.Q = 0.0
        self.u = 0.0
        self.prior = prior
        self.noise = dirichlet_noise if dirichlet_noise != 0 else prior
        self.deterministic = False

    cpdef void train(self):
        if self.deterministic:
            if not self.children:
                self.deterministic = False
                return
            for node in self.children.values():
                node.train()
            self.deterministic = False

    cpdef void eval(self):
        if not self.deterministic:
            if not self.children:
                self.deterministic = True
                return
            for node in self.children.values():
                node.eval()
            self.deterministic = True

    cpdef void expand(self, list action_probs, object noise=None):
        cdef int idx
        for idx in range(len(action_probs)):
            action, prior = action_probs[idx]
            if action not in self.children:
                if noise is None or self.deterministic:
                    self.children[action] = TreeNode(self, prior, prior)
                else:
                    self.children[action] = TreeNode(self, prior, noise[idx])

    cpdef tuple select(self, double c_puct):
        cdef object best_action = None
        cdef object best_node = None
        cdef double best_value = -1e9
        for action, node in self.children.items():
            val = node.PUCT(c_puct)
            if val > best_value:
                best_value = val
                best_action = action
                best_node = node
        return best_action, best_node

    cpdef void update(self, double leaf_value, double discount):
        if self.parent is not None:
            (<TreeNode>self.parent).update(-leaf_value * discount, discount)
        self.n_visits += 1
        self.Q += (leaf_value - self.Q) / self.n_visits

    cpdef double PUCT(self, double c_puct):
        cdef double prior_val
        if self.parent is not None and (<TreeNode>self.parent).is_root() and not self.deterministic:
            prior_val = 0.75 * self.prior + 0.25 * self.noise
        else:
            prior_val = self.prior
        cdef TreeNode parent_node = <TreeNode>self.parent
        self.u = c_puct * prior_val * np.sqrt(parent_node.n_visits) / (1 + self.n_visits)
        return self.Q + self.u

    cpdef bint is_leaf(self):
        return not self.children

    cpdef bint is_root(self):
        return self.parent is None


cdef class MCTS:
    cdef TreeNode root
    cdef object policy
    cdef double c_puct
    cdef int n_playout

    def __cinit__(self, policy_value_fn, double c_puct=1, int n_playout=1000):
        self.root = TreeNode(None, 1, 1)
        self.policy = policy_value_fn
        self.c_puct = c_puct
        self.n_playout = n_playout

    cpdef void train(self):
        self.root.train()

    cpdef void eval(self):
        self.root.eval()

    cpdef object select_leaf_node(self, object env):
        cdef object node = self.root
        while not node.is_leaf():
            action, node = node.select(self.c_puct)
            env.step(action)
        return node

    cpdef void playout(self, object env, double discount=1):
        cdef object node = self.select_leaf_node(env)
        action_probs, leaf_value = self.policy(env)
        if not env.done():
            node.expand(action_probs)
        node.update(-leaf_value, discount)

    cpdef tuple get_action(self, object env, double discount=1):
        cdef int i
        for i in range(self.n_playout):
            self.playout(env.copy(), discount)
        cdef object best_action = None
        cdef int best_visits = -1
        for action, node in self.root.children.items():
            if node.n_visits > best_visits:
                best_visits = node.n_visits
                best_action = action
        return best_action, self.root.children[best_action]

    cpdef void update_with_move(self, int last_move):
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = TreeNode(None, 1, 1)


cdef class MCTS_AZ(MCTS):
    cpdef void playout(self, object env, double dirichlet_alpha=0.3, double discount=1):
        cdef object node = self.select_leaf_node(env)
        action_probs, leaf_value = self.policy(env)
        if not env.done():
            noise = np.random.dirichlet([dirichlet_alpha for _ in range(len(action_probs))]) if dirichlet_alpha is not None else None
            node.expand(action_probs, noise)
        else:
            winner = env.winPlayer()
            if winner == 0:
                leaf_value = 0
            else:
                leaf_value = 1 if winner == env.turn else -1
        node.update(-leaf_value, discount)

    cpdef tuple get_action_visits(self, object env, double dirichlet_alpha=0.3, double discount=1):
        cdef int i
        for i in range(self.n_playout):
            self.playout(env.copy(), dirichlet_alpha, discount)
        cdef list act_visits = []
        for action, node in self.root.children.items():
            act_visits.append((action, (<TreeNode>node).n_visits))
        actions, visits = zip(*act_visits)
        return actions, visits