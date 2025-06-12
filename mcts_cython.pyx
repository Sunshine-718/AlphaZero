# cython: language_level=3, boundscheck=False, wraparound=False
# distutils: language = c++
import cython
import math
import numpy as np
cimport numpy as np

ctypedef long long ll

cdef class TreeNode:
    cdef public TreeNode parent
    cdef dict children
    cdef ll _n_visits
    cdef double Q, u, prior, noise
    cdef bint deterministic

    def __cinit__(self, TreeNode parent, double prior, double dirichlet_noise=0):
        self.parent = parent
        self.children = {}
        self._n_visits = 0
        self.Q = 0.0
        self.u = 0.0
        self.prior = prior
        self.noise = dirichlet_noise if dirichlet_noise != 0 else prior
        self.deterministic = False

    cpdef bint is_leaf(self):
        return not self.children

    cpdef bint is_root(self):
        return self.parent is None

    cpdef void expand(self, action_probs, object noise=None):
        for idx, (action, prior) in enumerate(action_probs):
            if action not in self.children:
                if noise is None or self.deterministic:
                    self.children[action] = TreeNode(self, prior, prior)
                else:
                    self.children[action] = TreeNode(self, prior, noise[idx])

    cpdef tuple select(self, double c_init, double c_base):
        cdef double best = -1e100
        cdef object best_k = None
        cdef TreeNode best_n = None
        cdef double score
        for k, n in self.children.items():
            score = n.PUCT(c_init, c_base)
            if score > best:
                best = score
                best_k = k
                best_n = n
        return best_k, best_n

    @property
    def n_visits(self):
        return self._n_visits

    cpdef double PUCT(self, double c_init, double c_base):
        cdef double prior_tmp
        if self.parent is not None and self.parent.is_root() and not self.deterministic:
            prior_tmp = 0.75 * self.prior + 0.25 * self.noise
        else:
            prior_tmp = self.prior
        self.u = (c_init + math.log((1.0 + self.parent._n_visits + c_base) / c_base)) * \
                 prior_tmp * math.sqrt(self.parent._n_visits) / (1.0 + self.n_visits)
        return self.Q + self.u

    cpdef void update(self, double leaf_value, double discount):
        if self.parent is not None:
            self.parent.update(-leaf_value * discount, discount)
        self._n_visits += 1
        self.Q += (leaf_value - self.Q) / self._n_visits


cdef class _BaseMCTS:
    cdef TreeNode root
    cdef object policy
    cdef double c_init, c_base
    cdef ll n_playout

    def __init__(self, policy_value_fn, double c_init=1.0, ll n_playout=800):
        self.root = TreeNode(None, 1.0, 1.0)
        self.policy = policy_value_fn
        self.c_init = c_init
        self.c_base = n_playout / 800.0 * 19652
        self.n_playout = n_playout

    cpdef void _single_playout(self, env, double discount=1.0, double dirichlet_alpha=-1):
        node = self.root
        while not node.is_leaf():
            action, node = node.select(self.c_init, self.c_base)
            env.step(action)

        env_aug, flipped = env.random_flip()
        action_probs, leaf_value = self.policy(env_aug)
        if flipped:
            action_probs = [(env.flip_action(a), p) for a, p in action_probs]

        if env.done():
            winner = env.winPlayer()
            leaf_value = 0 if winner == 0 else (1 if winner == env.turn else -1)
        else:
            if dirichlet_alpha > 0:
                noise = np.random.dirichlet([dirichlet_alpha] * len(action_probs))
            else:
                noise = None
            node.expand(action_probs, noise)
        node.update(-leaf_value, discount)

    cpdef void playout_batch(self, env, double discount=1.0, double dirichlet_alpha=-1):
        cdef ll i
        for i in range(self.n_playout):
            self._single_playout(env.copy(), discount, dirichlet_alpha)


cdef class MCTS(_BaseMCTS):
    cdef bint random_flip
    def __cinit__(self, policy_value_fn, double c_init=1.0, ll n_playout=800, bint random_flip=True):
        super().__init__(policy_value_fn, c_init, n_playout)
        self.random_flip = random_flip

    def playout(self, env, discount=1.0):
        self._single_playout(env, discount)

    def get_action(self, env, discount=1.0):
        self.playout_batch(env, discount)
        action = max(self.root.children.items(), key=lambda kv: kv[1].n_visits)[0]
        return action

    def update_with_move(self, last_move):
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = TreeNode(None, 1.0, 1.0)


cdef class MCTS_AZ(MCTS):
    def __cinit__(self, policy_value_fn, double c_init=1.0, ll n_playout=800):
        super().__init__(policy_value_fn, c_init, n_playout)

    def playout(self, env, double dirichlet_alpha=0.3, double discount=1.0):
        self._single_playout(env, discount, dirichlet_alpha)

    def get_action_visits(self, env, double dirichlet_alpha=0.3, double discount=1.0):
        self.playout_batch(env, discount, dirichlet_alpha)
        actions, visits = zip(*[(a, n.n_visits) for a, n in self.root.children.items()])
        return actions, visits

    def greedy_backup_value(self, env, double discount=1.0):
        node = self.root
        turn0 = env.turn
        while not node.is_leaf():
            best = max(node.children.items(), key=lambda kv: kv[1].n_visits)[0]
            env.step(best)
            node = node.children[best]
        if env.done():
            winner = env.winPlayer()
            leaf_val = 0 if winner == 0 else (1 if winner == turn0 else -1)
        else:
            _, leaf_val = self.policy(env)
        return float(leaf_val * discount)


from concurrent.futures import ThreadPoolExecutor

class RootParallelMCTS:
    def __init__(self, policy_value_fn, c_init=1.25, n_playout=800, num_worker=4):
        assert num_worker >= 1
        self.num_worker = num_worker
        sims_per_worker = max(1, n_playout // num_worker)
        self.workers = [MCTS_AZ(policy_value_fn, c_init, sims_per_worker)
                        for _ in range(num_worker)]
        self.n_actions = policy_value_fn.n_actions

    def train(self):
        for w in self.workers:
            w.train()

    def eval(self):
        for w in self.workers:
            w.eval()

    def update_with_move(self, last_move):
        for w in self.workers:
            w.update_with_move(last_move)

    def get_action_visits(self, env, double dirichlet_alpha=0.3, double discount=1.0):
        copies = [env.copy() for _ in range(self.num_worker)]
        with ThreadPoolExecutor(max_workers=self.num_worker) as pool:
            results = [pool.submit(w.get_action_visits, copies[i],
                                   dirichlet_alpha, discount)
                       for i, w in enumerate(self.workers)]
            results = [r.result() for r in results]

        import numpy as _np
        total = _np.zeros(self.n_actions, dtype=_np.int64)
        for acts, vis in results:
            for a, v in zip(acts, vis):
                total[a] += v
        acts = [a for a in range(self.n_actions) if total[a] > 0]
        return acts, total[acts]

    def greedy_backup_value(self, env, double discount=1.0):
        return self.workers[0].greedy_backup_value(env, discount)
