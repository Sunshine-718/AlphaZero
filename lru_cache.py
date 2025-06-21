from collections import OrderedDict
import torch
import numpy as np

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.state_map = {}
        self.hits = 0
        self.misses = 0

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key, value, env):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                oldest, _ = self.cache.popitem(last=False)
                self.state_map.pop(oldest, None)
        self.cache[key] = value
        self.state_map[key] = {'state': env.current_state(),
                       'env'  : env.copy()}

    def refresh(self, policy_value_fn):
        keys = list(self.state_map.keys())
        if not keys:
            return

        states = [self.state_map[k]['state'] for k in keys]
        envs   = [self.state_map[k]['env']   for k in keys]
        
        states = np.concatenate(states, axis=0)
        states_flipped = states[:, :, :, ::-1].copy()

        state_batch1 = torch.from_numpy(states).float().to(policy_value_fn.device)
        state_batch2 = torch.from_numpy(states_flipped).float().to(policy_value_fn.device)
        mask_batch1 = torch.tensor([e.valid_mask() for e in envs],dtype=torch.bool, device=policy_value_fn.device)
        mask_batch2 = torch.tensor([e.valid_mask()[::-1] for e in envs],dtype=torch.bool, device=policy_value_fn.device)
        self.cache.clear()
        with torch.no_grad():
            probs_batch1, values_batch1 = policy_value_fn.policy_value(state_batch1, mask_batch1)
            probs_batch2, values_batch2 = policy_value_fn.policy_value(state_batch2, mask_batch2)
        probs_batch = (probs_batch1 + probs_batch2) / 2
        values_batch = (values_batch1 + values_batch2) / 2

        for key, prob_vec, val in zip(keys, probs_batch, values_batch):
            valid = envs[keys.index(key)].valid_move()
            action_probs = tuple(zip(valid, prob_vec[valid]))
            self.cache[key] = (action_probs, float(val))

    def hit_rate(self):
        total = self.hits + self.misses
        if total == 0:
            return "缓存命中率：无请求"
        rate = 100 * self.hits / total
        return f"缓存命中率：{rate:.2f}%（{self.hits}/{total}）"
