import numpy as np
import random
from typing import Tuple, List
from .sum_tree import SumTree


class ReplayBuffer:
    """
    Experience replay buffer.
    • uniform mode (default) – behaves exactly like before.
    • per=True            – uses proportional PER (Schaul et al. 2015).
    """

    def __init__(self,
                 capacity: int,
                 seed: int | None = None,
                 *,
                 per: bool = False,
                 alpha: float = 0.6,
                 beta_start: float = 0.4,
                 beta_end: float = 1.0,
                 beta_steps: int = 100_000,
                 eps_priority: float = 1e-6):
        self.capacity = capacity
        self.per = per
        self.storage = []
        self.next_idx = 0
        self.rng = random.Random(seed)

        if per:
            tree_cap = 1 << (capacity - 1).bit_length()
            self.tree = SumTree(tree_cap)
            self.alpha = alpha
            self.beta_start = beta_start
            self.beta_end = beta_end
            self.beta_inc = (beta_end - beta_start) / beta_steps
            self.eps_priority = float(eps_priority)
            self.beta = beta_start
            self._leaf_map = {}  # ← ניהול מפה קבועה

    def append(self, s, a, r, s2, done, *, td_error: float | None = None):
        """
        td_error: needed only when per=True (priority -|δ|).
        """
        data = (s, a, r, s2, done)
        if len(self.storage) < self.capacity:
            self.storage.append(data)
        else:
            self.storage[self.next_idx] = data

        if self.per:
            priority = (abs(td_error) + self.eps_priority) ** self.alpha if td_error is not None else 1.0
            leaf_idx = self.tree.add(priority)
            self._leaf_map[leaf_idx] = self.next_idx

        self.next_idx = (self.next_idx + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[List, ...]:
        """
        Returns:
          states, actions, rewards, next_states, dones,
          indices, is_weights   (אחרונים קיימים רק ב-per)
        """
        if not self.per:
            idxs = self.rng.sample(range(len(self.storage)), batch_size)
            batch = [self.storage[i] for i in idxs]
            return [np.array(x) for x in zip(*batch)]
          
        if len(self) < batch_size:                       
            raise BufferError("Not enough samples in buffer")   
                   
        segment = self.tree.total / batch_size
        idxs, priorities = [], []
        for i in range(batch_size):
            mass = self.rng.uniform(i * segment, (i + 1) * segment)
            leaf = self.tree.sample(mass)
            idx = self._leaf_map.get(leaf)
            if idx is None:
                idx = self.rng.randint(0, len(self.storage) - 1)
            idxs.append(idx)
            priorities.append(self.tree.tree[leaf])

        batch = [self.storage[i] for i in idxs]
        beta = min(self.beta + self.beta_inc, self.beta_end)
        self.beta = beta

        probs = np.array(priorities) / self.tree.total
        probs = np.clip(probs, 1e-8, 1.0)  # ← הוספת חסם תחתון
        weights = (len(self.storage) * probs) ** (-beta)
        weights /= weights.max()


        return (*[np.array(x) for x in zip(*batch)], np.array(idxs), weights)

    def update_priorities(self, idxs, td_errors):
        if not self.per:
            return
        for idx, err in zip(idxs, td_errors):
            leaf = next((k for k, v in self._leaf_map.items() if v == idx), None)
            if leaf is not None:
                priority = (abs(err) + self.eps_priority) ** self.alpha
                self.tree.update(leaf, priority)

    def __len__(self):
        return len(self.storage)
