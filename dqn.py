"""
movierec_rl/dqn.py  –  Double-Dueling DQN *patched* (2025-06-11)

Changes in this version
-----------------------
✓ Two hidden layers instead of one  → higher expressive power  
✓ Huber (smooth-L1) loss           → stabler gradients  
✓ Soft target updates (Polyak τ)   → smoother convergence  
✓ `batch_size` & `tau` read from cfg so train.py can rely on them  
✓ Loss-based PER priority update re-uses the same TD-errors  
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .utils import get_device


# ------------------------------------------------------------
# Network definitions
# ------------------------------------------------------------
class DuelingDQN(nn.Module):
    """Two-hidden-layer dueling architecture."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        # shared embedding
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # value & advantage heads
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        f = self.feature(x)
        v = self.value(f)
        a = self.advantage(f)
        # combine streams
        return v + a - a.mean(dim=1, keepdim=True)


# ------------------------------------------------------------
# Agent wrapper
# ------------------------------------------------------------
class DQNAgent:
    """Handles policy / target nets, ε-schedule and PER-aware learning."""

    def __init__(self, cfg, action_dim: int):
        self.device = get_device(cfg.device)
        self.state_dim = cfg.state_dim
        self.action_dim = action_dim
        self.gamma = cfg.agent.gamma
        self.slate_size = cfg.env.slate_size

        # ---- networks ----------------------------------------------------
        self.policy_net = DuelingDQN(
            self.state_dim, action_dim, cfg.agent.hidden_dim
        ).to(self.device)
        self.target_net = DuelingDQN(
            self.state_dim, action_dim, cfg.agent.hidden_dim
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=cfg.agent.lr
        )

        # exploration schedule
        self.eps_start = cfg.agent.epsilon_start
        self.eps_end = cfg.agent.epsilon_end
        self.eps_decay_steps = cfg.agent.epsilon_decay_steps
        self.global_step = 0

        # soft-update & batch parameters
        self.tau = getattr(cfg.agent, "tau", 0.01)
        self.batch_size = cfg.agent.batch_size

    # --------------------------------------------------------
    # Slate ε-greedy policy
    # --------------------------------------------------------
    def act(
        self,
        state: np.ndarray,
        *,
        explore: bool = True,
        valid_mask: Optional[np.ndarray] = None,
    ) -> list[int]:
        self.global_step += 1
        eps = self.epsilon()
        k = self.slate_size

        valid_actions = (
            np.flatnonzero(valid_mask)
            if valid_mask is not None
            else np.arange(self.action_dim)
        )

        # exploration
        if explore and random.random() < eps:
            return list(np.random.choice(valid_actions, size=k, replace=False))

        # exploitation
        st = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.policy_net(st).squeeze(0).cpu().numpy()
        if valid_mask is not None:
            q_vals[~valid_mask] = -np.inf
        topk = np.argsort(q_vals)[-k:][::-1]
        return topk.tolist()

    # --------------------------------------------------------
    # Observe transition – compute TD-error when PER is on
    # --------------------------------------------------------
    def observe(self, buffer, state, action, reward, next_state, done):
        if not buffer.per:
            buffer.append(state, action, reward, next_state, done, td_error=None)
            return

        with torch.no_grad():
            s_t = (
                torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            )
            ns_t = (
                torch.from_numpy(next_state).float().unsqueeze(0).to(self.device)
            )
            q_val = self.policy_net(s_t)[0, action]
            q_next = self.target_net(ns_t).max(1)[0]
            td_target = reward + self.gamma * q_next * (1 - done)
            td_error = (q_val - td_target).abs().item()

        buffer.append(
            state, action, reward, next_state, done, td_error=td_error
        )

    # --------------------------------------------------------
    # Learning step – Double-DQN with PER support
    # --------------------------------------------------------
    def learn(self, buffer):
        """
        One gradient step on a mini-batch.
        Returns float(loss) or None if buffer not ready.
        """
        if len(buffer) < self.batch_size:
            return None

        try:
            batch = buffer.sample(self.batch_size)
        except Exception as e:
            print(f"Buffer sampling error: {e}")
            return None

        # Unpack batch
        if buffer.per:
            (
                states,
                actions,
                rewards,
                next_states,
                dones,
                idxs,
                is_w,
            ) = batch
            is_w = torch.tensor(
                is_w, dtype=torch.float32, device=self.device
            ).unsqueeze(1)
        else:
            states, actions, rewards, next_states, dones = batch
            idxs, is_w = None, 1.0

        # Tensors
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(
            next_states, dtype=torch.float32, device=self.device
        )
        actions = (
            torch.tensor(actions, dtype=torch.int64, device=self.device)
            .unsqueeze(1)
        )
        rewards = (
            torch.tensor(rewards, dtype=torch.float32, device=self.device)
            .unsqueeze(1)
        )
        dones = (
            torch.tensor(dones, dtype=torch.float32, device=self.device)
            .unsqueeze(1)
        )

        # Q(s,a) and targets
        q_sa = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(
                dim=1, keepdim=True
            )
            next_q_vals = self.target_net(next_states).gather(
                1, next_actions
            )
            td_target = rewards + self.gamma * next_q_vals * (1 - dones)

        # ---- Huber loss ----
        td_errors = q_sa - td_target
        # clip TD errors to prevent loss explosions
        td_errors = torch.clamp(td_errors, min=-1.0, max=1.0)
        loss = (is_w * F.smooth_l1_loss(q_sa, td_target, reduction="none")).mean()

        # ---- optimisation ----
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # ---- soft-update target net ----
        with torch.no_grad():
            for tgt, src in zip(
                self.target_net.parameters(), self.policy_net.parameters()
            ):
                tgt.data.mul_(1.0 - self.tau).add_(self.tau * src.data)

        # ---- PER priority update ----
        if buffer.per:
            buffer.update_priorities(idxs, td_errors.squeeze(1).detach().cpu().numpy())

        # numeric safety
        loss_val = float(loss.item())
        assert np.isfinite(loss_val), f"Loss is not finite: {loss_val}"
        return loss_val

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------
    def epsilon(self) -> float:
        progress = min(self.global_step / self.eps_decay_steps, 1.0)
        return self.eps_start * (1 - progress) + self.eps_end * progress

    def save(self, path: str):
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "global_step": self.global_step,
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.global_step = checkpoint["global_step"]
