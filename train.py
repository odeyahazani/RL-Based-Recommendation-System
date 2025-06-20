from __future__ import annotations

import os
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from .dqn import DQNAgent
from .env import MovieRecEnvironment
from .evaluate import greedy_policy
from .replay_buffer import ReplayBuffer
from .utils import AttrDict, TBLogger, seed_everything


def train_agent(cfg):
    seed_everything(cfg.seed)
    tb = TBLogger(cfg)

    env = MovieRecEnvironment(
        max_steps=cfg.env.max_steps,
        seed=cfg.seed,
        reward_mode=cfg.env.reward_mode,
        slate_size=cfg.env.slate_size,
        movies_per_genre=cfg.env.movies_per_genre,
    )
    thresholds = getattr(cfg.eval, "engagement_thresholds", None)
    env.engagement_threshold = (
        thresholds[0]
        if thresholds is not None
        else getattr(cfg.eval, "engagement_threshold", 0.3)
    )

    agent = DQNAgent(cfg, action_dim=env.action_dim)

    rb_cfg = cfg.replay_buffer
    buffer = ReplayBuffer(
        capacity=rb_cfg.capacity,
        seed=cfg.seed,
        per=rb_cfg.per,
        alpha=rb_cfg.alpha,
        beta_start=rb_cfg.beta_start,
        beta_end=rb_cfg.beta_end,
        beta_steps=rb_cfg.beta_steps,
        eps_priority=rb_cfg.eps_priority,
    )

    returns_last_10 = deque(maxlen=10)
    loss_running_mean: Optional[float] = None

    # ----------------------------------------------------------
    # 🟡 Pre-fill replay buffer with greedy policy (warm-up)
    # ----------------------------------------------------------
    print("📥 Pre-filling buffer with greedy policy...")
    while len(buffer) < agent.batch_size * 10:
        state = env.reset()
        done = False
        while not done:
            slate = greedy_policy(state, env, cfg.env.slate_size)
            next_state, reward, done, info = env.step(slate)
            chosen_movie = info["chosen_movie"] if info["chosen_movie"] != -1 else slate[0]
            agent.observe(buffer, state, chosen_movie, reward, next_state, done)
            state = next_state
    print(f"✅ Buffer pre-filled: {len(buffer)} transitions")

    # ============================================================
    for ep in range(1, cfg.train.episodes + 1):
        state = env.reset()
        # print(f"[DEBUG] Ep {ep:04d}  buffer={len(buffer)}")

        done, ep_ret = False, 0.0
        losses_episode: list[float] = []

        num_skips = 0
        total_watch = 0.0

        while not done:
            valid_mask = env.get_valid_actions()
            slate = agent.act(state, explore=True, valid_mask=valid_mask)
            next_state, reward, done, info = env.step(slate)
            chosen_movie = info.get("chosen_movie", -1)
            watch_fraction = info.get("watch_fraction", 0.0)

            if chosen_movie == -1:
                chosen_movie = slate[0]
                num_skips += 1
            if watch_fraction == 0.0:
                num_skips += 1
            total_watch += watch_fraction

            agent.observe(buffer, state, chosen_movie, reward, next_state, done)

            loss_val = None
            if len(buffer) >= agent.batch_size:
                loss_val = agent.learn(buffer)
                if loss_val is not None and agent.global_step == 0:
                    print(f"[DEBUG] First loss = {loss_val:.4f}")
                # NEW: warn if loss is too large
                if loss_val is not None and loss_val > 10.0:
                    print(
                        f"⚠️ [Warning] High loss at step {agent.global_step}: "
                        f"{loss_val:.2f}"
                    )

            if loss_val is not None:
                losses_episode.append(loss_val)
                tb.scalar("Loss/step", loss_val, agent.global_step)

            state = next_state
            ep_ret += reward

        # ---- episode metrics -----------------------------------
        returns_last_10.append(ep_ret)
        avg_ret10 = np.mean(returns_last_10)

        avg_loss_ep = np.mean(losses_episode) if losses_episode else 0.0
        loss_running_mean = (
            avg_loss_ep
            if loss_running_mean is None
            else 0.9 * loss_running_mean + 0.1 * avg_loss_ep
        )

        avg_watch_fraction = total_watch / env.max_steps
        skip_rate = num_skips / env.max_steps

        tb.scalar("Reward/avg_10ep", avg_ret10, ep)
        tb.scalar("Loss/avg_ep", loss_running_mean, ep)
        tb.scalar("Exploration/eps", agent.epsilon(), ep)
        tb.scalar("Engagement/avg_watch_fraction", avg_watch_fraction, ep)
        tb.scalar("Engagement/skip_rate", skip_rate, ep)

        if ep % cfg.train.checkpoint_every == 0:
            Path(cfg.train.save_dir).mkdir(parents=True, exist_ok=True)
            agent.save(Path(cfg.train.save_dir) / f"ep{ep:04d}.pt")

        if ep % cfg.train.log_every == 0:
            print(
                f"[Ep {ep:03d}] "
                f"avg_ret={avg_ret10:.2f} "
                f"avg_loss={loss_running_mean:.4f} "
                f"ε={agent.epsilon():.3f} "
                f"watch={avg_watch_fraction:.2f} "
                f"skips={skip_rate:.2f}"
            )

    tb.close()


def load_cfg(path: str = "config.yaml") -> AttrDict:
    with open(path, "r", encoding="utf-8") as f:
        return AttrDict(yaml.safe_load(f))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", default="movierec_rl/config.yaml")
    args = p.parse_args()
    cfg = load_cfg(args.config)
    train_agent(cfg)
