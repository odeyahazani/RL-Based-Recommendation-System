"""
movierec_rl/evaluate.py  – Compare DQN agent to heuristic baselines (extended for multi-threshold engagement analysis).
"""

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import entropy

from .dqn import DQNAgent
from .env import MovieRecEnvironment
from .utils import AttrDict, seed_everything, TBLogger   

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _sample_k(rng, candidates: list[int], k: int) -> list[int]:
    """Select *k* unique indices from *candidates* using the given RNG."""
    assert len(candidates) >= k, "Not enough valid actions to form a slate"
    return rng.sample(candidates, k)

# ------------------------------------------------------------
# Baseline policies (all return List[int] of length *k*)
# ------------------------------------------------------------
def random_policy(state, env, k: int):
    valid_actions = np.flatnonzero(env.get_valid_actions())
    return _sample_k(env.rng, list(valid_actions), k)

def greedy_policy(state, env, k: int):
    """
    Greedy baseline: pick top-k actions by their immediate “score” in *state*.
    Handles mismatch between len(state)=84 and action_dim≈500.
    """
    import numpy as np
    valid_mask = env.get_valid_actions()

    # Build full score vector
    scores = np.full(env.action_dim, -np.inf, dtype=float)
    copy_len = min(len(state), env.action_dim)
    scores[:copy_len] = state[:copy_len]
    scores[~valid_mask] = -np.inf

    topk = np.argsort(scores)[-k:][::-1]
    return topk.tolist()

def make_dqn_policy(agent):
    """Return a policy(state, env) -> slate that uses the trained DQN agent."""
    def _policy(state, env):
        valid_mask = env.get_valid_actions()
        return agent.act(state, explore=False, valid_mask=valid_mask)
    return _policy

# ------------------------------------------------------------
# Static / Popularity / Onboarding policies
# ------------------------------------------------------------
def make_static_policy(k: int):
    """Always recommend movies from the user’s currently favourite genre."""
    def policy(state, env):
        if not hasattr(env, "fav_genre"):
            genre_dim = len(env.genre_view_ratio)
            env.fav_genre = int(np.argmax(state[:genre_dim]))
        genre_idx = [i for i, m in enumerate(env.movies) if m["genre"] == env.fav_genre]
        slate = genre_idx[:k]
        if len(slate) < k:
            remaining = [i for i in range(env.action_dim) if i not in slate]
            slate += _sample_k(env.rng, remaining, k - len(slate))
        return slate
    return policy

def make_popularity_policy(env_template, k: int):
    genre_dim = len(env_template.genre_view_ratio)
    most_popular = int(np.argmax(env_template.genre_view_ratio[:genre_dim]))

    def policy(state, env):
        genre_idx = [i for i, m in enumerate(env.movies) if m["genre"] == most_popular]
        slate = genre_idx[:k]
        if len(slate) < k:
            remaining = [i for i in range(env.action_dim) if i not in slate]
            slate += _sample_k(env.rng, remaining, k - len(slate))
        return slate
    return policy

def make_onboarding_policy(k: int = 5):
    """Cycle deterministically through the first *k*×*k* movies, then best genre."""
    def policy(state, env):
        if not hasattr(env, "onb_step"):
            env.onb_step = 0
            env.genre_scores = np.zeros(len(env.genre_view_ratio))

        if env.onb_step < k:
            # ensure unique indices modulo action_dim
            start = env.onb_step * k
            slate = [(start + i) % env.action_dim for i in range(k)]
            env.onb_step += 1
            return slate
        else:
            best_genre = int(np.argmax(env.genre_scores))
            genre_idx = [i for i, m in enumerate(env.movies) if m["genre"] == best_genre]
            slate = genre_idx[:k]
            if len(slate) < k:
                remaining = [i for i in range(env.action_dim) if i not in slate]
                slate += _sample_k(env.rng, remaining, k - len(slate))
            return slate
    return policy

# ------------------------------------------------------------
# Episode rollout
# ------------------------------------------------------------
def run_episode(env, policy_fn):
    """Roll out a single episode using *policy_fn* and collect metrics."""
    state = env.reset()
    done = False
    ep_ret = 0.0
    watch_fractions, genres = [], []

    while not done:
        slate = policy_fn(state, env)
        next_state, reward, done, info = env.step(slate)

        watch_fractions.append(info["watch_fraction"])

        g = info["genre"]
        if g >= 0:                      # Ignore skipped slates (g == -1)
            genres.append(g)
            if hasattr(env, "genre_scores"):
                env.genre_scores[g] += info["watch_fraction"]

        ep_ret += reward
        state = next_state

    avg_watch = float(np.mean(watch_fractions))
    return ep_ret, avg_watch, watch_fractions, genres

# ------------------------------------------------------------
# Evaluation loop (now supports multi-threshold engagement analysis)
# ------------------------------------------------------------
def evaluate(cfg):
    """
    Evaluate DQN and heuristic baselines; save metrics to CSV.
    Supports multi-threshold engagement analysis for research flexibility.
    """
    seed_everything(cfg.seed)
    k = cfg.env.slate_size
    tb = TBLogger(cfg)

    thresholds = getattr(cfg.eval, "engagement_thresholds", None)
    if thresholds is None:
        thresholds = [getattr(cfg.eval, "engagement_threshold", 0.3)]
    else:
        thresholds = list(thresholds)

    env_template = MovieRecEnvironment(
        max_steps=cfg.env.max_steps,
        seed=cfg.seed,
        reward_mode=cfg.env.reward_mode,
        slate_size=k,
        movies_per_genre=cfg.env.movies_per_genre,
    )

    agent = DQNAgent(cfg, action_dim=env_template.action_dim)
    ckpt = Path(cfg.train.save_dir) / cfg.eval.ckpt_file
    agent.load(ckpt)

    approaches = {
        "Random": lambda s, e: random_policy(s, e, k),
        "Greedy": lambda s, e: greedy_policy(s, e, k),
        "Static": make_static_policy(k),
        "Popularity": make_popularity_policy(env_template, k),
        "Onboarding": make_onboarding_policy(k),
        "DQN": make_dqn_policy(agent),
    }

    rows = []
    all_watch_histograms = {}

    for name, policy in approaches.items():
        total_ret, watch_lst, genres_all = 0.0, [], []
        watch_fraction_full = []

        for _ in range(cfg.eval.episodes):
            env = MovieRecEnvironment(
                max_steps=cfg.env.max_steps,
                seed=cfg.seed + 1,
                reward_mode=cfg.env.reward_mode,
                slate_size=k,
                movies_per_genre=cfg.env.movies_per_genre,
            )
            ep_ret, avg_watch, watch_fractions, genres = run_episode(env, policy)
            total_ret += ep_ret
            watch_lst.append(avg_watch)
            watch_fraction_full.extend(watch_fractions)
            genres_all.extend(genres)

        engagement_rates = {}
        for thr in thresholds:
            engagement_rates[f"engagement_rate_{thr:.2f}"] = float(
                np.mean(np.array(watch_fraction_full) >= thr)
            )

        mean_watch_fraction = float(np.mean(watch_fraction_full))

        genre_dist = np.bincount(genres_all, minlength=len(env_template.genre_view_ratio))
        genre_probs = (
            genre_dist / genre_dist.sum()
            if genre_dist.sum() > 0
            else np.ones_like(genre_dist) / len(genre_dist)
        )
        genre_div = entropy(genre_probs)

        all_watch_histograms[name] = np.array(watch_fraction_full)

        row = dict(
            approach=name,
            avg_return=total_ret / cfg.eval.episodes,
            avg_watch_fraction=float(np.mean(watch_lst)),
            mean_watch_fraction=mean_watch_fraction,
            genre_diversity=genre_div,
        )
        row["skip_rate"] = float(np.mean(np.array(watch_fraction_full) == 0.0))
        row.update(engagement_rates)
        rows.append(row)

        if tb.enabled:
            step = 0
            tb.scalar(f"{name}/avg_return", row["avg_return"], step)
            tb.scalar(f"{name}/mean_watch_frac", mean_watch_fraction, step)
            tb.scalar(f"{name}/genre_diversity", genre_div, step)
            for k_thr, v_thr in engagement_rates.items():
                tb.scalar(f"{name}/{k_thr}", v_thr, step)

        all_watch_histograms[name] = np.array(watch_fraction_full)

    out_csv = Path(cfg.eval.csv_out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved metrics to {out_csv}")

    for name, hist in all_watch_histograms.items():
        np.save(out_csv.parent / f"{name}_watch_fractions.npy", hist)

    tb.close()
    print(f"Saved all watch fraction histograms to {out_csv.parent}")

# ------------------------------------------------------------
# Public main() – for import usage
# ------------------------------------------------------------
def main(config_path: str = "movierec_rl/config.yaml"):
    cfg = AttrDict(yaml.safe_load(open(config_path, "r", encoding="utf-8")))
    evaluate(cfg)

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="movierec_rl/config.yaml")
    args = p.parse_args()
    main(args.config)
