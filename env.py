"""
movierec_rl/env.py – User-centric movie-recommendation simulator (v2-patched)

Key features
------------
* Per-user genre preference drawn from Dirichlet prior (age × gender)
* Compatibility score per movie: preference − boredom + novelty
* Soft-choice among slate (Softmax with temperature) → user may skip entire slate
* Watch-fraction ∝ preference × (1 − boredom)  (can be 0)
* Penalty on skipped slate (`slate_skip_penalty`)
* **NEW (2025-06-11)** – clip `genre_view_ratio` to [0, 1] so the state always
  stays in a bounded range (prevents exploding inputs to the network)
"""

from __future__ import annotations

import random
from collections import deque
from typing import List, Tuple

import numpy as np

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------
GENRES: List[str] = [
    "action", "adventure", "animation", "comedy",
    "crime", "drama", "fantasy", "horror", "romance", "sci-fi",
]
NUM_GENRES = len(GENRES)

AGE_GROUPS: List[str] = [
    "under10", "10-17", "18-24", "25-34",
    "35-44", "45-54", "55-64", "65plus",
]
NUM_AGE_HOT = len(AGE_GROUPS)

DEMOGRAPHIC_PREFERENCES = {
    "F": {"romance", "drama", "comedy", "family"},
    "M": {"action", "sci-fi", "crime", "horror"},
    "under10": {"animation", "family"},
    "10-17": {"animation", "fantasy", "adventure"},
    "18-24": {"horror", "action", "comedy", "romance"},
    "25-34": {"action", "comedy", "drama", "romance"},
    "35-44": {"drama", "crime", "comedy"},
    "45-54": {"drama", "crime", "history"},
    "55-64": {"drama", "history", "documentary"},
    "65plus": {"drama", "romance", "documentary"},
}

AGE_RESTRICTIONS = {
    "under10": {"horror", "crime"},
    "10-17": {"horror"},
}

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def min_max(x: float, lo: float, hi: float) -> float:
    """Normalise x to [0, 1] given range [lo, hi]."""
    if hi == lo:
        return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


def _softmax(x: np.ndarray, tau: float) -> np.ndarray:
    z = x / tau
    e = np.exp(z - z.max())
    return e / e.sum()

# ------------------------------------------------------------
# Environment class
# ------------------------------------------------------------
class MovieRecEnvironment:
    """Environment that simulates a user with rational choice over a movie slate."""

    # --------------------------------------------------------
    # Construction & reset
    # --------------------------------------------------------
    def __init__(
        self,
        max_steps: int = 30,
        seed: int | None = None,
        reward_mode: str = "shaped",
        recent_genres_len: int = 5,
        slate_size: int = 5,
        movies_per_genre: int = 50,
        temperature: float = 0.5,
        slate_skip_penalty: float = 0.1,
    ) -> None:
        self.max_steps = max_steps
        self.reward_mode = reward_mode
        self.slate_size = slate_size
        self.movies_per_genre = movies_per_genre
        self.temperature = temperature
        self.slate_skip_penalty = slate_skip_penalty

        self.rng = random.Random(seed)
        self.engagement_threshold = 0.3  # for external metrics

        self.recent_genres_len = recent_genres_len
        self.recent_genres = deque(maxlen=recent_genres_len)
        self.watch_fractions = deque(maxlen=recent_genres_len)

        self.global_step = 0
        self.reset()

    # --------------------------------------------------------
    # User profile helpers
    # --------------------------------------------------------
    def _dirichlet_prior(self, age: str, gender: str) -> np.ndarray:
        """Return genre-level Dirichlet α parameters conditioned on age & gender."""
        alpha = np.ones(NUM_GENRES)
        for key in (gender, age):
            for i, gname in enumerate(GENRES):
                if gname in DEMOGRAPHIC_PREFERENCES.get(key, set()):
                    alpha[i] += 3.0
        return alpha

    def _init_user_profile(self):
        """Sample per-user preference and behavioural parameters."""
        alpha = self._dirichlet_prior(self.age_group, self.gender)

        # default_rng **must** receive an int seed, not a float
        seed_int = self.rng.randrange(0, 2**32)
        rng_np = np.random.default_rng(seed_int)

        self.genre_pref = rng_np.dirichlet(alpha)        # length = NUM_GENRES
        self.novelty_w = self.rng.uniform(0.5, 1.0)     # weight for unseen movies
        self.boredom_w = self.rng.uniform(0.4, 0.7)     # boredom growth rate

    # --------------------------------------------------------
    # Movie pool
    # --------------------------------------------------------
    def _generate_movie_pool(self):
        """Create a fixed catalogue of movies (each with genre & length)."""
        movies = []
        for g in range(NUM_GENRES):
            for _ in range(self.movies_per_genre):
                length = self.rng.randint(60, 180)        # minutes
                movies.append({"genre": g, "length": length})
        self.action_dim = len(movies)
        return movies

    # --------------------------------------------------------
    # Compatibility & choice
    # --------------------------------------------------------
    def _compatibility(self, movie_id: int) -> float:
        """Preference score − boredom + novelty."""
        movie = self.movies[movie_id]
        genre = movie["genre"]

        pref     = self.genre_pref[genre]
        boredom  = self.boredom_w * min(self.genre_view_ratio[genre], 1.0)
        novelty  = self.novelty_w if movie_id not in self.history else 0.0
        return pref + novelty - boredom

    def choose_movie(self, slate: List[int]) -> int:
        """Softmax choice among slate; may return −1 to indicate skipping the slate."""
        compat = np.array([self._compatibility(m) for m in slate])
        probs  = _softmax(compat, self.temperature)
        idx    = int(np.searchsorted(np.cumsum(probs), self.rng.random()))
        movie_id = slate[idx]
        if self.rng.random() < 0.05:
          return self.rng.choice(slate)

        # NEW: soften skip decision to make user more forgivin
        p_skip = _sigmoid(-0.4 * compat[idx])
        if self.rng.random() < p_skip:
            return -1
        return movie_id

    # --------------------------------------------------------
    # RL interface
    # --------------------------------------------------------
    def reset(self) -> np.ndarray:  # type: ignore[override]
        self.movies = self._generate_movie_pool()
        self.age_group = self.rng.choice(AGE_GROUPS)
        self.gender    = self.rng.choice(["M", "F"])
        self.genre_view_ratio = np.zeros(NUM_GENRES)
        self._init_user_profile()

        self.genre_view_ratio = np.zeros(NUM_GENRES)
        self.total_watched    = 0.0
        self.step_count       = 0
        self.last_genre       = -1
        self.last_length      = 90
        self.history: List[int] = []
        self.genre_history: List[int] = []
        self.recent_genres.clear()
        self.watch_fractions.clear()
        self.global_step = 0
        return self._state()

    def _demographic_alignment_bonus(self, genre: int) -> float:
        """Small shaped reward for genre matching demographic preferences."""
        gname = GENRES[genre]
        bonus = 0.0
        bonus += 0.02 if gname in DEMOGRAPHIC_PREFERENCES.get(self.gender, set()) else -0.01
        bonus += 0.02 if gname in DEMOGRAPHIC_PREFERENCES.get(self.age_group, set()) else -0.01
        return bonus

    def step(self, slate: List[int]) -> Tuple[np.ndarray, float, bool, dict]:  # type: ignore[override]
        self.genre_view_ratio *= 0.65  # decay boredom each step-------------------------
        movie_id = self.choose_movie(slate)

        if movie_id == -1:  # user skipped entire slate
            watch_fraction = 0.0
            reward = -self.slate_skip_penalty
            genre  = -1
            length = 0

            # slight boredom increase for all genres in slate (user scanned them)
            for mid in slate:
                g = self.movies[mid]["genre"]
                self.genre_view_ratio[g] += 0.01
        else:  # user picked a movie
            movie  = self.movies[movie_id]
            genre  = movie["genre"]
            length = movie["length"]

            pref    = self.genre_pref[genre]
            boredom = self.boredom_w * min(self.genre_view_ratio[genre], 1.0)#min(self.genre_view_ratio[genre], 1.0)
            base    = self.rng.uniform(0.4, 0.9)
            watch_fraction = base * pref * (1.0 - boredom)

            reward = watch_fraction * 1.5  # amplify early rewards
#            if self.reward_mode == "shaped":
#                if movie_id in self.history:  # repeat → penalty
#                    reward -= 0.05
#                if len(self.genre_history) >= 2 and genre not in self.genre_history[-2:]:
#                    reward += 0.02  # mild diversity bonus
#                lambda_demo = 0.5 * np.exp(-self.global_step / 5000)
#                reward += lambda_demo * self._demographic_alignment_bonus(genre)
#                reward *= length / 180.0  # longer movies more valuable

            # update internal state
            self.genre_view_ratio[genre] += watch_fraction
            self.total_watched += watch_fraction * length
            self.last_genre = genre
            self.last_length = length
            self.history.append(movie_id)
            self.genre_history.append(genre)
            self.recent_genres.append(genre)
            self.watch_fractions.append(watch_fraction)

        self.step_count  += 1
        self.global_step += 1
        done = self.step_count >= self.max_steps
        info = {"chosen_movie": movie_id,
                "watch_fraction": watch_fraction,
                "genre": genre}

        return self._state(), reward, done, info

    # --------------------------------------------------------
    # State representation
    # --------------------------------------------------------
    def _state(self) -> np.ndarray:
        """Return a feature vector summarising the current user state."""
        # One-hot of last watched genre
        last_genre_oh = np.zeros(NUM_GENRES, dtype=float)
        if 0 <= self.last_genre < NUM_GENRES:
            last_genre_oh[self.last_genre] = 1.0

        # Demographic one-hots
        age_oh = np.zeros(NUM_AGE_HOT, dtype=float)
        age_oh[AGE_GROUPS.index(self.age_group)] = 1.0
        gender_oh = np.array([1.0, 0.0]) if self.gender == "M" else np.array([0.0, 1.0])

        # Normalised scalars
        norm_total  = min_max(self.total_watched, 0, 100)         # minutes
        norm_step   = min_max(self.step_count, 0, self.max_steps)
        norm_length = min_max(self.last_length, 60, 180)

        # Recent-genres one-hot history
        recent_genre_oh = np.zeros(NUM_GENRES * self.recent_genres_len, dtype=float)
        for i, g in enumerate(self.recent_genres):
            if 0 <= g < NUM_GENRES:
                recent_genre_oh[i * NUM_GENRES + g] = 1.0

        avg_watch_fraction = (
            np.mean(self.watch_fractions) if self.watch_fractions else 0.0
        )

        # : clip view-ratio to [0,1] so values never explode ---
        self.genre_view_ratio = np.minimum(self.genre_view_ratio, 1.0)

        state = np.concatenate([
            self.genre_view_ratio,                 # 10
            np.array([norm_total, norm_step, norm_length]),  # 3
            last_genre_oh,                         # 10
            age_oh,                                # 8
            gender_oh,                             # 2
            recent_genre_oh,                       # 10 × recent_genres_len
            np.array([avg_watch_fraction]),        # 1
        ])

        return state.astype(np.float32)

    # --------------------------------------------------------
    # Action-mask helper
    # --------------------------------------------------------
    def get_valid_actions(self) -> np.ndarray:
        """
        Return a boolean mask (length == action_dim) indicating which movies are
        selectable given age restrictions.
        """
        mask = np.ones(self.action_dim, dtype=bool)
        restricted = AGE_RESTRICTIONS.get(self.age_group, set())
        for mid, movie in enumerate(self.movies):
            if GENRES[movie["genre"]] in restricted:
                mask[mid] = False
        return mask
