# 🎬 RL-Based Movie Recommendation System

This project implements a personalized movie recommendation system using **Deep Reinforcement Learning**. It was developed as part of an advanced academic course in Reinforcement Learning, and demonstrates the application of **Double-Dueling DQN** with **Prioritized Experience Replay (PER)** to sequential recommendation tasks.

---

## 🚀 Project Overview

- **Goal:** Recommend a slate of movies to a user in order to **maximize long-term engagement**.
- **Approach:** The agent interacts with a simulated user, learning an optimal recommendation policy by observing feedback (watch time or slate skip).
- **Environment:** The simulator models user preferences using a **Dirichlet distribution**, genre-based boredom, and novelty effects.
- **Key Techniques:**  
  ✅ Double-DQN to reduce Q-value overestimation  
  ✅ Dueling architecture for stable value/advantage learning  
  ✅ Prioritized Experience Replay for efficient sample usage  
  ✅ Huber loss and soft target updates for training stability

---

## 🧠 Core Concepts

- **State space:** 84-dimensional vector representing user demographics, engagement history, and genre fatigue.
- **Action space:** Recommend a slate of `k=5` movies from a pool of 500.
- **Reward:** Based on watch fraction; penalty for skipped slates.
- **Objective:** Maximize cumulative user engagement across a 30-step episode.

---

## 📂 Project Structure

movierec_rl/
├── env.py # User simulator and environment dynamics
├── dqn.py # Dueling DQN agent with PER support
├── replay_buffer.py # Experience replay with SumTree-based PER
├── sum_tree.py # Efficient data structure for PER
├── train.py # Training loop and metrics logging
├── evaluate.py # Benchmarking vs. heuristic baselines
├── utils.py # Utilities (seeding, logging, config parsing)
├── config.yaml # Main configuration file

---

## 📊 Key Results

| Method      | Avg Return | Engagement ≥30% | Skip Rate | Genre Diversity |
|-------------|------------|------------------|-----------|-----------------|
| **DQN**     | **3.03**   | **40%**          | 33%       | Moderate (1.11) |
| Random      | 1.23       | 20%              | 37%       | High (2.23)     |
| Static      | 0.28       | 0%               | 33%       | None (0.00)     |

🟢 The DQN agent significantly outperformed all baselines in return and engagement while maintaining reasonable diversity.

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/your-username/rl-movie-rec.git
cd rl-movie-rec

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
📌 Highlights
Fully modular architecture, easy to extend or swap components.

Supports both heuristic and learned recommendation policies.

Includes structured evaluation for watch time, skip rate, and genre diversity.

