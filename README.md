# FRA503 – Deep Reinforcement Learning  
## Homework 1: Multi-Armed Bandit

### Author
Student Solution

---

## Objective

Implement a **multi-armed bandit framework from scratch** and compare:
- **Epsilon-Greedy**
- **Upper Confidence Bound (UCB)**

using reward-based performance and convergence behavior.

---

## Part 1: Multi-Armed Bandit Setup

### Bandit Environment
- Each arm follows a **Bernoulli reward distribution**
- Reward ∈ {0, 1}
- Hidden success probabilities are predefined

Implemented in:
- `Bandit` class
  - Constructor initializes arms and probabilities
  - `pull(action)` returns stochastic reward

### Agent Structure
Both agents maintain:
- Action-value estimates `Q(a)`
- Action counts `N(a)`

---

## Part 2: Epsilon-Greedy Algorithm

- Exploration probability: **ε = 0.1**
- Action selection:
  - Explore with probability ε
  - Exploit best estimated action otherwise
- Q-values updated using **incremental average**

### Result
- Gradually converges to the optimal arm
- Slower convergence due to constant exploration

Generated plots:
- `reward_comparison.png`
- `epsilon_greedy_actions.png`
- `cumulative_reward.png`

---

## Part 3: Upper Confidence Bound (UCB)

- Exploration term:  
  \[
  Q(a) + c \sqrt{\frac{\ln t}{N(a)}}
  \]
- Encourages systematic exploration
- Each arm is selected at least once

### Result
- Faster and more stable convergence than epsilon-greedy
- Reduced unnecessary exploration

Generated plots:
- `ucb_actions.png`
- `action_comparison.png`

---

## Convergence Analysis

- **UCB converges earlier** than epsilon-greedy
- Convergence observed when:
  - Optimal arm dominates action selection
  - Average reward plateaus

Empirically:
- UCB converges **within fewer timesteps**
- Epsilon-greedy converges slower due to persistent exploration

---

## How to Run
```bash
python bandit_hw1.py
