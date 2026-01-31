# Homework 1: Multi-Armed Bandit

### Author
- Chantouch Orungrote (6340500011)
- Sasish Kaewsing (6340500076)
- 
---

## Goal of this homework

Implement a classic multi-armed bandit environment and two well-known algorithms from scratch:

- **ε-greedy**  
- **UCB** (Upper Confidence Bound)

Then run systematic experiments, visualize behavior, and answer the question:

> “Which timestep (episode) did the algorithm converge?”

## Implemented Components

### 1. Bandit environment (`bandit.py`)

- Supports two reward types:
  - **Bernoulli** → binary rewards (0 or 1) with fixed probabilities
  - **Gaussian** → continuous rewards ~ 𝒩(μ, 1) with random means sampled anew each run
- Method `pull(arm)` returns a stochastic reward
- Properties `best_arm` and `best_value` give access to the true optimum

### 2. Agent (`agent.py`)

- Unified class supporting both strategies
- **ε-greedy**: explores randomly with probability ε, otherwise picks current best arm
- **UCB**: uses the standard UCB1 formula  
  `UCB(a) = Q(a) + c × √(ln(t) / N(a))`
- Incremental sample-average update rule:  
  `Q ← Q + (r - Q) / N`

### 3. Experiment runner (`main.py`)

- Runs 3000 independent episodes per configuration
- Compares multiple ε and c values
- Computes and averages:
  - instantaneous & cumulative reward
  - % optimal action
  - cumulative regret
- Estimates convergence time (first timestep where 50-step rolling average of optimal action ≥ 85% and stays ≥ 85% until the end)

## How to run

```bash
# (optional but recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

pip install numpy matplotlib

python main.py
```

Expected runtime: ≈ 40–100 seconds (depending on hardware and N_RUNS)

## Generated files

All plots are saved under `figures/`

```
figures/
├── bernoulli/
│   ├── average_reward.png
│   ├── percent_optimal.png           ← shows convergence lines
│   ├── cumulative_regret.png
│   ├── action_selection_eps_best.png
│   └── action_selection_ucb_best.png
└── gaussian/
    └── (same set of plots)
```

## Typical convergence behavior (example from runs)

**Bernoulli** (fixed probabilities, best arm p=0.80)

- ε = 0.1          → converges ≈ 220–300 steps
- UCB c = 2.0      → converges ≈ 110–170 steps

**Gaussian** (random means each run)

- ε = 0.1          → converges ≈ 150–260 steps
- UCB c = 2.0      → converges ≈ 70–140 steps

→ UCB usually converges noticeably faster and reaches higher final performance.

## Quick parameter tuning guide

Change values in `config.py`:

```python
N_RUNS          = 2000          # more → smoother curves, longer runtime
N_STEPS         = 1000           # how long each episode runs
EPSILON_VALS    = [0.0, 0.01, 0.05, 0.1, 0.5]
UCB_C_VALS      = [0.0, 0.5, 1.0, 2.0, 5.0]
CONVERGENCE_THRESHOLD = 0.85 
ROLLING_WINDOW        = 50      # smoothing window for convergence detection
```
