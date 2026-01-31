# Homework 1: Multi-Armed Bandit

**Authors**  
- Chantouch Orungrote (6340500011)  
- Sasish Kaewsing (6340500076)

## Goal
Implement a multi-armed bandit framework from scratch, including ε-greedy and UCB algorithms, run systematic experiments on both Bernoulli and Gaussian environments, visualize performance, and determine convergence time.

---

### Part 1: Setup and Run

1. Open terminal in the project folder
2. Install dependencies:
   ```bash
   pip install numpy matplotlib
   ```
3. Run the experiment:
   ```bash
   python main.py
   ```
   
---

### Part 2: Parameter Definition

| Parameter               | Default | Description |
|-------------------------|---------|-------------|
| N_ARMS                  | 10      | Number of arms |
| N_RUNS                  | 2000    | Number of independent runs (for averaging) |
| N_STEPS                 | 1000    | Timesteps per experiment |
| REWARD_TYPES            | ["bernoulli", "gaussian"] | Reward distributions to compare |
| EPSILON_VALS            | [0.0, 0.01, 0.05, 0.1, 0.5] | Exploration rates for ε-greedy |
| UCB_C_VALS              | [0.0, 0.5, 1.0, 2.0, 5.0] | Exploration constants for UCB |
| CONVERGENCE_THRESHOLD   | 0.85    | Threshold for convergence (85% optimal action) |
| ROLLING_WINDOW          | 50      | Window size for rolling average |

**Convergence definition**:  
The first timestep where the 50-step rolling average of % optimal action ≥ 85% and remains ≥ 85% until the end.

---

### Part 3: Config

All experiment configurations are defined in `config.py`:

```python
# Main configurations
EPSILON_VALS = [0.0, 0.01, 0.05, 0.1, 0.5]
UCB_C_VALS   = [0.5, 1.0, 1.5, 2.0, 5.0]

CONFIGS = (
    [("eps", {"epsilon": eps}) for eps in EPSILON_VALS] +
    [("ucb", {"c": c}) for c in UCB_C_VALS]
)
```
---

### Part 4: Where is it saved and structure of files

All outputs are saved in the `figures/` folder, separated by reward type:

```
figures/
├── bernoulli/
│   ├── average_reward.png
│   ├── percent_optimal.png           ← with convergence markers
│   ├── cumulative_regret.png
│   ├── action_selection_eps_best.png
│   └── action_selection_ucb_best.png
└── gaussian/
    ├── average_reward.png
    ├── percent_optimal.png
    ├── cumulative_regret.png
    ├── action_selection_eps_best.png
    └── action_selection_ucb_best.png
```

- `percent_optimal.png` shows convergence points with vertical dashed lines  
- All plots are high-resolution (dpi=180) and publication-ready
