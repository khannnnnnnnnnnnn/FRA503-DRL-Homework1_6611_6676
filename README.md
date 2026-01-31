# Homework 1: Multi-Armed Bandit

**Authors**  
- Chantouch Orungrote (6340500011)  
- Sasish Kaewsing (6340500076)

## Objectives
Implement a multi-armed bandit framework from scratch, including ε-greedy and UCB algorithms, run systematic experiments with multiple parameter values, visualize performance across all configurations, and analyze convergence behavior and cumulative regret.

---

### Part 1: Setup and Run

1. Open terminal in the project folder
2. Install dependencies:
   ```bash
   pip install numpy matplotlib
   ```
3. Run all experiments:
   ```bash
   python run_all_experiments.py
   ```

**Alternative: Run Individual Experiments**
- Run only Epsilon-Greedy experiments:
  ```bash
  python experiment_epsilon_greedy.py
  ```
- Run only UCB experiments:
  ```bash
  python experiment_ucb.py
  ```

---

### Part 2: Parameter Definition

| Parameter               | Default | Description |
|-------------------------|---------|-------------|
| N_ARMS                  | 10      | Number of arms |
| N_EXPERIMENTS           | 10000   | Number of independent runs (for averaging) |
| N_STEPS                 | 500     | Timesteps per experiment |
| BANDIT_PROBS            | [0.10, 0.50, 0.60, 0.80, 0.10, 0.25, 0.60, 0.45, 0.75, 0.65] | Bernoulli success probabilities for each arm |
| EPSILON_VALS            | [0.0, 0.01, 0.05, 0.1, 0.5] | Exploration rates for ε-greedy |
| UCB_C_VALS              | [0.5, 1.0, 2.0, 3.0, 5.0] | Exploration constants for UCB |

#### **Note: Random seed (42) for all experiments**

---

### Part 3: Configuration

All experiment configurations are defined in the experiment files:

**Epsilon-Greedy** (`experiment_epsilon_greedy.py`):
```python
bandit_probs = [0.10, 0.50, 0.60, 0.80, 0.10,
                0.25, 0.60, 0.45, 0.75, 0.65]
epsilon_values = [0.0, 0.01, 0.05, 0.1, 0.5]
n_experiments = 10000
n_steps = 500
```

**UCB** (`experiment_ucb.py`):
```python
bandit_probs = [0.10, 0.50, 0.60, 0.80, 0.10,
                0.25, 0.60, 0.45, 0.75, 0.65]
c_values = [0.5, 1.0, 2.0, 3.0, 5.0]
n_experiments = 10000
n_steps = 500
```

---

### Part 4: Results Location

All outputs are saved in the `output/` folder, separated by algorithm:

```
output/
├── epsilon_greedy/
│   ├── epsilon_0_0_reward.png           # Individual plots for ε=0.0
│   ├── epsilon_0_0_optimal.png
│   ├── epsilon_0_0_regret.png
│   ├── epsilon_0_01_reward.png          # Individual plots for ε=0.01
│   ├── epsilon_0_01_optimal.png
│   ├── epsilon_0_01_regret.png
│   ├── epsilon_0_05_reward.png          # Individual plots for ε=0.05
│   ├── epsilon_0_05_optimal.png
│   ├── epsilon_0_05_regret.png
│   ├── epsilon_0_1_reward.png           # Individual plots for ε=0.1
│   ├── epsilon_0_1_optimal.png
│   ├── epsilon_0_1_regret.png
│   ├── epsilon_0_5_reward.png           # Individual plots for ε=0.5
│   ├── epsilon_0_5_optimal.png
│   ├── epsilon_0_5_regret.png
│   ├── epsilon_comparison_reward.png    # Comparison of all ε values
│   ├── epsilon_comparison_optimal.png
│   └── epsilon_comparison_regret.png
│
├── ucb/
│   ├── ucb_0_5_reward.png               # Individual plots for c=0.5
│   ├── ucb_0_5_optimal.png
│   ├── ucb_0_5_regret.png
│   ├── ucb_1_0_reward.png               # Individual plots for c=1.0
│   ├── ucb_1_0_optimal.png
│   ├── ucb_1_0_regret.png
│   ├── ucb_2_0_reward.png               # Individual plots for c=2.0
│   ├── ucb_2_0_optimal.png
│   ├── ucb_2_0_regret.png
│   ├── ucb_3_0_reward.png               # Individual plots for c=3.0
│   ├── ucb_3_0_optimal.png
│   ├── ucb_3_0_regret.png
│   ├── ucb_5_0_reward.png               # Individual plots for c=5.0
│   ├── ucb_5_0_optimal.png
│   ├── ucb_5_0_regret.png
│   ├── ucb_comparison_reward.png        # Comparison of all c values
│   ├── ucb_comparison_optimal.png
│   └── ucb_comparison_regret.png
│
└── comparison/
    ├── best_comparison_reward.png       # Best ε vs Best c
    ├── best_comparison_optimal.png
    ├── best_comparison_regret.png
    ├── all_comparison_reward.png        # All ε vs All c
    ├── all_comparison_optimal.png
    └── all_comparison_regret.png
```

**Total plots generated: 42**
- 15 individual Epsilon-Greedy plots (5 ε × 3 metrics)
- 3 Epsilon-Greedy comparison plots
- 15 individual UCB plots (5 c × 3 metrics)
- 3 UCB comparison plots
- 6 algorithm comparison plots

---

### Part 5: Structure

```
.
├── bandit.py                      # Bandit environment (Bernoulli rewards)
├── agents.py                      # ε-greedy and UCB agent implementation
├── utils.py                       # Simulation runners and analysis functions
├── experiment_epsilon_greedy.py   # Epsilon-greedy experiments
├── experiment_ucb.py              # UCB experiments
├── run_all_experiments.py         # Main simulation loop and result generation
└── output/                        # Generated plots
    ├── epsilon_greedy/            # Epsilon-greedy results
    ├── ucb/                       # UCB results
    └── comparison/                # Algorithm comparison plots
```
