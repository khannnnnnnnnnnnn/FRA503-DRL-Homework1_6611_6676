# Multi-Armed Bandit Framework
## FRA 503: Deep Reinforcement Learning - Homework 1

This implementation provides a complete multi-armed bandit framework with epsilon-greedy and UCB algorithms, designed according to the homework requirements and comprehensive experimental design.

## Project Structure

```
├── bandit.py          # Bandit class with reward distributions
├── agent.py           # Agent class with epsilon-greedy and UCB
├── simulation.py      # Main simulation script with experiments
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install numpy matplotlib seaborn tqdm
```

## Running the Experiments

To run all experiments:
```bash
python simulation.py
```

This will execute three comprehensive experiments:
1. **Experiment 1**: Epsilon-Greedy with ε ∈ {0.01, 0.05, 0.1, 0.2, 0.3}
2. **Experiment 2**: UCB with c ∈ {0.5, 1.0, 2.0, 3.0, 5.0}
3. **Experiment 3**: Direct comparison of optimal parameters

The script will generate:
- `epsilon_greedy_results.png` - Epsilon-greedy analysis plots
- `ucb_results.png` - UCB analysis plots
- `comparison_results.png` - Direct algorithm comparison
- Console output with convergence analysis and statistics

## Experimental Design

### Configuration
- **Number of bandits**: 10
- **Time steps**: 10,000 per run
- **Independent runs**: 100 (for statistical significance)
- **Reward distribution**: Gaussian (Normal) with unit variance

### Experiment 1: Epsilon-Greedy

**Purpose**: Evaluate exploration-exploitation trade-off with different epsilon values

**Independent Variables**:
- Epsilon values: 0.01, 0.05, 0.1, 0.2, 0.3

**Dependent Variables**:
- Average reward per timestep
- Cumulative reward
- Optimal action selection rate
- Cumulative regret

**Hypothesis**:
- Lower ε (0.01-0.05): Higher long-term reward, slower convergence
- Higher ε (0.2-0.3): More exploration, lower cumulative reward
- Optimal ε ≈ 0.1: Best balance

### Experiment 2: UCB

**Purpose**: Evaluate systematic exploration based on uncertainty

**Independent Variables**:
- Confidence parameter c: 0.5, 1.0, 2.0, 3.0, 5.0

**Dependent Variables**:
- Average reward per timestep
- Cumulative reward
- Optimal action selection rate
- Cumulative regret

**Hypothesis**:
- UCB achieves higher cumulative reward than epsilon-greedy
- UCB converges faster due to systematic exploration
- Optimal c ≈ 2.0

### Experiment 3: Algorithm Comparison

**Purpose**: Direct comparison under optimal configurations

**Algorithms**:
- Epsilon-Greedy (ε = 0.1)
- UCB (c = 2.0)

**Expected Results**:
- UCB should outperform in cumulative reward
- UCB should show faster convergence
- Lower variance in UCB performance

## Code Structure

### Bandit Class (`bandit.py`)

```python
Bandit(n_bandits, reward_type='gaussian', seed=None)
```

**Features**:
- Initializes n bandit arms with hidden reward distributions
- `pull(action)`: Returns reward for selected arm
- `get_optimal_action()`: Returns index of best arm
- `get_true_values()`: Returns true expected values

### Agent Class (`agent.py`)

```python
Agent(n_actions, algorithm='epsilon-greedy', epsilon=0.1, ucb_c=2.0)
```

**Features**:
- Maintains Q-value estimates and action counts
- `select_action()`: Chooses action using specified algorithm
- `update(action, reward)`: Updates Q-values based on reward
- Supports both epsilon-greedy and UCB algorithms

### Simulation Script (`simulation.py`)

**Key Functions**:
- `run_single_experiment()`: Single run with specified parameters
- `run_multiple_experiments()`: Multiple runs for statistical analysis
- `plot_results()`: Generates comprehensive visualization
- `analyze_convergence()`: Determines convergence timestep

## Output Interpretation

### Plots Generated

Each experiment produces 4 subplots:

1. **Average Reward over Time**: Shows learning progress (smoothed)
2. **Cumulative Reward**: Total reward accumulated
3. **Optimal Action Selection Rate**: % of time best arm is pulled
4. **Cumulative Regret**: Difference from optimal performance

### Convergence Analysis

The script analyzes when each configuration converges, defined as:
- Maintaining ≥95% optimal action selection
- For 100 consecutive timesteps

**Answering the homework question**: "Which episode did it converge?"
The console output will show:
```
Epsilon    | Converged At | Final Optimal Rate
--------------------------------------------------------
0.01       | Step    XXXX | XX.XX%
0.05       | Step    XXXX | XX.XX%
...
```

## Customization

### Modify Experimental Parameters

Edit these variables in `simulation.py`:

```python
N_BANDITS = 10        # Number of arms
N_STEPS = 10000       # Timesteps per run
N_RUNS = 100          # Independent runs

epsilon_values = [0.01, 0.05, 0.1, 0.2, 0.3]
ucb_c_values = [0.5, 1.0, 2.0, 3.0, 5.0]
```

### Run Single Test

```python
from bandit import Bandit
from agent import Agent

# Create bandit and agent
bandit = Bandit(n_bandits=10, seed=42)
agent = Agent(n_actions=10, algorithm='epsilon-greedy', epsilon=0.1)

# Run for 1000 steps
for step in range(1000):
    action = agent.select_action()
    reward = bandit.pull(action)
    agent.update(action, reward)

print(f"Final Q-values: {agent.Q}")
print(f"Action counts: {agent.N}")
```

## Expected Runtime

- Single run (10k steps): <1 second
- Full experiment (100 runs × 5 parameters): ~2-5 minutes per algorithm
- Total runtime for all experiments: ~10-15 minutes

## Key Results to Look For

1. **Epsilon-Greedy**:
   - ε = 0.01 should show slow but stable convergence
   - ε = 0.1 should show good balance
   - ε = 0.3 should show high regret due to over-exploration

2. **UCB**:
   - Should outperform epsilon-greedy in cumulative reward
   - c = 2.0 typically optimal
   - Faster convergence than epsilon-greedy

3. **Comparison**:
   - UCB typically achieves 5-10% better cumulative reward
   - UCB converges 20-30% faster
   - UCB shows lower variance across runs

## Homework Requirements Checklist

✅ **Part 1**: Multi-armed Bandit Framework
- [x] Bandit class with constructor and pull function
- [x] Agent class with constructor and update function
- [x] Simulation script for experiments

✅ **Part 2**: Epsilon-Greedy Implementation
- [x] Algorithm implemented in Agent class
- [x] Results analyzed with multiple epsilon values
- [x] Plots: timesteps vs reward for each configuration

✅ **Part 3**: UCB Implementation
- [x] Algorithm implemented in Agent class
- [x] Results analyzed with multiple c values
- [x] Plots: timesteps vs reward for each configuration
- [x] Convergence analysis (answers "which episode did it converge?")

## References

- Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.)
- Chapter 2: Multi-armed Bandits

## Troubleshooting

**Issue**: ModuleNotFoundError
**Solution**: Install requirements: `pip install -r requirements.txt`

**Issue**: Plots not showing
**Solution**: Plots are saved to PNG files automatically. Check your working directory.

**Issue**: Out of memory
**Solution**: Reduce N_RUNS or N_STEPS in simulation.py

## Author

Created for FRA 503: Deep Reinforcement Learning - Homework 1
