# Multi-Armed Bandit - Homework 1

## Overview
This project implements and compares two classic reinforcement learning algorithms for solving the multi-armed bandit problem: **Epsilon-Greedy** and **Upper Confidence Bound (UCB)**.

**What is Multi-Armed Bandit?**
Imagine you're at a casino with 10 slot machines (bandits). Each machine has a different hidden probability of winning. Your goal is to figure out which machine is best while maximizing your total winnings. Should you keep trying the machine that seems best (exploit) or try other machines to see if they're better (explore)?

---

## Part 1: Setup

### Prerequisites
- Python 3.7 or higher
- Required packages:
  - numpy
  - matplotlib

### Installation

1. **Install Python packages:**
```bash
pip install numpy matplotlib
```

2. **Download the code:**
Save the provided Python script as `bandit_experiment.py`

3. **Verify setup:**
```bash
python --version  # Should show Python 3.7+
python -c "import numpy, matplotlib; print('Setup OK!')"
```

### Project Structure
```
project/
├── bandit_experiment.py    # Main script
└── output/                 # Generated plots (created automatically)
    ├── reward_comparison.png
    ├── epsilon_greedy_actions.png
    ├── ucb_actions.png
    ├── action_comparison.png
    └── cumulative_reward.png
```

---

## Part 2: Understanding Parameters

### Bandit Configuration

#### `bandit_probs` (List of probabilities)
- **What it is:** The hidden win probability for each slot machine (arm)
- **Default:** `[0.10, 0.50, 0.60, 0.80, 0.10, 0.25, 0.60, 0.45, 0.75, 0.65]`
- **Meaning:** 
  - Arm 1 has 10% chance of reward
  - Arm 4 has 80% chance of reward (best arm!)
  - Arm 5 has 10% chance of reward
- **Range:** Each value must be between 0.0 (never wins) and 1.0 (always wins)
- **Example:** `[0.3, 0.7, 0.5]` means 3 arms with 30%, 70%, and 50% win rates

### Experiment Parameters

#### `n_experiments` (Number of independent runs)
- **What it is:** How many times to repeat the entire experiment
- **Default:** `10000`
- **Why it matters:** More experiments = more reliable average results
- **Impact:** 
  - Low (100): Fast but noisy results
  - Medium (1000): Balanced
  - High (10000): Slow but very smooth graphs
- **Recommended:** 1000-10000 for reliable results

#### `n_steps` (Timesteps per experiment)
- **What it is:** How many times the agent pulls arms in each experiment
- **Default:** `500`
- **Why it matters:** More steps = agent has more time to learn which arm is best
- **Impact:**
  - Low (100): Agent barely learns
  - Medium (500): Good balance
  - High (2000): Agent fully converges to best arm
- **Recommended:** 500-1000 for most experiments

### Epsilon-Greedy Parameters

#### `epsilon` (Exploration rate)
- **What it is:** Probability of exploring (trying random arms) vs exploiting (using best known arm)
- **Default:** `0.1`
- **Range:** 0.0 to 1.0
- **Behavior:**
  - `0.0`: Always exploit (never explore) - might miss the best arm
  - `0.1`: 10% explore, 90% exploit - balanced approach
  - `0.5`: 50% explore, 50% exploit - lots of exploration
  - `1.0`: Always explore (never exploit) - completely random
- **Impact:**
  - Too low: Gets stuck on suboptimal arms
  - Too high: Wastes time on bad arms
- **Recommended:** 0.05-0.2 for most problems

### UCB Parameters

#### `c` (Exploration parameter)
- **What it is:** Controls how much the agent favors uncertain arms
- **Default:** `2.0`
- **Range:** Typically 0.5 to 3.0
- **Behavior:**
  - Low (0.5): Conservative, exploits more
  - Medium (2.0): Balanced exploration
  - High (5.0): Aggressive exploration of uncertain arms
- **Formula:** UCB selects arm with highest: `Q(a) + c × √(log(t) / N(a))`
  - `Q(a)`: Average reward of arm a
  - `t`: Total timesteps so far
  - `N(a)`: How many times arm a was pulled
- **Impact:**
  - Too low: May not explore enough
  - Too high: Over-explores even when best arm is known
- **Recommended:** 1.0-3.0 for most problems

### Output Configuration

#### `output_dir` (Output directory)
- **What it is:** Folder where plots are saved
- **Default:** `./output`
- **Behavior:** Automatically created if it doesn't exist

---

## Part 3: Configure Experiments

### How to Modify Parameters

Open `bandit_experiment.py` and find the "Main Execution" section (around line 250):

```python
if __name__ == "__main__":
    # Configuration
    bandit_probs = [0.10, 0.50, 0.60, 0.80, 0.10,
                    0.25, 0.60, 0.45, 0.75, 0.65]
    
    n_experiments = 10000  # <-- Change this
    n_steps = 500          # <-- Change this
    epsilon = 0.1          # <-- Change this
    c = 0.2                # <-- Change this
```

### Example Configurations

#### Quick Test (Fast Results)
```python
n_experiments = 100   # Few experiments
n_steps = 200         # Short episodes
epsilon = 0.1
c = 2.0
```

#### Standard Experiment (Recommended)
```python
n_experiments = 2000  # Good sample size
n_steps = 500         # Enough learning time
epsilon = 0.1
c = 2.0
```

#### Detailed Analysis (Slow but Thorough)
```python
n_experiments = 10000  # Very smooth curves
n_steps = 1000         # Full convergence
epsilon = 0.1
c = 2.0
```

#### Testing Different Strategies

**High Exploration (Epsilon-Greedy)**
```python
epsilon = 0.3  # 30% exploration
```

**Low Exploration (Epsilon-Greedy)**
```python
epsilon = 0.01  # 1% exploration
```

**Aggressive UCB**
```python
c = 5.0  # Explore uncertain arms more
```

**Conservative UCB**
```python
c = 0.5  # Exploit known good arms more
```

### Changing the Bandit Problem

#### Easy Problem (Clear Best Arm)
```python
bandit_probs = [0.1, 0.2, 0.3, 0.9]  # Arm 4 is obviously best
```

#### Hard Problem (Similar Arms)
```python
bandit_probs = [0.48, 0.49, 0.50, 0.51]  # Hard to tell which is best
```

#### Custom Problem
```python
bandit_probs = [0.2, 0.7, 0.3]  # 3 arms with your chosen probabilities
```

---

## Part 4: Run Experiments

### Basic Usage

1. **Run with default settings:**
```bash
python bandit_experiment.py
```

2. **Expected output:**
```
======================================================================
FRA 503 Homework 1: Multi-Armed Bandit
======================================================================
Configuration:
  - Number of arms: 10
  - Arm probabilities: [0.1, 0.5, 0.6, 0.8, 0.1, 0.25, 0.6, 0.45, 0.75, 0.65]
  - Optimal arm: Arm 4 (p=0.8)
  - Experiments: 10000
  - Steps per experiment: 500
======================================================================

Part 2: Running Epsilon-Greedy Algorithm (ε=0.1)
----------------------------------------------------------------------
  Progress: 1000/10000 experiments complete
  Progress: 2000/10000 experiments complete
  ...
```

3. **Check results:**
```bash
ls output/
# Should show 5 PNG files
```

### Advanced Usage

#### Run Multiple Configurations

Create a script `run_experiments.sh`:
```bash
#!/bin/bash

# Test different epsilon values
python bandit_experiment.py  # epsilon=0.1
# Edit epsilon to 0.05 and run again
# Edit epsilon to 0.2 and run again

echo "All experiments complete!"
```

#### Run in Background
```bash
# For long experiments
nohup python bandit_experiment.py > experiment.log 2>&1 &

# Check progress
tail -f experiment.log
```

### Troubleshooting

**Problem:** `ModuleNotFoundError: No module named 'numpy'`
```bash
# Solution:
pip install numpy matplotlib
```

**Problem:** Plots not saving
```bash
# Solution: Check write permissions
mkdir -p output
chmod 755 output
```

**Problem:** Script runs too slowly
```bash
# Solution: Reduce n_experiments
# Change from 10000 to 1000 in the script
```

---

## Part 5: Results

### Understanding the Output

#### 5 Generated Plots

**1. `reward_comparison.png` - Which Algorithm Wins?**
- **Shows:** Average reward over time for both algorithms
- **What to look for:**
  - Which line is higher = better algorithm
  - Steeper slope = faster learning
  - Flat line = fully learned
- **Example:** UCB often starts higher but Epsilon-Greedy catches up

**2. `epsilon_greedy_actions.png` - How Does Epsilon-Greedy Learn?**
- **Shows:** Percentage of times each arm is selected
- **What to look for:**
  - Best arm should dominate (high percentage)
  - Bad arms should decrease over time
  - Some exploration continues (10% random)
- **Example:** Arm 4 (80%) should reach ~90% selection by end

**3. `ucb_actions.png` - How Does UCB Learn?**
- **Shows:** Percentage of times each arm is selected
- **What to look for:**
  - UCB tries each arm multiple times initially
  - Best arm gradually dominates
  - Eventually almost 100% best arm (less exploration than Epsilon-Greedy)
- **Example:** Arm 4 should reach ~95-100% selection

**4. `action_comparison.png` - Side-by-Side Algorithm Behavior**
- **Shows:** Both algorithms' arm selection patterns
- **What to look for:**
  - UCB: Systematic initial exploration
  - Epsilon-Greedy: More random exploration pattern
  - Both converge to best arm eventually

**5. `cumulative_reward.png` - Total Winnings Over Time**
- **Shows:** Total accumulated rewards
- **What to look for:**
  - Steeper slope = earning more rewards
  - Final height = total performance
  - Gap between algorithms = performance difference
- **Example:** UCB might accumulate slightly more total reward

### Interpreting Performance Metrics

#### Terminal Output Analysis

```
Epsilon-Greedy Results:
  - Total reward: 325.45
  - Average reward per step: 0.6509

UCB Results:
  - Total reward: 340.12
  - Average reward per step: 0.6802
```

**What this means:**
- Average reward per step should approach the best arm's probability (0.80)
- UCB's higher average (0.68 vs 0.65) means it found the best arm faster
- Neither reaches 0.80 because they still explore occasionally

### Expected Results

#### With Default Settings (10000 experiments, 500 steps):

**Epsilon-Greedy (ε=0.1):**
- Final best arm selection: ~90%
- Average reward: ~0.65-0.70
- Behavior: Continuous 10% random exploration

**UCB (c=2.0):**
- Final best arm selection: ~95-99%
- Average reward: ~0.70-0.75
- Behavior: Almost exclusively best arm after learning

#### Performance Comparison

| Metric | Epsilon-Greedy | UCB | Winner |
|--------|---------------|-----|--------|
| Early learning (0-100 steps) | Slower | Faster | UCB |
| Late performance (400-500 steps) | Good | Excellent | UCB |
| Consistency | More variation | More stable | UCB |
| Simplicity | Simpler | More complex | Epsilon-Greedy |

### Key Findings

1. **UCB generally performs better** because it:
   - Systematically explores all arms initially
   - Balances exploration and exploitation mathematically
   - Reduces exploration as confidence increases

2. **Epsilon-Greedy is easier to understand** because:
   - Simple: "10% random, 90% best known arm"
   - No complex calculations needed
   - Good enough for many applications

3. **Both algorithms eventually find the best arm** given enough time

### Analyzing Your Results

#### Questions to Ask:

1. **Which arm is selected most at the end?**
   - Should be the arm with highest probability
   - Look at final timesteps in action plots

2. **How quickly does learning happen?**
   - Check when reward curve flattens
   - UCB typically faster initial learning

3. **What's the exploration-exploitation trade-off?**
   - Epsilon-Greedy: Fixed 10% exploration forever
   - UCB: Decreasing exploration over time

4. **Does changing epsilon/c affect performance?**
   - Lower epsilon: Less exploration, might miss best arm
   - Higher epsilon: More exploration, lower immediate rewards
   - Higher c: More UCB exploration, slower to exploit best arm

### Experiment Ideas

**Try these to learn more:**

1. **Change epsilon to 0.05 and 0.2**
   - Does lower epsilon perform better or worse?
   
2. **Make all arms have similar probabilities**
   - Example: `[0.45, 0.50, 0.48, 0.52]`
   - Is it harder for algorithms to find the best?

3. **Increase steps to 2000**
   - Do both algorithms eventually reach 100% best arm selection?

4. **Use only 3 arms**
   - Does learning happen faster with fewer choices?

---

## Summary

This homework demonstrates the fundamental **exploration-exploitation dilemma** in reinforcement learning:
- **Explore** to find better options
- **Exploit** to maximize immediate rewards

Both Epsilon-Greedy and UCB solve this dilemma differently, with UCB typically performing better through systematic, confidence-based exploration.

## Files Generated

After running the experiment, you'll have:
- 5 PNG plots showing algorithm performance
- Terminal output with performance metrics
- All files saved in the `output/` directory

**Total runtime:** ~1-5 minutes depending on your configuration

---

## Quick Reference

```bash
# Basic run
python bandit_experiment.py

# View results
ls output/
open output/*.png  # Mac
xdg-open output/*.png  # Linux
start output\*.png  # Windows
```

**Need help?** Check the terminal output for progress and error messages!
