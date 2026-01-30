# Quick Start Guide
## Multi-Armed Bandit Framework - FRA 503 Homework 1

## Files Included

- `bandit.py` - Bandit class implementation
- `agent.py` - Agent class with epsilon-greedy and UCB algorithms
- `simulation.py` - Full experimental simulation (main file)
- `demo.py` - Quick demonstration script
- `multi_armed_bandit.ipynb` - Jupyter notebook for interactive exploration
- `requirements.txt` - Python dependencies
- `README.md` - Comprehensive documentation

## Quick Setup (3 steps)

### Step 1: Install Dependencies
```bash
pip install numpy matplotlib seaborn tqdm
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

### Step 2: Run Quick Demo (5 seconds)
```bash
python demo.py
```

This will:
- Test both algorithms on a 5-armed bandit
- Show convergence and performance metrics
- Generate a comparison plot (`demo_results.png`)

### Step 3: Run Full Experiments (~10-15 minutes)
```bash
python simulation.py
```

This will run comprehensive experiments:
- Experiment 1: Epsilon-Greedy with 5 different ε values
- Experiment 2: UCB with 5 different c values  
- Experiment 3: Direct comparison of optimal configurations

Each experiment runs 100 independent trials for statistical significance.

## Expected Output

### Console Output
```
============================================================
MULTI-ARMED BANDIT EXPERIMENTS
============================================================
Configuration:
  - Number of bandits: 10
  - Time steps per run: 10000
  - Independent runs: 100
============================================================

EXPERIMENT 1: EPSILON-GREEDY ALGORITHM
...
CONVERGENCE ANALYSIS
Epsilon    | Converged At | Final Optimal Rate
------------------------------------------------------------
0.01       | Step   XXXX | XX.XX%
0.05       | Step   XXXX | XX.XX%
...
```

### Generated Plots
1. `epsilon_greedy_results.png` - 4 subplots showing:
   - Average reward over time
   - Cumulative reward
   - Optimal action selection rate
   - Cumulative regret

2. `ucb_results.png` - Same 4 subplots for UCB

3. `comparison_results.png` - Direct comparison between optimal parameters

4. `demo_results.png` - Quick demo comparison

## Interactive Exploration

For interactive experimentation, use the Jupyter notebook:

```bash
jupyter notebook multi_armed_bandit.ipynb
```

The notebook allows you to:
- Modify parameters on the fly
- Visualize results immediately
- Test different configurations
- Understand the algorithms step-by-step

## Homework Questions Answered

### Part 1: Setting up Multi-armed Bandit ✓
- **Bandit class**: `bandit.py` (constructor with n bandits, pull function)
- **Agent class**: `agent.py` (constructor with learnable parameters, update function)
- **Simulation script**: `simulation.py` (runs experiments)

### Part 2: Epsilon-greedy algorithm ✓
- **Implementation**: Lines 71-84 in `agent.py`
- **Analysis**: `simulation.py` lines 174-226 (Experiment 1)
- **Plots**: Timesteps vs reward generated in `epsilon_greedy_results.png`

### Part 3: UCB ✓
- **Implementation**: Lines 86-108 in `agent.py`
- **Analysis**: `simulation.py` lines 228-270 (Experiment 2)  
- **Plots**: Timesteps vs reward generated in `ucb_results.png`
- **Convergence answer**: Console output shows "Converged at step: XXXX"

## Customization

### Change number of bandits
Edit `simulation.py` line 370:
```python
N_BANDITS = 10  # Change to desired number
```

### Change number of steps
Edit `simulation.py` line 371:
```python
N_STEPS = 10000  # Change to desired length
```

### Test different parameter values
Edit `simulation.py` lines 387 and 410:
```python
epsilon_values = [0.01, 0.05, 0.1, 0.2, 0.3]  # Add/remove values
ucb_c_values = [0.5, 1.0, 2.0, 3.0, 5.0]      # Add/remove values
```

### Reduce runtime for testing
Edit `simulation.py` line 372:
```python
N_RUNS = 10  # Reduce from 100 to 10 for faster testing
```

## Troubleshooting

**Problem**: `ModuleNotFoundError: No module named 'numpy'`  
**Solution**: Install dependencies: `pip install -r requirements.txt`

**Problem**: "Permission denied" error  
**Solution**: Make sure you have write permissions in the directory

**Problem**: Plots not showing  
**Solution**: Plots are automatically saved as PNG files. Check your directory for:
- `epsilon_greedy_results.png`
- `ucb_results.png`
- `comparison_results.png`

**Problem**: Script takes too long  
**Solution**: Reduce `N_RUNS` in `simulation.py` (line 372) from 100 to 10 or 20

## Understanding the Results

### Convergence
Convergence is defined as maintaining ≥95% optimal action selection for 100 consecutive steps.

### Key Metrics
- **Average Reward**: How well the agent is performing moment-to-moment
- **Cumulative Reward**: Total reward accumulated (higher is better)
- **Optimal Action Rate**: % of times the best arm was pulled (higher is better)
- **Regret**: Difference from optimal performance (lower is better)

### Expected Findings
1. **Epsilon-Greedy**: Lower ε (0.01-0.05) achieves highest cumulative reward but slower convergence
2. **UCB**: Typically outperforms epsilon-greedy with c ≈ 2.0
3. **Comparison**: UCB converges 20-30% faster and achieves 5-10% higher reward

## Need Help?

1. Read the comprehensive `README.md` for detailed explanations
2. Try the `demo.py` script first to verify installation
3. Use the Jupyter notebook for step-by-step understanding
4. Check code comments for implementation details

## Submission Checklist

For homework submission, include:
- [ ] All Python files (`.py`)
- [ ] Generated plots (`.png` files)
- [ ] Console output (copy-paste or screenshot)
- [ ] Convergence analysis results
- [ ] Brief written analysis of findings

Good luck with your homework! 🎰
