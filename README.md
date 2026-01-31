# Homework 0: Cartpole RL Exploration

**Authors**  
- Chantouch Orungrote (66340500011)  
- Sasish Keawsing (66340500076)

## Overview
This homework explores the Isaac-Cartpole-v0 environment in Isaac Sim using reinforcement learning. Part 1 examines the default agent setup, training, and environment configuration. Part 2 conducts targeted experiments by scaling individual reward weights to analyze their impact on agent behavior and performance. Part 3 maps core reinforcement learning fundamentals to the Cartpole task.

## Part 1: Look at Cartpole RL Agent

### 1.1 Train the Cartpole RL Agent
The Isaac-Cartpole-v0 task is trained in **Isaac Sim**. The environment features a cart-pole system on a grid with physics simulation.

### 1.2 Visualize the Result
Training metrics (cumulative reward, episode length, etc.) are monitored via **TensorBoard**.

### 1.3 Questionnaires

**Question 1:** Where to edit environment configuration, action space, observation space, reward function, or termination condition?  
**Answer:** `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/cartpole/cartpole_env_cfg.py`  
- Environment config: `CartpoleEnvCfg` class  
- Action space: `ActionsCfg`  
- Observation space: `ObservationsCfg`  
- Rewards: `RewardCfg`  
- Terminations: `TerminationsCfg`

**Question 2:** Action space and observation space?  
**Answer:**  
- **Action space**: Single continuous value (joint effort force on `slide_to_cart`, scaled by 100.0).  
- **Observation space**: [$x$, $x_dot$, $theta$, $theta_dot$].

**Question 3:** Episode termination conditions?  
**Answer:**  
- Time out (max episode length reached)  
- Cart out of bounds (|slider_to_cart| > 3.0)

**Question 4:** Number of reward terms?  
**Answer:** 5 reward terms (default weights shown):  
- +1.0 Constant running reward (alive/staying alive)  
- -2.0 Termination penalty  
- -1.0 Pole position penalty (primary task)  
- -0.01 Cart velocity penalty (shaping)  
- -0.005 Pole angular velocity penalty (shaping)

---

## Part 2: Playing with Cartpole RL Agent

Baseline model at 448,000 steps is used. Experiments adjust one reward weight at a time (0.0 or ×10) while keeping others default. Each experiment observes a single episode of ~300 timesteps.

### 2.1 Experiment 1: Staying Alive Reward (+1.0)
**Hypothesis**: 0.0 → immediate failure; +1.0 → balanced survival; +10.0 → chaotic longevity focus.  
**Variables**:  
| Independent | Dependent | Control |
|-------------|-----------|---------|
| Alive Reward Weight (W_alive) | Total Reward, Physical Stability, Stabilization Time | Model (448k), 300 timesteps, other weights |

**Results & Analysis**:  
- **0.0**: Flat reward near zero, immediate failure, cart slides left.  
- **+1.0 (baseline)**: Steady learning, stable upright pole, smooth cart motion.  
- **+10.0**: High-variance/spiky reward, erratic motion, prioritizes survival over stability (abandons pole to avoid bounds).

**Conclusion**: Alive reward is the core motivation; extreme scaling leads to unstable policies.

### 2.2 Experiment 2: Termination Penalty (-2.0)
**Hypothesis**: 0.0 → reduced failure avoidance; -2.0 → balanced; -20.0 → conservative/fast stabilization.  

**Results & Analysis**:  
- **0.0**: Fast initial rise but unstable, less refined motion, wider swings.  
- **-2.0 (baseline)**: Consistent learning, balanced pressure.  
- **-20.0**: Volatile training, extremely conservative policy, rigid/fast pole locking.

**Conclusion**: Termination penalty drives risk sensitivity; higher values yield conservative control.

### 2.3 Experiment 3: Pole Position Penalty (-1.0)
**Hypothesis**: 0.0 → pole free-fall; -1.0 → upright focus; -10.0 → aggressive adjustments.

**Results & Analysis** (Note: results partially contradicted hypothesis):  
- **0.0**: High numerical reward (false positive from alive term), chaotic oscillations, active but failed stabilization due to termination pressure.  
- **-1.0 (baseline)**: Steady state, good balance.  
- **-10.0**: Steepest recovery, most stable plots, minimal steady-state error.

**Conclusion**: Pole penalty dominates upright task; null weight still forces some control via termination avoidance.

### 2.4 Experiment 4: Cart Velocity Penalty (-0.01)
**Hypothesis**: 0.0 → variable/aggressive moves; -0.01 → smooth; -0.1 → overly constrained.

**Results & Analysis**:  
- **0.0**: Highest reward, aggressive over-corrections, noisy velocity.  
- **-0.01 (baseline)**: Standard trajectory, subtle smoothing.  
- **-0.1**: Longest recovery, highly conservative (near-stationary cart).

**Conclusion**: Velocity shaping encourages smooth motion; excessive penalty restricts necessary corrections.

### 2.5 Experiment 5: Pole Angular Velocity Penalty (-0.005)
**Hypothesis**: 0.0 → oscillations; -0.005 → balanced dampening; -0.05 → overly damped/slow response.

**Results & Analysis**:  
- **0.0**: Fast convergence but noisy control loop, frequent oscillations.  
- **-0.005 (baseline)**: Steady, balanced speed/stability.  
- **-0.05**: Lower reward, highly damped/slow motion, risk of slow recovery.

**Conclusion**: Angular velocity shaping smooths motion; extremes trade responsiveness for calmness.

---

## Part 3: Mapping RL Fundamentals

### 3.1 Questionnaires

**Question 1**: What is reinforcement learning and its components? (Cartpole examples)  
**Answer**: RL is a machine learning paradigm where an agent learns to maximize cumulative reward through interaction.  
Components (see diagrams):  
- **Agent**: Controller deciding cart force.  
- **State (S_t)**: (x, ẋ, θ, θ̇).  
- **Action (A_t)**: Horizontal force on cart.  
- **Environment**: Physics (cart, pole, gravity).  
- **Reward (R_{t+1})**: Positive for upright, penalties for deviation/fall.  
- **Next State (S_{t+1})**: Updated physics state.

**Question 2**: Difference between reward, return, and value function?  

| Concept          | Temporal Scope      | Provided By | Function              |
|------------------|---------------------|-------------|-----------------------|
| Reward ($R_{t+1}$) | One step            | Environment | Feedback signal       |
| Return (G_t)     | Overall episode     | Rewards     | Optimization objective|
| Value (V^π(s))   | Expected future     | Agent       | Decision guidance     |

**Question 3**: Mathematical functions (input/output)?

| Concept              | Function                          | Input(s)              | Output                     |
|----------------------|-----------------------------------|-----------------------|----------------------------|
| Policy               | π(s) = a                          | State                 | Action                     |
| Environment State    | S_t = O_t                         | Observation           | State                      |
| Agent State Update   | S_{t+1} = u(...)                  | History               | Next State                 |
| Value Function       | V^π(s) = E[G_t \| s, π]           | State                 | Expected Return            |
| Model (Transition)   | P(s,a,s')                         | State, Action         | Next State (prob)          |
| Model (Reward)       | R(s,a)                            | State, Action         | Expected Reward            |

## Insights
- Reward design directly shapes agent behavior: survival terms drive longevity, penalties enforce constraints/stability.  
- Extreme scaling often produces unstable or overly conservative policies.  
- Balanced defaults yield smooth, effective balancing.
