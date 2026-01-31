# FRA503 – Deep Reinforcement Learning for Robotics  
## Homework 0: Cartpole RL Agent

**Authors**
- Chantouch Orungrote (66340500011)  
- Sasish Keawsing (66340500076)

---

## Objective

This homework studies the **Isaac-Cartpole-v0** reinforcement learning task to:
- Understand environment structure and RL components
- Analyze action space, observation space, rewards, and termination
- Experiment with **reward weight tuning**
- Relate practical results to **reinforcement learning fundamentals**

---

## Config file
```
IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/cartpole/cartpole_env_cfg.py
```

### Action Space
- Horizontal force applied to the cart joint (`joint_effort`)

### Observation Space
- 4D continuous state:
  - Cart position (x)
  - Cart velocity (ẋ)
  - Pole angle (θ)
  - Pole angular velocity (θ̇)

### Termination Conditions
- Episode timeout
- Cart position outside `[-3.0, 3.0]`

### Reward Terms (Default)
- Alive reward (+1.0)
- Termination penalty (-2.0)
- Pole position penalty (-1.0)
- Cart velocity penalty (-0.01)
- Pole angular velocity penalty (-0.005)

---

## Experiments
**Baseline model:** Default model @448,000 timesteps
**Single episode:** 300 timesteps

Reward-weight experiments were conducted:
1. Alive reward
2. Termination penalty
3. Pole position penalty
4. Cart velocity penalty
5. Pole angular velocity penalty

Each reward was tested at:
- `0.0` (non-significant)
- Default value
- `10×` default (highly significant)

**Key Result:**  
Agent behavior changes directly with reward design, confirming the **Reward Hypothesis**.

---

## Reinforcement Learning Fundamentals
- **Agent:** Cart controller  
- **State:** Cart and pole positions & velocities  
- **Action:** Force applied to cart  
- **Reward:** Scalar feedback  
- **Policy:** \( \pi(s) \rightarrow a \)  
- **Value Function:** Expected return from a state  

---

## Report
Full results, plots, and analysis are available in:
```
report/HW0_DRL_6611_6676.pdf
```
