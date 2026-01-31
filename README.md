# Homework 0: Cartpole RL Agent

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

### Config file
```
IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/cartpole/cartpole_env_cfg.py
```

### Action Space
- Horizontal force applied to the cart joint (`joint_effort`)

### Observation Space
`[x,x_dot,theta,theta_dot]`
  
### Termination Conditions
- Episode timeout (300 timesteps)
- cart_pos (`x`) outside `[-3.0, 3.0]`

### Reward Terms (Default values)
- Alive reward (+1.0)
- Termination penalty (-2.0)
- Pole position penalty (-1.0)
- Cart velocity penalty (-0.01)
- Pole angular velocity penalty (-0.005)

---

## Experiments
Agent behavior changes directly with reward design, confirming the **Reward Hypothesis**.
- **Baseline model:** Default model @448,000 timesteps
- **Single episode:** 300 timesteps

Experimental Design:
- `0.0` (non-significant)
- Default value
- `10×` (highly significant)

Reward Weight Experimental Set:
1. Alive reward `[0.0, 1.0, 10.0]`
2. Termination penalty `[0.0, -2.0, -20.0]`
3. Pole position penalty `[0.0, -1.0, -10.0]`
4. Cart velocity penalty `[0.0, -0.01, -0.1]`
5. Pole angular velocity penalty `[0.0, -0.005, -0.05]`

---

## Report
All results, plots, and analysis are available in:
```
HW0_DRL_6611_6676.pdf
```
