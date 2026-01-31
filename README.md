# FRA 503: Deep Reinforcement Learning - Homework 0

## Abstract
This project investigates the impact of reward shaping on the behavior and performance of a reinforcement learning (RL) agent in the **Isaac-Cartpole-v0** environment. Using a trained baseline model, individual reward terms are systematically scaled to analyze their influence on stability, control strategy, and learning dynamics. The results demonstrate that agent behavior is a direct consequence of reward design, validating the reward hypothesis in a continuous control setting.

---

## Environment
- **Simulator**: NVIDIA Isaac Sim  
- **Task**: Isaac-Cartpole-v0  
- **Observation Space**: 4D continuous state  
- **Action Space**: 1D continuous force input  

---

## Methodology
A trained policy at **448k steps** is used as a baseline.  
Each experiment modifies a single reward term while keeping others fixed.

**Evaluated Reward Terms**
1. Survival (alive) reward  
2. Termination penalty  
3. Pole angle penalty  
4. Cart velocity penalty  
5. Pole angular velocity penalty  

Performance is evaluated using:
- Mean episode reward  
- Mean episode length  
- Physical stability and motion characteristics  

---

## Key Findings
- Reward weights strongly dictate learned control behavior  
- Removing essential rewards can lead to misleading high returns  
- Excessive penalties or rewards produce unstable or overly conservative policies  
- Balanced reward shaping yields smooth and physically stable control  

---

## Conclusion
The study confirms that effective reinforcement learning policies depend critically on reward formulation. Properly balanced reward functions are essential for stable, interpretable, and robust agent behavior in physics-based environments.

---

## Authors
- Chantouch Orungrote  
- Sasish Keawsing  

## Course
FRA503 – Deep Reinforcement Learning
