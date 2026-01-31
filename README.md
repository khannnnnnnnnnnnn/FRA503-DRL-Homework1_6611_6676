# FRA 503: Deep Reinforcement Learning – Homework 0

## Abstract
This project studies how **reward** affects the behavior and performance of a reinforcement learning (RL) agent in the **Isaac-Cartpole-v0** environment.

---

## Environment
- **Simulator**: NVIDIA Isaac Sim  
- **Task**: Isaac-Cartpole-v0  
- **Observation Space**:  
  - Cart position \(x\)  
  - Cart velocity \(\dot{x}\)  
  - Pole angle \(\theta\)  
  - Pole angular velocity \(\dot{\theta}\)
- **Action Space**:  
  - Horizontal force applied to the cart

---

## Methodology
A policy trained for **448,000 steps** is used as the baseline. 
Each experiment changes **one reward term at a time**, while all other rewards remain unchanged.

**Reward Terms Studied**
1. Alive reward  
2. Termination penalty
3. Pole angle penalty
4. Cart velocity penalty  
5. Pole angular velocity penalty  

**Agent Performance Evaluation**
- Average reward per episode
- Episode length
- Cartpole's motion in Isaac Sim

---

## Authors
- Chantouch Orungrote 66340500011
- Sasish Keawsing 66340500076
