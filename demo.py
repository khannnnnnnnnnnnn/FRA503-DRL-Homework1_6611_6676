"""
Quick Demo Script
Demonstrates the multi-armed bandit framework with a simple example
"""

import numpy as np
import matplotlib.pyplot as plt
from bandit import Bandit
from agent import Agent


def demo_single_run():
    """
    Run a simple demonstration with both algorithms.
    """
    print("="*60)
    print("MULTI-ARMED BANDIT DEMO")
    print("="*60)
    
    # Setup
    n_bandits = 5
    n_steps = 1000
    seed = 42
    
    print(f"\nConfiguration:")
    print(f"  - Number of bandits: {n_bandits}")
    print(f"  - Time steps: {n_steps}")
    print(f"  - Random seed: {seed}")
    
    # Initialize bandit
    bandit = Bandit(n_bandits, reward_type='gaussian', seed=seed)
    
    print(f"\nTrue bandit values:")
    for i, value in enumerate(bandit.get_true_values()):
        marker = " ← OPTIMAL" if i == bandit.get_optimal_action() else ""
        print(f"  Bandit {i}: {value:.3f}{marker}")
    
    # Test Epsilon-Greedy
    print("\n" + "-"*60)
    print("Testing Epsilon-Greedy (ε=0.1)")
    print("-"*60)
    
    agent_eg = Agent(n_bandits, algorithm='epsilon-greedy', epsilon=0.1)
    rewards_eg = []
    
    for step in range(n_steps):
        action = agent_eg.select_action()
        reward = bandit.pull(action)
        agent_eg.update(action, reward)
        rewards_eg.append(reward)
    
    print(f"\nResults after {n_steps} steps:")
    print(f"  Learned Q-values: {agent_eg.Q}")
    print(f"  Action counts: {agent_eg.N.astype(int)}")
    print(f"  Average reward: {np.mean(rewards_eg):.3f}")
    print(f"  Cumulative reward: {agent_eg.get_cumulative_reward():.3f}")
    
    action_percentages = agent_eg.get_action_percentages()
    print(f"\n  Action selection percentages:")
    for i, pct in enumerate(action_percentages):
        marker = " ← OPTIMAL" if i == bandit.get_optimal_action() else ""
        print(f"    Bandit {i}: {pct:.2f}%{marker}")
    
    # Test UCB
    print("\n" + "-"*60)
    print("Testing UCB (c=2.0)")
    print("-"*60)
    
    # Reset bandit with same seed for fair comparison
    bandit = Bandit(n_bandits, reward_type='gaussian', seed=seed)
    agent_ucb = Agent(n_bandits, algorithm='ucb', ucb_c=2.0)
    rewards_ucb = []
    
    for step in range(n_steps):
        action = agent_ucb.select_action()
        reward = bandit.pull(action)
        agent_ucb.update(action, reward)
        rewards_ucb.append(reward)
    
    print(f"\nResults after {n_steps} steps:")
    print(f"  Learned Q-values: {agent_ucb.Q}")
    print(f"  Action counts: {agent_ucb.N.astype(int)}")
    print(f"  Average reward: {np.mean(rewards_ucb):.3f}")
    print(f"  Cumulative reward: {agent_ucb.get_cumulative_reward():.3f}")
    
    action_percentages = agent_ucb.get_action_percentages()
    print(f"\n  Action selection percentages:")
    for i, pct in enumerate(action_percentages):
        marker = " ← OPTIMAL" if i == bandit.get_optimal_action() else ""
        print(f"    Bandit {i}: {pct:.2f}%{marker}")
    
    # Plot comparison
    print("\n" + "-"*60)
    print("Generating comparison plot...")
    print("-"*60)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Cumulative rewards
    ax = axes[0]
    cumsum_eg = np.cumsum(rewards_eg)
    cumsum_ucb = np.cumsum(rewards_ucb)
    timesteps = np.arange(n_steps)
    
    ax.plot(timesteps, cumsum_eg, label='Epsilon-Greedy (ε=0.1)', linewidth=2)
    ax.plot(timesteps, cumsum_ucb, label='UCB (c=2.0)', linewidth=2)
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('Cumulative Reward Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Running average reward
    ax = axes[1]
    window = 50
    avg_eg = np.convolve(rewards_eg, np.ones(window)/window, mode='valid')
    avg_ucb = np.convolve(rewards_ucb, np.ones(window)/window, mode='valid')
    timesteps_smooth = timesteps[:len(avg_eg)]
    
    ax.plot(timesteps_smooth, avg_eg, label='Epsilon-Greedy (ε=0.1)', linewidth=2)
    ax.plot(timesteps_smooth, avg_ucb, label='UCB (c=2.0)', linewidth=2)
    ax.axhline(y=bandit.get_optimal_value(), color='green', 
               linestyle='--', label='Optimal Value', linewidth=2)
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Average Reward (smoothed)')
    ax.set_title('Average Reward Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/claude/demo_results.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to: demo_results.png")
    
    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    print(f"\n{'Algorithm':<20} {'Cumulative Reward':<20} {'Avg Reward':<15}")
    print("-"*60)
    print(f"{'Epsilon-Greedy':<20} {cumsum_eg[-1]:<20.2f} {np.mean(rewards_eg):<15.3f}")
    print(f"{'UCB':<20} {cumsum_ucb[-1]:<20.2f} {np.mean(rewards_ucb):<15.3f}")
    print(f"{'Optimal (theoretical)':<20} {bandit.get_optimal_value()*n_steps:<20.2f} {bandit.get_optimal_value():<15.3f}")
    
    improvement = ((cumsum_ucb[-1] - cumsum_eg[-1]) / cumsum_eg[-1]) * 100
    print(f"\nUCB improvement over Epsilon-Greedy: {improvement:.2f}%")
    
    print("\n" + "="*60)
    print("DEMO COMPLETED!")
    print("="*60)
    print("\nTo run full experiments, execute: python simulation.py")


if __name__ == "__main__":
    demo_single_run()
