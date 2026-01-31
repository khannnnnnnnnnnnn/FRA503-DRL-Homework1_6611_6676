"""
experiment_epsilon_greedy.py

Run Epsilon-Greedy experiments with multiple epsilon values.
Tests ε ∈ {0.0, 0.01, 0.05, 0.1, 0.5}

FRA 503 Homework 1
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from bandit import Bandit
from agents import EpsilonGreedyAgent
from utils import (
    run_multiple_experiments,
    calculate_statistics,
    print_experiment_results,
    print_summary_table,
    find_best_parameter
)


def plot_epsilon_greedy_results(all_results, bandit_probs, n_experiments, output_dir):
    """
    Generate separate plots for epsilon-greedy experiments.
    
    Args:
        all_results (dict): Dictionary of {epsilon: stats}
        bandit_probs (list): Arm probabilities
        n_experiments (int): Number of experiments run
        output_dir (str): Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    epsilon_values = sorted(all_results.keys())
    optimal_prob = np.max(bandit_probs)
    
    # Plot 1: Average Reward Over Time (separate plot for each epsilon)
    for eps in epsilon_values:
        stats = all_results[eps]
        plt.figure(figsize=(10, 6))
        plt.plot(stats['avg_rewards'], linewidth=2, color='steelblue')
        plt.axhline(y=optimal_prob, color='red', linestyle='--', 
                   linewidth=1.5, label=f'Optimal ({optimal_prob:.2f})')
        plt.xlabel("Timestep", fontsize=12)
        plt.ylabel("Average Reward", fontsize=12)
        plt.title(f"Epsilon-Greedy: Average Reward (ε={eps})", fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        filename = f"epsilon_{str(eps).replace('.', '_')}_reward.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")
    
    # Plot 2: Optimal Action Percentage (separate plot for each epsilon)
    for eps in epsilon_values:
        stats = all_results[eps]
        plt.figure(figsize=(10, 6))
        plt.plot(stats['optimal_action_pct'], linewidth=2, color='forestgreen')
        plt.xlabel("Timestep", fontsize=12)
        plt.ylabel("Optimal Action (%)", fontsize=12)
        plt.title(f"Epsilon-Greedy: Optimal Action Selection (ε={eps})", fontsize=14)
        plt.ylim([0, 105])
        plt.grid(alpha=0.3)
        plt.tight_layout()
        filename = f"epsilon_{str(eps).replace('.', '_')}_optimal.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")
    
    # Plot 3: Cumulative Regret (separate plot for each epsilon)
    for eps in epsilon_values:
        stats = all_results[eps]
        plt.figure(figsize=(10, 6))
        plt.plot(stats['cumulative_regret'], linewidth=2, color='crimson')
        plt.xlabel("Timestep", fontsize=12)
        plt.ylabel("Cumulative Regret", fontsize=12)
        plt.title(f"Epsilon-Greedy: Cumulative Regret (ε={eps})", fontsize=14)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        filename = f"epsilon_{str(eps).replace('.', '_')}_regret.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")
    
    # Plot 4: Comparison of all epsilon values - Average Reward
    plt.figure(figsize=(12, 7))
    colors = plt.cm.viridis(np.linspace(0, 1, len(epsilon_values)))
    for idx, eps in enumerate(epsilon_values):
        stats = all_results[eps]
        plt.plot(stats['avg_rewards'], linewidth=2, 
                label=f'ε={eps}', color=colors[idx], alpha=0.8)
    plt.axhline(y=optimal_prob, color='red', linestyle='--', 
               linewidth=2, label=f'Optimal ({optimal_prob:.2f})')
    plt.xlabel("Timestep", fontsize=12)
    plt.ylabel("Average Reward", fontsize=12)
    plt.title(f"Epsilon-Greedy: Comparison of All ε Values ({n_experiments} experiments)", 
              fontsize=14)
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "epsilon_comparison_reward.png"), 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: epsilon_comparison_reward.png")
    
    # Plot 5: Comparison of all epsilon values - Optimal Action %
    plt.figure(figsize=(12, 7))
    for idx, eps in enumerate(epsilon_values):
        stats = all_results[eps]
        plt.plot(stats['optimal_action_pct'], linewidth=2, 
                label=f'ε={eps}', color=colors[idx], alpha=0.8)
    plt.xlabel("Timestep", fontsize=12)
    plt.ylabel("Optimal Action (%)", fontsize=12)
    plt.title(f"Epsilon-Greedy: Comparison of Optimal Action Selection ({n_experiments} experiments)", 
              fontsize=14)
    plt.legend(fontsize=10, loc='lower right')
    plt.ylim([0, 105])
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "epsilon_comparison_optimal.png"), 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: epsilon_comparison_optimal.png")
    
    # Plot 6: Comparison of all epsilon values - Cumulative Regret
    plt.figure(figsize=(12, 7))
    for idx, eps in enumerate(epsilon_values):
        stats = all_results[eps]
        plt.plot(stats['cumulative_regret'], linewidth=2, 
                label=f'ε={eps}', color=colors[idx], alpha=0.8)
    plt.xlabel("Timestep", fontsize=12)
    plt.ylabel("Cumulative Regret", fontsize=12)
    plt.title(f"Epsilon-Greedy: Comparison of Cumulative Regret ({n_experiments} experiments)", 
              fontsize=14)
    plt.legend(fontsize=10, loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "epsilon_comparison_regret.png"), 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: epsilon_comparison_regret.png")


def run_epsilon_greedy_experiments(bandit_probs, epsilon_values, n_experiments, n_steps, output_dir):
    """
    Run epsilon-greedy experiments with multiple epsilon values.
    
    Args:
        bandit_probs (list): Arm probabilities
        epsilon_values (list): List of epsilon values to test
        n_experiments (int): Number of experiments per epsilon
        n_steps (int): Steps per experiment
        output_dir (str): Directory to save results
    
    Returns:
        dict: Results for all epsilon values
    """
    print("\n" + "=" * 70)
    print("EPSILON-GREEDY EXPERIMENTS")
    print("=" * 70)
    print(f"Testing epsilon values: {epsilon_values}")
    print(f"Running {n_experiments} experiments with {n_steps} steps each")
    print("=" * 70)
    
    optimal_reward = np.max(bandit_probs)
    all_results = {}
    
    for eps in epsilon_values:
        print(f"\nRunning experiments with ε = {eps}...")
        print("-" * 70)
        
        # Run experiments
        results = run_multiple_experiments(
            bandit_probs, 
            'epsilon-greedy', 
            n_experiments, 
            n_steps, 
            epsilon=eps
        )
        
        # Calculate statistics
        stats = calculate_statistics(results, optimal_reward, n_steps)
        all_results[eps] = stats
        
        # Print results
        print_experiment_results('ε', eps, stats, verbose=True)
    
    # Print summary table
    print_summary_table(all_results, 'ε', n_steps)
    
    # Find best epsilon
    best_eps_reward = find_best_parameter(all_results, 'total_reward')
    best_eps_convergence = find_best_parameter(all_results, 'convergence_step')
    best_eps_regret = find_best_parameter(all_results, 'final_regret')
    
    print("\n" + "=" * 70)
    print("BEST PARAMETERS")
    print("=" * 70)
    print(f"  Best ε (Total Reward):    {best_eps_reward}")
    print(f"  Best ε (Convergence):     {best_eps_convergence}")
    print(f"  Best ε (Lowest Regret):   {best_eps_regret}")
    print("=" * 70)
    
    # Generate plots
    print("\n" + "=" * 70)
    print("Generating plots...")
    print("-" * 70)
    plot_epsilon_greedy_results(all_results, bandit_probs, n_experiments, output_dir)
    
    return all_results


if __name__ == "__main__":
    # Configuration
    bandit_probs = [0.10, 0.50, 0.60, 0.80, 0.10,
                    0.25, 0.60, 0.45, 0.75, 0.65]
    epsilon_values = [0.0, 0.01, 0.05, 0.1, 0.5]
    n_experiments = 10000
    n_steps = 500
    output_dir = os.path.join(os.getcwd(), "output", "epsilon_greedy")
    
    # Run experiments
    results = run_epsilon_greedy_experiments(
        bandit_probs, 
        epsilon_values, 
        n_experiments, 
        n_steps, 
        output_dir
    )
    
    print("\n" + "=" * 70)
    print("EPSILON-GREEDY EXPERIMENTS COMPLETE!")
    print("=" * 70)
    print(f"Results saved to: {output_dir}/")
    print("=" * 70)
