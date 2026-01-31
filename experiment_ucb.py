import os
import numpy as np
import matplotlib.pyplot as plt
from bandit import Bandit
from agents import UCBAgent
from utils import (
    run_multiple_experiments,
    calculate_statistics,
    print_experiment_results,
    print_summary_table,
    find_best_parameter
)


def plot_ucb_results(all_results, bandit_probs, n_experiments, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    c_values = sorted(all_results.keys())
    optimal_prob = np.max(bandit_probs)
    
    # Plot 1: Average Reward Over Time (separate plot for each c)
    for c in c_values:
        stats = all_results[c]
        plt.figure(figsize=(10, 6))
        plt.plot(stats['avg_rewards'], linewidth=2, color='steelblue')
        plt.axhline(y=optimal_prob, color='red', linestyle='--', 
                   linewidth=1.5, label=f'Optimal ({optimal_prob:.2f})')
        plt.xlabel("Timestep", fontsize=12)
        plt.ylabel("Average Reward", fontsize=12)
        plt.title(f"UCB: Average Reward (c={c})", fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        filename = f"ucb_{str(c).replace('.', '_')}_reward.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")
    
    # Plot 2: Optimal Action Percentage (separate plot for each c)
    for c in c_values:
        stats = all_results[c]
        plt.figure(figsize=(10, 6))
        plt.plot(stats['optimal_action_pct'], linewidth=2, color='forestgreen')
        plt.xlabel("Timestep", fontsize=12)
        plt.ylabel("Optimal Action (%)", fontsize=12)
        plt.title(f"UCB: Optimal Action Selection (c={c})", fontsize=14)
        plt.ylim([0, 105])
        plt.grid(alpha=0.3)
        plt.tight_layout()
        filename = f"ucb_{str(c).replace('.', '_')}_optimal.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")
    
    # Plot 3: Cumulative Regret (separate plot for each c)
    for c in c_values:
        stats = all_results[c]
        plt.figure(figsize=(10, 6))
        plt.plot(stats['cumulative_regret'], linewidth=2, color='crimson')
        plt.xlabel("Timestep", fontsize=12)
        plt.ylabel("Cumulative Regret", fontsize=12)
        plt.title(f"UCB: Cumulative Regret (c={c})", fontsize=14)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        filename = f"ucb_{str(c).replace('.', '_')}_regret.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")
    
    # Plot 4: Comparison of all c values - Average Reward
    plt.figure(figsize=(12, 7))
    colors = plt.cm.plasma(np.linspace(0, 1, len(c_values)))
    for idx, c in enumerate(c_values):
        stats = all_results[c]
        plt.plot(stats['avg_rewards'], linewidth=2, 
                label=f'c={c}', color=colors[idx], alpha=0.8)
    plt.axhline(y=optimal_prob, color='red', linestyle='--', 
               linewidth=2, label=f'Optimal ({optimal_prob:.2f})')
    plt.xlabel("Timestep", fontsize=12)
    plt.ylabel("Average Reward", fontsize=12)
    plt.title(f"UCB: Comparison of All c Values ({n_experiments} experiments)", 
              fontsize=14)
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ucb_comparison_reward.png"), 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: ucb_comparison_reward.png")
    
    # Plot 5: Comparison of all c values - Optimal Action %
    plt.figure(figsize=(12, 7))
    for idx, c in enumerate(c_values):
        stats = all_results[c]
        plt.plot(stats['optimal_action_pct'], linewidth=2, 
                label=f'c={c}', color=colors[idx], alpha=0.8)
    plt.xlabel("Timestep", fontsize=12)
    plt.ylabel("Optimal Action (%)", fontsize=12)
    plt.title(f"UCB: Comparison of Optimal Action Selection ({n_experiments} experiments)", 
              fontsize=14)
    plt.legend(fontsize=10, loc='lower right')
    plt.ylim([0, 105])
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ucb_comparison_optimal.png"), 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: ucb_comparison_optimal.png")
    
    # Plot 6: Comparison of all c values - Cumulative Regret
    plt.figure(figsize=(12, 7))
    for idx, c in enumerate(c_values):
        stats = all_results[c]
        plt.plot(stats['cumulative_regret'], linewidth=2, 
                label=f'c={c}', color=colors[idx], alpha=0.8)
    plt.xlabel("Timestep", fontsize=12)
    plt.ylabel("Cumulative Regret", fontsize=12)
    plt.title(f"UCB: Comparison of Cumulative Regret ({n_experiments} experiments)", 
              fontsize=14)
    plt.legend(fontsize=10, loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ucb_comparison_regret.png"), 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: ucb_comparison_regret.png")


def run_ucb_experiments(bandit_probs, c_values, n_experiments, n_steps, output_dir):
    print("\n" + "=" * 70)
    print("UCB EXPERIMENTS")
    print("=" * 70)
    print(f"Testing c values: {c_values}")
    print(f"Running {n_experiments} experiments with {n_steps} steps each")
    print("=" * 70)
    
    optimal_reward = np.max(bandit_probs)
    all_results = {}
    
    for c in c_values:
        print(f"\nRunning experiments with c = {c}...")
        print("-" * 70)
        
        # Run experiments
        results = run_multiple_experiments(
            bandit_probs, 
            'ucb', 
            n_experiments, 
            n_steps, 
            c=c
        )
        
        # Calculate statistics
        stats = calculate_statistics(results, optimal_reward, n_steps)
        all_results[c] = stats
        
        # Print results
        print_experiment_results('c', c, stats, verbose=True)
    
    # Print summary table
    print_summary_table(all_results, 'c', n_steps)
    
    # Find best c
    best_c_reward = find_best_parameter(all_results, 'total_reward')
    best_c_convergence = find_best_parameter(all_results, 'convergence_step')
    best_c_regret = find_best_parameter(all_results, 'final_regret')
    
    print("\n" + "=" * 70)
    print("BEST PARAMETERS")
    print("=" * 70)
    print(f"  Best c (Total Reward):    {best_c_reward}")
    print(f"  Best c (Convergence):     {best_c_convergence}")
    print(f"  Best c (Lowest Regret):   {best_c_regret}")
    print("=" * 70)
    
    # Generate plots
    print("\n" + "=" * 70)
    print("Generating plots...")
    print("-" * 70)
    plot_ucb_results(all_results, bandit_probs, n_experiments, output_dir)
    
    return all_results


if __name__ == "__main__":
    # Configuration
    bandit_probs = [0.10, 0.50, 0.60, 0.80, 0.10,
                    0.25, 0.60, 0.45, 0.75, 0.65]
    c_values = [0.5, 1.0, 2.0, 3.0, 5.0]
    n_experiments = 10000
    n_steps = 500
    output_dir = os.path.join(os.getcwd(), "output", "ucb")
    
    # Run experiments
    results = run_ucb_experiments(
        bandit_probs, 
        c_values, 
        n_experiments, 
        n_steps, 
        output_dir
    )
    
    print("\n" + "=" * 70)
    print("UCB EXPERIMENTS COMPLETE!")
    print("=" * 70)
    print(f"Results saved to: {output_dir}/")
    print("=" * 70)
