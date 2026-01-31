import os
import time
import numpy as np
import matplotlib.pyplot as plt
from utils import print_config
from experiment_epsilon_greedy import run_epsilon_greedy_experiments
from experiment_ucb import run_ucb_experiments


def plot_epsilon_vs_ucb_comparison(epsilon_results, ucb_results, bandit_probs, 
                                   epsilon_values, c_values, n_experiments, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    optimal_prob = np.max(bandit_probs)
    
    # Find best parameters
    best_eps = max(epsilon_results.keys(), key=lambda k: epsilon_results[k]['total_reward'])
    best_c = max(ucb_results.keys(), key=lambda k: ucb_results[k]['total_reward'])
    
    epsilon_stats = epsilon_results[best_eps]
    ucb_stats = ucb_results[best_c]
    
    # Plot 1: Best Epsilon vs Best UCB - Average Reward
    plt.figure(figsize=(12, 7))
    plt.plot(epsilon_stats['avg_rewards'], linewidth=2.5, 
            label=f'Epsilon-Greedy (ε={best_eps})', color='steelblue', alpha=0.9)
    plt.plot(ucb_stats['avg_rewards'], linewidth=2.5, 
            label=f'UCB (c={best_c})', color='darkorange', alpha=0.9)
    plt.axhline(y=optimal_prob, color='red', linestyle='--', 
               linewidth=2, label=f'Optimal ({optimal_prob:.2f})')
    plt.xlabel("Timestep", fontsize=13)
    plt.ylabel("Average Reward", fontsize=13)
    plt.title(f"Epsilon-Greedy vs UCB: Average Reward (Best Parameters, {n_experiments} experiments)", 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "best_comparison_reward.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: best_comparison_reward.png")
    
    # Plot 2: Best Epsilon vs Best UCB - Optimal Action %
    plt.figure(figsize=(12, 7))
    plt.plot(epsilon_stats['optimal_action_pct'], linewidth=2.5, 
            label=f'Epsilon-Greedy (ε={best_eps})', color='steelblue', alpha=0.9)
    plt.plot(ucb_stats['optimal_action_pct'], linewidth=2.5, 
            label=f'UCB (c={best_c})', color='darkorange', alpha=0.9)
    plt.xlabel("Timestep", fontsize=13)
    plt.ylabel("Optimal Action (%)", fontsize=13)
    plt.title(f"Epsilon-Greedy vs UCB: Optimal Action Selection (Best Parameters, {n_experiments} experiments)", 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=12, loc='lower right')
    plt.ylim([0, 105])
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "best_comparison_optimal.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: best_comparison_optimal.png")
    
    # Plot 3: Best Epsilon vs Best UCB - Cumulative Regret
    plt.figure(figsize=(12, 7))
    plt.plot(epsilon_stats['cumulative_regret'], linewidth=2.5, 
            label=f'Epsilon-Greedy (ε={best_eps})', color='steelblue', alpha=0.9)
    plt.plot(ucb_stats['cumulative_regret'], linewidth=2.5, 
            label=f'UCB (c={best_c})', color='darkorange', alpha=0.9)
    plt.xlabel("Timestep", fontsize=13)
    plt.ylabel("Cumulative Regret", fontsize=13)
    plt.title(f"Epsilon-Greedy vs UCB: Cumulative Regret (Best Parameters, {n_experiments} experiments)", 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "best_comparison_regret.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: best_comparison_regret.png")
    
    # Plot 4: All Epsilon vs All UCB - Average Reward
    plt.figure(figsize=(14, 8))
    
    # Plot all epsilon values
    colors_eps = plt.cm.Blues(np.linspace(0.4, 0.9, len(epsilon_values)))
    for idx, eps in enumerate(sorted(epsilon_values)):
        plt.plot(epsilon_results[eps]['avg_rewards'], linewidth=2, 
                label=f'ε={eps}', color=colors_eps[idx], alpha=0.8, linestyle='-')
    
    # Plot all UCB values
    colors_ucb = plt.cm.Oranges(np.linspace(0.4, 0.9, len(c_values)))
    for idx, c in enumerate(sorted(c_values)):
        plt.plot(ucb_results[c]['avg_rewards'], linewidth=2, 
                label=f'UCB c={c}', color=colors_ucb[idx], alpha=0.8, linestyle='--')
    
    plt.axhline(y=optimal_prob, color='red', linestyle=':', 
               linewidth=2.5, label=f'Optimal ({optimal_prob:.2f})')
    plt.xlabel("Timestep", fontsize=13)
    plt.ylabel("Average Reward", fontsize=13)
    plt.title(f"All Epsilon-Greedy vs All UCB: Average Reward ({n_experiments} experiments)", 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=9, loc='lower right', ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_comparison_reward.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: all_comparison_reward.png")
    
    # Plot 5: All Epsilon vs All UCB - Optimal Action %
    plt.figure(figsize=(14, 8))
    
    # Plot all epsilon values
    for idx, eps in enumerate(sorted(epsilon_values)):
        plt.plot(epsilon_results[eps]['optimal_action_pct'], linewidth=2, 
                label=f'ε={eps}', color=colors_eps[idx], alpha=0.8, linestyle='-')
    
    # Plot all UCB values
    for idx, c in enumerate(sorted(c_values)):
        plt.plot(ucb_results[c]['optimal_action_pct'], linewidth=2, 
                label=f'UCB c={c}', color=colors_ucb[idx], alpha=0.8, linestyle='--')
    
    plt.xlabel("Timestep", fontsize=13)
    plt.ylabel("Optimal Action (%)", fontsize=13)
    plt.title(f"All Epsilon-Greedy vs All UCB: Optimal Action Selection ({n_experiments} experiments)", 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=9, loc='lower right', ncol=2)
    plt.ylim([0, 105])
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_comparison_optimal.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: all_comparison_optimal.png")
    
    # Plot 6: All Epsilon vs All UCB - Cumulative Regret
    plt.figure(figsize=(14, 8))
    
    # Plot all epsilon values
    for idx, eps in enumerate(sorted(epsilon_values)):
        plt.plot(epsilon_results[eps]['cumulative_regret'], linewidth=2, 
                label=f'ε={eps}', color=colors_eps[idx], alpha=0.8, linestyle='-')
    
    # Plot all UCB values
    for idx, c in enumerate(sorted(c_values)):
        plt.plot(ucb_results[c]['cumulative_regret'], linewidth=2, 
                label=f'UCB c={c}', color=colors_ucb[idx], alpha=0.8, linestyle='--')
    
    plt.xlabel("Timestep", fontsize=13)
    plt.ylabel("Cumulative Regret", fontsize=13)
    plt.title(f"All Epsilon-Greedy vs All UCB: Cumulative Regret ({n_experiments} experiments)", 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=9, loc='upper left', ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_comparison_regret.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: all_comparison_regret.png")


def main():
    bandit_probs = [0.10, 0.50, 0.60, 0.80, 0.10,
                    0.25, 0.60, 0.45, 0.75, 0.65]
    
    epsilon_values = [0.0, 0.01, 0.05, 0.1, 0.5]
    c_values = [0.5, 1.0, 2.0, 3.0, 5.0]
    
    n_experiments = 10000  # Number of independent runs
    n_steps = 500          # Timesteps per experiment
    
    output_base_dir = os.path.join(os.getcwd(), "output")
    epsilon_output_dir = os.path.join(output_base_dir, "epsilon_greedy")
    ucb_output_dir = os.path.join(output_base_dir, "ucb")

    print("\n" + "=" * 70)
    print(" " * 15 + "FRA 503 HOMEWORK 1")
    print(" " * 10 + "Multi-Armed Bandit Experiments")
    print("=" * 70)
    
    # Print configuration
    print_config(bandit_probs, n_experiments, n_steps)
    
    print("\nExperiment Parameters:")
    print(f"  - Epsilon values: {epsilon_values}")
    print(f"  - UCB c values:   {c_values}")
    print("=" * 70)
    
    start_time = time.time()
    
    # Part 1: Epsilon-Greedy Experiments
    print("\n\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 20 + "PART 1: EPSILON-GREEDY" + " " * 26 + "║")
    print("╚" + "=" * 68 + "╝")
    
    epsilon_start = time.time()
    epsilon_results = run_epsilon_greedy_experiments(
        bandit_probs,
        epsilon_values,
        n_experiments,
        n_steps,
        epsilon_output_dir
    )
    epsilon_time = time.time() - epsilon_start
    
    # Part 2: UCB Experiments
    print("\n\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 28 + "PART 2: UCB" + " " * 29 + "║")
    print("╚" + "=" * 68 + "╝")
    
    ucb_start = time.time()
    ucb_results = run_ucb_experiments(
        bandit_probs,
        c_values,
        n_experiments,
        n_steps,
        ucb_output_dir
    )
    ucb_time = time.time() - ucb_start
    
    # Part 3: Epsilon-Greedy vs UCB Comparison
    print("\n\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 18 + "PART 3: ALGORITHM COMPARISON" + " " * 22 + "║")
    print("╚" + "=" * 68 + "╝")
    
    comparison_output_dir = os.path.join(output_base_dir, "comparison")
    
    print("\n" + "=" * 70)
    print("Generating Epsilon-Greedy vs UCB Comparison Plots...")
    print("-" * 70)
    plot_epsilon_vs_ucb_comparison(
        epsilon_results,
        ucb_results,
        bandit_probs,
        epsilon_values,
        c_values,
        n_experiments,
        comparison_output_dir
    )
    
    total_time = time.time() - start_time

    print("\n\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 22 + "EXPERIMENT COMPLETE!" + " " * 25 + "║")
    print("╚" + "=" * 68 + "╝")
    
    print("\n" + "=" * 70)
    print("EXECUTION TIME")
    print("=" * 70)
    print(f"  Epsilon-Greedy: {epsilon_time:.2f} seconds")
    print(f"  UCB:            {ucb_time:.2f} seconds")
    print(f"  Total:          {total_time:.2f} seconds")
    print("=" * 70)
    
    print("\n" + "=" * 70)
    print("OUTPUT DIRECTORIES")
    print("=" * 70)
    print(f"  Epsilon-Greedy: {epsilon_output_dir}/")
    print(f"  UCB:            {ucb_output_dir}/")
    print(f"  Comparison:     {comparison_output_dir}/")
    print("=" * 70)
    
    print("\n" + "=" * 70)
    print("GENERATED FILES")
    print("=" * 70)
    print("\nEpsilon-Greedy (for each ε):")
    print("  - epsilon_X_X_reward.png    : Average reward over time")
    print("  - epsilon_X_X_optimal.png   : Optimal action selection %")
    print("  - epsilon_X_X_regret.png    : Cumulative regret")
    print("\nEpsilon-Greedy (comparisons):")
    print("  - epsilon_comparison_reward.png  : All ε values compared (reward)")
    print("  - epsilon_comparison_optimal.png : All ε values compared (optimal %)")
    print("  - epsilon_comparison_regret.png  : All ε values compared (regret)")
    
    print("\nUCB (for each c):")
    print("  - ucb_X_X_reward.png        : Average reward over time")
    print("  - ucb_X_X_optimal.png       : Optimal action selection %")
    print("  - ucb_X_X_regret.png        : Cumulative regret")
    print("\nUCB (comparisons):")
    print("  - ucb_comparison_reward.png  : All c values compared (reward)")
    print("  - ucb_comparison_optimal.png : All c values compared (optimal %)")
    print("  - ucb_comparison_regret.png  : All c values compared (regret)")
    
    print("\nEpsilon-Greedy vs UCB (best parameters):")
    print("  - best_comparison_reward.png  : Best ε vs Best c (reward)")
    print("  - best_comparison_optimal.png : Best ε vs Best c (optimal %)")
    print("  - best_comparison_regret.png  : Best ε vs Best c (regret)")
    
    print("\nEpsilon-Greedy vs UCB (all parameters):")
    print("  - all_comparison_reward.png   : All ε vs All c (reward)")
    print("  - all_comparison_optimal.png  : All ε vs All c (optimal %)")
    print("  - all_comparison_regret.png   : All ε vs All c (regret)")
    print("=" * 70)
    
    # Print best parameters
    print("\n" + "=" * 70)
    print("BEST PARAMETERS (OVERALL)")
    print("=" * 70)
    
    # Find best epsilon
    best_eps_reward = max(epsilon_results.keys(), 
                         key=lambda k: epsilon_results[k]['total_reward'])
    best_eps_convergence = min(epsilon_results.keys(), 
                              key=lambda k: epsilon_results[k]['convergence_step'])
    best_eps_regret = min(epsilon_results.keys(), 
                         key=lambda k: epsilon_results[k]['final_regret'])
    
    print("\nEpsilon-Greedy:")
    print(f"  Best ε (Reward):      {best_eps_reward} "
          f"({epsilon_results[best_eps_reward]['total_reward']:.2f})")
    print(f"  Best ε (Convergence): {best_eps_convergence} "
          f"(step {epsilon_results[best_eps_convergence]['convergence_step']})")
    print(f"  Best ε (Regret):      {best_eps_regret} "
          f"({epsilon_results[best_eps_regret]['final_regret']:.2f})")
    
    # Find best c
    best_c_reward = max(ucb_results.keys(), 
                       key=lambda k: ucb_results[k]['total_reward'])
    best_c_convergence = min(ucb_results.keys(), 
                            key=lambda k: ucb_results[k]['convergence_step'])
    best_c_regret = min(ucb_results.keys(), 
                       key=lambda k: ucb_results[k]['final_regret'])
    
    print("\nUCB:")
    print(f"  Best c (Reward):      {best_c_reward} "
          f"({ucb_results[best_c_reward]['total_reward']:.2f})")
    print(f"  Best c (Convergence): {best_c_convergence} "
          f"(step {ucb_results[best_c_convergence]['convergence_step']})")
    print(f"  Best c (Regret):      {best_c_regret} "
          f"({ucb_results[best_c_regret]['final_regret']:.2f})")
    print("=" * 70)
    
    # Final message
    print("\n" + "╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "ALL EXPERIMENTS COMPLETED SUCCESSFULLY!" + " " * 14 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

if __name__ == "__main__":
    main()
