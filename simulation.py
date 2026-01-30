"""
Multi-Armed Bandit Framework
Simulation Script for Running Experiments
"""

import numpy as np
import matplotlib.pyplot as plt
from bandit import Bandit
from agent import Agent
import seaborn as sns
from tqdm import tqdm
import os
import json
import pandas as pd
from pathlib import Path

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Create output directories
FIGURES_DIR = Path("figures")
LOGS_DIR = Path("logs")
FIGURES_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


def run_single_experiment(n_bandits, n_steps, algorithm, epsilon=None, ucb_c=None, 
                          seed=None, verbose=False):
    """
    Run a single experiment with specified parameters.
    
    Parameters:
    -----------
    n_bandits : int
        Number of bandit arms
    n_steps : int
        Number of time steps
    algorithm : str
        'epsilon-greedy' or 'ucb'
    epsilon : float
        Epsilon parameter for epsilon-greedy
    ucb_c : float
        Confidence parameter for UCB
    seed : int
        Random seed for reproducibility
    verbose : bool
        Print progress information
        
    Returns:
    --------
    results : dict
        Dictionary containing experiment results
    """
    # Initialize bandit and agent
    bandit = Bandit(n_bandits, reward_type='gaussian', seed=seed)
    
    if algorithm == 'epsilon-greedy':
        agent = Agent(n_bandits, algorithm='epsilon-greedy', epsilon=epsilon)
    elif algorithm == 'ucb':
        agent = Agent(n_bandits, algorithm='ucb', ucb_c=ucb_c)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Run simulation
    rewards = []
    cumulative_rewards = []
    optimal_actions = []
    regrets = []
    cumulative_regret = 0
    q_values_history = []  # Track Q-value evolution
    
    for step in range(n_steps):
        # Agent selects action
        action = agent.select_action()
        
        # Pull bandit arm and get reward
        reward = bandit.pull(action)
        
        # Update agent
        agent.update(action, reward)
        
        # Track metrics
        rewards.append(reward)
        cumulative_rewards.append(agent.get_cumulative_reward())
        
        # Check if optimal action was selected
        is_optimal = (action == bandit.get_optimal_action())
        optimal_actions.append(1 if is_optimal else 0)
        
        # Calculate regret (difference from optimal)
        regret = bandit.get_optimal_value() - reward
        cumulative_regret += regret
        regrets.append(cumulative_regret)
        
        # Store Q-values at key points
        if step in [99, 499, 999, n_steps-1] or step % 1000 == 0:
            q_values_history.append({
                'step': step,
                'q_values': agent.Q.copy(),
                'action_counts': agent.N.copy()
            })
    
    # Compile results
    results = {
        'rewards': np.array(rewards),
        'cumulative_rewards': np.array(cumulative_rewards),
        'optimal_actions': np.array(optimal_actions),
        'regrets': np.array(regrets),
        'final_Q_values': agent.Q.copy(),
        'action_counts': agent.N.copy(),
        'true_values': bandit.get_true_values(),
        'optimal_bandit': bandit.get_optimal_action(),
        'q_values_history': q_values_history,
        'agent': agent
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Algorithm: {algorithm}")
        if algorithm == 'epsilon-greedy':
            print(f"Epsilon: {epsilon}")
        else:
            print(f"UCB c: {ucb_c}")
        print(f"{'='*60}")
        print(f"True bandit values: {bandit.get_true_values()}")
        print(f"Optimal bandit: {bandit.get_optimal_action()} "
              f"(value: {bandit.get_optimal_value():.3f})")
        print(f"Final Q-values: {agent.Q}")
        print(f"Action percentages: {agent.get_action_percentages()}")
        print(f"Average reward: {np.mean(rewards):.3f}")
        print(f"Cumulative reward: {cumulative_rewards[-1]:.3f}")
        print(f"Cumulative regret: {regrets[-1]:.3f}")
        print(f"Optimal action rate: {np.mean(optimal_actions)*100:.2f}%")
    
    return results


def save_experiment_results(results_dict, algorithm, param_value, param_name, n_runs):
    """
    Save detailed results for a specific configuration including logs and figures.
    
    Parameters:
    -----------
    results_dict : dict
        Aggregated results from multiple runs
    algorithm : str
        Algorithm name
    param_value : float
        Parameter value used
    param_name : str
        Parameter name (epsilon or c)
    n_runs : int
        Number of runs performed
    """
    # Create experiment name
    if algorithm == 'epsilon-greedy':
        exp_name = f"epsilon_greedy_eps{param_value}"
    elif algorithm == 'ucb':
        exp_name = f"ucb_c{param_value}"
    else:
        exp_name = f"{algorithm}_param{param_value}"
    
    # Create directories
    fig_dir = FIGURES_DIR / exp_name
    log_dir = LOGS_DIR / exp_name
    fig_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    
    # Save log files
    save_logs(results_dict, log_dir, exp_name, algorithm, param_value, param_name, n_runs)
    
    # Generate and save individual plots
    save_individual_plots(results_dict, fig_dir, exp_name, algorithm, param_value)
    
    return exp_name


def save_logs(results_dict, log_dir, exp_name, algorithm, param_value, param_name, n_runs):
    """Save JSON log and CSV results."""
    
    # Create JSON log with metadata and summary statistics
    log_data = {
        'experiment_name': exp_name,
        'algorithm': algorithm,
        'parameter': {
            'name': param_name,
            'value': param_value
        },
        'configuration': {
            'n_runs': n_runs,
            'n_steps': len(results_dict['rewards_mean'])
        },
        'summary_statistics': {
            'final_cumulative_reward': {
                'mean': float(results_dict['cumulative_rewards_mean'][-1]),
                'std': float(results_dict['cumulative_rewards_std'][-1])
            },
            'final_regret': {
                'mean': float(results_dict['regrets_mean'][-1]),
                'std': float(results_dict['regrets_std'][-1])
            },
            'average_reward': {
                'mean': float(np.mean(results_dict['rewards_mean'])),
                'std': float(np.mean(results_dict['rewards_std']))
            },
            'final_optimal_action_rate': {
                'mean': float(np.mean(results_dict['optimal_actions_mean'][-100:])),
                'std': float(np.mean(results_dict['optimal_actions_std'][-100:]))
            }
        }
    }
    
    with open(log_dir / f"{exp_name}_log.json", 'w') as f:
        json.dump(log_data, f, indent=2)
    
    # Create CSV with detailed results
    df = pd.DataFrame({
        'step': np.arange(len(results_dict['rewards_mean'])),
        'reward_mean': results_dict['rewards_mean'],
        'reward_std': results_dict['rewards_std'],
        'cumulative_reward_mean': results_dict['cumulative_rewards_mean'],
        'cumulative_reward_std': results_dict['cumulative_rewards_std'],
        'optimal_action_mean': results_dict['optimal_actions_mean'],
        'optimal_action_std': results_dict['optimal_actions_std'],
        'regret_mean': results_dict['regrets_mean'],
        'regret_std': results_dict['regrets_std']
    })
    
    df.to_csv(log_dir / f"{exp_name}_results.csv", index=False)


def save_individual_plots(results_dict, fig_dir, exp_name, algorithm, param_value):
    """Generate and save individual plots for each experiment."""
    
    n_steps = len(results_dict['rewards_mean'])
    timesteps = np.arange(n_steps)
    
    # Plot 1: Q-value Comparison (final Q-values if available)
    plt.figure(figsize=(10, 6))
    if 'final_Q_values_mean' in results_dict:
        q_mean = results_dict['final_Q_values_mean']
        q_std = results_dict['final_Q_values_std']
        true_values = results_dict.get('true_values_mean', np.zeros_like(q_mean))
        
        x = np.arange(len(q_mean))
        plt.bar(x - 0.2, true_values, 0.4, label='True Values', alpha=0.7)
        plt.bar(x + 0.2, q_mean, 0.4, label='Learned Q-values', alpha=0.7)
        plt.errorbar(x + 0.2, q_mean, yerr=q_std, fmt='none', color='black', alpha=0.5)
        plt.xlabel('Action')
        plt.ylabel('Value')
        plt.title(f'Q-value Comparison: {exp_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / 'q_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Reward Distribution
    plt.figure(figsize=(10, 6))
    mean = results_dict['rewards_mean']
    std = results_dict['rewards_std']
    window = min(100, n_steps // 10)
    mean_smooth = np.convolve(mean, np.ones(window)/window, mode='valid')
    std_smooth = np.convolve(std, np.ones(window)/window, mode='valid')
    timesteps_smooth = timesteps[:len(mean_smooth)]
    
    plt.plot(timesteps_smooth, mean_smooth, linewidth=2, label='Mean Reward')
    plt.fill_between(timesteps_smooth, 
                     mean_smooth - std_smooth, 
                     mean_smooth + std_smooth, 
                     alpha=0.3, label='±1 Std Dev')
    plt.xlabel('Time Steps')
    plt.ylabel('Reward')
    plt.title(f'Reward Distribution over Time: {exp_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / 'reward_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Action Counts (if available)
    plt.figure(figsize=(10, 6))
    if 'action_counts_mean' in results_dict:
        counts_mean = results_dict['action_counts_mean']
        counts_std = results_dict['action_counts_std']
        x = np.arange(len(counts_mean))
        plt.bar(x, counts_mean, alpha=0.7)
        plt.errorbar(x, counts_mean, yerr=counts_std, fmt='none', color='black', alpha=0.5)
        plt.xlabel('Action')
        plt.ylabel('Selection Count')
        plt.title(f'Action Selection Counts: {exp_name}')
        plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(fig_dir / 'action_counts.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Q-value Error over time (difference from true values)
    plt.figure(figsize=(10, 6))
    # This would require tracking Q-values over time - placeholder for now
    plt.plot(timesteps, results_dict['regrets_mean'] / (timesteps + 1), linewidth=2)
    plt.fill_between(timesteps, 
                     (results_dict['regrets_mean'] - results_dict['regrets_std']) / (timesteps + 1),
                     (results_dict['regrets_mean'] + results_dict['regrets_std']) / (timesteps + 1),
                     alpha=0.3)
    plt.xlabel('Time Steps')
    plt.ylabel('Average Regret per Step')
    plt.title(f'Average Regret per Step: {exp_name}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / 'q_error.png', dpi=300, bbox_inches='tight')
    plt.close()


def run_multiple_experiments(n_bandits, n_steps, algorithm, param_values, 
                            param_name, n_runs=100):
    """
    Run multiple experiments with different parameter values.
    
    Parameters:
    -----------
    n_bandits : int
        Number of bandit arms
    n_steps : int
        Number of time steps per run
    algorithm : str
        'epsilon-greedy' or 'ucb'
    param_values : list
        List of parameter values to test
    param_name : str
        Name of parameter ('epsilon' or 'ucb_c')
    n_runs : int
        Number of independent runs per parameter value
        
    Returns:
    --------
    all_results : dict
        Dictionary mapping parameter values to their results
    exp_names : dict
        Dictionary mapping parameter values to experiment names
    """
    all_results = {}
    exp_names = {}
    
    for param_value in param_values:
        print(f"\nRunning experiments with {param_name}={param_value}")
        
        # Store results for all runs
        rewards_all = []
        cumulative_rewards_all = []
        optimal_actions_all = []
        regrets_all = []
        final_Q_values_all = []
        action_counts_all = []
        true_values_all = []
        
        for run in tqdm(range(n_runs), desc=f"{param_name}={param_value}"):
            # Use different seed for each run
            if algorithm == 'epsilon-greedy':
                results = run_single_experiment(
                    n_bandits, n_steps, algorithm, 
                    epsilon=param_value, seed=run
                )
            else:  # ucb
                results = run_single_experiment(
                    n_bandits, n_steps, algorithm, 
                    ucb_c=param_value, seed=run
                )
            
            rewards_all.append(results['rewards'])
            cumulative_rewards_all.append(results['cumulative_rewards'])
            optimal_actions_all.append(results['optimal_actions'])
            regrets_all.append(results['regrets'])
            final_Q_values_all.append(results['final_Q_values'])
            action_counts_all.append(results['action_counts'])
            true_values_all.append(results['true_values'])
        
        # Calculate statistics across runs
        results_dict = {
            'rewards_mean': np.mean(rewards_all, axis=0),
            'rewards_std': np.std(rewards_all, axis=0),
            'cumulative_rewards_mean': np.mean(cumulative_rewards_all, axis=0),
            'cumulative_rewards_std': np.std(cumulative_rewards_all, axis=0),
            'optimal_actions_mean': np.mean(optimal_actions_all, axis=0),
            'optimal_actions_std': np.std(optimal_actions_all, axis=0),
            'regrets_mean': np.mean(regrets_all, axis=0),
            'regrets_std': np.std(regrets_all, axis=0),
            'final_Q_values_mean': np.mean(final_Q_values_all, axis=0),
            'final_Q_values_std': np.std(final_Q_values_all, axis=0),
            'action_counts_mean': np.mean(action_counts_all, axis=0),
            'action_counts_std': np.std(action_counts_all, axis=0),
            'true_values_mean': np.mean(true_values_all, axis=0),
        }
        
        all_results[param_value] = results_dict
        
        # Save results for this configuration
        exp_name = save_experiment_results(
            results_dict, algorithm, param_value, param_name, n_runs
        )
        exp_names[param_value] = exp_name
    
    return all_results, exp_names


def save_combined_plots(all_results, param_name, algorithm, n_steps, prefix=""):
    """
    Generate and save combined comparison plots.
    
    Parameters:
    -----------
    all_results : dict
        Results from run_multiple_experiments
    param_name : str
        Name of parameter being varied
    algorithm : str
        Algorithm name for title
    n_steps : int
        Number of time steps
    prefix : str
        Prefix for filename
    """
    timesteps = np.arange(n_steps)
    
    # Combined Rewards Plot
    plt.figure(figsize=(12, 6))
    for param_value, results in all_results.items():
        mean = results['rewards_mean']
        std = results['rewards_std']
        window = min(100, n_steps // 10)
        mean_smooth = np.convolve(mean, np.ones(window)/window, mode='valid')
        std_smooth = np.convolve(std, np.ones(window)/window, mode='valid')
        timesteps_smooth = timesteps[:len(mean_smooth)]
        
        plt.plot(timesteps_smooth, mean_smooth, label=f'{param_name}={param_value}', linewidth=2)
        plt.fill_between(timesteps_smooth, 
                        mean_smooth - std_smooth, 
                        mean_smooth + std_smooth, 
                        alpha=0.2)
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    plt.title(f'{algorithm}: Average Reward over Time', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = f"{prefix}combined_rewards.png" if prefix else "combined_rewards.png"
    plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Combined Optimal Actions Plot
    plt.figure(figsize=(12, 6))
    for param_value, results in all_results.items():
        mean = results['optimal_actions_mean']
        cumsum = np.cumsum(mean)
        running_avg = cumsum / (timesteps + 1) * 100
        plt.plot(timesteps, running_avg, label=f'{param_name}={param_value}', linewidth=2)
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('% Optimal Action', fontsize=12)
    plt.title(f'{algorithm}: Optimal Action Selection Rate', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 100])
    plt.tight_layout()
    filename = f"{prefix}combined_optimal.png" if prefix else "combined_optimal.png"
    plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Combined Regret Plot
    plt.figure(figsize=(12, 6))
    for param_value, results in all_results.items():
        mean = results['regrets_mean']
        std = results['regrets_std']
        plt.plot(timesteps, mean, label=f'{param_name}={param_value}', linewidth=2)
        plt.fill_between(timesteps, mean - std, mean + std, alpha=0.2)
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Cumulative Regret', fontsize=12)
    plt.title(f'{algorithm}: Cumulative Regret', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = f"{prefix}combined_regret.png" if prefix else "combined_regret.png"
    plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Combined Q-error (average regret per step)
    plt.figure(figsize=(12, 6))
    for param_value, results in all_results.items():
        mean = results['regrets_mean']
        avg_regret = mean / (timesteps + 1)
        plt.plot(timesteps, avg_regret, label=f'{param_name}={param_value}', linewidth=2)
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Average Regret per Step', fontsize=12)
    plt.title(f'{algorithm}: Average Regret per Step', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = f"{prefix}combined_q_error.png" if prefix else "combined_q_error.png"
    plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()


def save_group_plot(all_results, param_name, algorithm, n_steps):
    """
    Save a comprehensive 2x2 group plot for an algorithm.
    
    Parameters:
    -----------
    all_results : dict
        Results from experiments
    param_name : str
        Parameter name
    algorithm : str
        Algorithm name
    n_steps : int
        Number of timesteps
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    timesteps = np.arange(n_steps)
    
    # Plot 1: Average Reward over Time
    ax = axes[0, 0]
    for param_value, results in all_results.items():
        mean = results['rewards_mean']
        std = results['rewards_std']
        window = min(100, n_steps // 10)
        mean_smooth = np.convolve(mean, np.ones(window)/window, mode='valid')
        std_smooth = np.convolve(std, np.ones(window)/window, mode='valid')
        timesteps_smooth = timesteps[:len(mean_smooth)]
        
        ax.plot(timesteps_smooth, mean_smooth, label=f'{param_name}={param_value}', linewidth=2)
        ax.fill_between(timesteps_smooth, 
                        mean_smooth - std_smooth, 
                        mean_smooth + std_smooth, 
                        alpha=0.2)
    ax.set_xlabel('Time Steps', fontsize=11)
    ax.set_ylabel('Average Reward', fontsize=11)
    ax.set_title('Average Reward over Time', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative Reward
    ax = axes[0, 1]
    for param_value, results in all_results.items():
        mean = results['cumulative_rewards_mean']
        std = results['cumulative_rewards_std']
        ax.plot(timesteps, mean, label=f'{param_name}={param_value}', linewidth=2)
        ax.fill_between(timesteps, mean - std, mean + std, alpha=0.2)
    ax.set_xlabel('Time Steps', fontsize=11)
    ax.set_ylabel('Cumulative Reward', fontsize=11)
    ax.set_title('Cumulative Reward', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Optimal Action Selection Rate
    ax = axes[1, 0]
    for param_value, results in all_results.items():
        mean = results['optimal_actions_mean']
        cumsum = np.cumsum(mean)
        running_avg = cumsum / (timesteps + 1) * 100
        ax.plot(timesteps, running_avg, label=f'{param_name}={param_value}', linewidth=2)
    ax.set_xlabel('Time Steps', fontsize=11)
    ax.set_ylabel('% Optimal Action', fontsize=11)
    ax.set_title('Optimal Action Selection Rate', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    
    # Plot 4: Cumulative Regret
    ax = axes[1, 1]
    for param_value, results in all_results.items():
        mean = results['regrets_mean']
        std = results['regrets_std']
        ax.plot(timesteps, mean, label=f'{param_name}={param_value}', linewidth=2)
        ax.fill_between(timesteps, mean - std, mean + std, alpha=0.2)
    ax.set_xlabel('Time Steps', fontsize=11)
    ax.set_ylabel('Cumulative Regret', fontsize=11)
    ax.set_title('Cumulative Regret', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Overall title
    algorithm_name = algorithm.replace('_', ' ').title()
    fig.suptitle(f'{algorithm_name} - Performance Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save with algorithm-specific name
    if 'epsilon' in algorithm or 'greedy' in algorithm.lower():
        filename = "group_epsilon_greedy.png"
    elif 'ucb' in algorithm.lower():
        filename = "group_ucb.png"
    else:
        filename = f"group_{algorithm}.png"
    
    plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename


def analyze_convergence(all_results, param_values, threshold=0.95):
    """
    Analyze when each configuration converges.
    
    Convergence is defined as maintaining optimal action selection
    above threshold for 100 consecutive steps.
    
    Parameters:
    -----------
    all_results : dict
        Results from experiments
    param_values : list
        Parameter values tested
    threshold : float
        Threshold for optimal action selection (default 0.95 = 95%)
        
    Returns:
    --------
    convergence_data : dict
        Convergence information for each parameter value
    """
    convergence_data = {}
    window_size = 100
    
    for param_value in param_values:
        optimal_actions = all_results[param_value]['optimal_actions_mean']
        
        # Calculate rolling average
        converged_step = None
        for i in range(len(optimal_actions) - window_size):
            window = optimal_actions[i:i+window_size]
            if np.mean(window) >= threshold:
                converged_step = i
                break
        
        convergence_data[param_value] = {
            'converged_at': converged_step,
            'final_optimal_rate': np.mean(optimal_actions[-100:])
        }
    
    return convergence_data


def print_convergence_analysis(convergence_data, param_name):
    """Print convergence analysis results."""
    print(f"\n{'='*60}")
    print("CONVERGENCE ANALYSIS")
    print(f"{'='*60}")
    print(f"Convergence defined as: 95% optimal action rate for 100 consecutive steps")
    print(f"\n{param_name:>10} | Converged At | Final Optimal Rate")
    print("-" * 60)
    
    for param_value, data in convergence_data.items():
        converged = data['converged_at']
        final_rate = data['final_optimal_rate'] * 100
        
        if converged is not None:
            print(f"{param_value:>10} | Step {converged:>7} | {final_rate:>6.2f}%")
        else:
            print(f"{param_value:>10} | Not converged | {final_rate:>6.2f}%")


if __name__ == "__main__":
    # Experimental parameters
    N_BANDITS = 10
    N_STEPS = 10000
    N_RUNS = 100
    
    print("="*60)
    print("MULTI-ARMED BANDIT EXPERIMENTS")
    print("="*60)
    print(f"Configuration:")
    print(f"  - Number of bandits: {N_BANDITS}")
    print(f"  - Time steps per run: {N_STEPS}")
    print(f"  - Independent runs: {N_RUNS}")
    print(f"\nOutput directories:")
    print(f"  - Figures: {FIGURES_DIR.absolute()}")
    print(f"  - Logs: {LOGS_DIR.absolute()}")
    print("="*60)
    
    # ========================================================================
    # EXPERIMENT 1: Epsilon-Greedy
    # ========================================================================
    print("\n" + "="*60)
    print("EXPERIMENT 1: EPSILON-GREEDY ALGORITHM")
    print("="*60)
    
    epsilon_values = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3]
    
    epsilon_results, epsilon_exp_names = run_multiple_experiments(
        n_bandits=N_BANDITS,
        n_steps=N_STEPS,
        algorithm='epsilon-greedy',
        param_values=epsilon_values,
        param_name='epsilon',
        n_runs=N_RUNS
    )
    
    # Generate combined plots
    print("\nGenerating epsilon-greedy combined plots...")
    save_combined_plots(epsilon_results, 'ε', 'Epsilon-Greedy', N_STEPS)
    
    # Generate group plot
    print("Generating epsilon-greedy group plot...")
    save_group_plot(epsilon_results, 'ε', 'Epsilon-Greedy', N_STEPS)
    
    # Analyze convergence
    epsilon_convergence = analyze_convergence(epsilon_results, epsilon_values)
    print_convergence_analysis(epsilon_convergence, 'Epsilon')
    
    # ========================================================================
    # EXPERIMENT 2: UCB
    # ========================================================================
    print("\n" + "="*60)
    print("EXPERIMENT 2: UCB ALGORITHM")
    print("="*60)
    
    ucb_c_values = [0.5, 1.0, 2.0, 3.0, 5.0]
    
    ucb_results, ucb_exp_names = run_multiple_experiments(
        n_bandits=N_BANDITS,
        n_steps=N_STEPS,
        algorithm='ucb',
        param_values=ucb_c_values,
        param_name='ucb_c',
        n_runs=N_RUNS
    )
    
    # Generate combined plots
    print("\nGenerating UCB combined plots...")
    save_combined_plots(ucb_results, 'c', 'UCB', N_STEPS)
    
    # Generate group plot
    print("Generating UCB group plot...")
    save_group_plot(ucb_results, 'c', 'UCB', N_STEPS)
    
    # Analyze convergence
    ucb_convergence = analyze_convergence(ucb_results, ucb_c_values)
    print_convergence_analysis(ucb_convergence, 'UCB c')
    
    # ========================================================================
    # EXPERIMENT 3: Best Overlay (Comparison)
    # ========================================================================
    print("\n" + "="*60)
    print("EXPERIMENT 3: BEST PARAMETERS COMPARISON")
    print("="*60)
    
    # Find best parameters based on final cumulative reward
    best_epsilon = max(epsilon_values, 
                       key=lambda e: epsilon_results[e]['cumulative_rewards_mean'][-1])
    best_ucb_c = max(ucb_c_values,
                     key=lambda c: ucb_results[c]['cumulative_rewards_mean'][-1])
    
    print(f"\nBest parameters:")
    print(f"  - Epsilon-Greedy: ε={best_epsilon}")
    print(f"  - UCB: c={best_ucb_c}")
    
    # Create best overlay plot
    print("\nGenerating best overlay comparison...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    timesteps = np.arange(N_STEPS)
    colors = {'Epsilon-Greedy': '#2E86AB', 'UCB': '#A23B72'}
    
    comparison_data = {
        'Epsilon-Greedy': epsilon_results[best_epsilon],
        'UCB': ucb_results[best_ucb_c]
    }
    
    # Average Reward
    ax = axes[0, 0]
    for name, results in comparison_data.items():
        mean = results['rewards_mean']
        std = results['rewards_std']
        window = 100
        mean_smooth = np.convolve(mean, np.ones(window)/window, mode='valid')
        std_smooth = np.convolve(std, np.ones(window)/window, mode='valid')
        timesteps_smooth = timesteps[:len(mean_smooth)]
        ax.plot(timesteps_smooth, mean_smooth, label=name, color=colors[name], linewidth=2.5)
        ax.fill_between(timesteps_smooth, mean_smooth - std_smooth, 
                        mean_smooth + std_smooth, alpha=0.2, color=colors[name])
    ax.set_xlabel('Time Steps', fontsize=11)
    ax.set_ylabel('Average Reward', fontsize=11)
    ax.set_title('Average Reward Comparison', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Cumulative Reward
    ax = axes[0, 1]
    for name, results in comparison_data.items():
        mean = results['cumulative_rewards_mean']
        std = results['cumulative_rewards_std']
        ax.plot(timesteps, mean, label=name, color=colors[name], linewidth=2.5)
        ax.fill_between(timesteps, mean - std, mean + std, 
                        alpha=0.2, color=colors[name])
    ax.set_xlabel('Time Steps', fontsize=11)
    ax.set_ylabel('Cumulative Reward', fontsize=11)
    ax.set_title('Cumulative Reward Comparison', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Optimal Action Rate
    ax = axes[1, 0]
    for name, results in comparison_data.items():
        mean = results['optimal_actions_mean']
        cumsum = np.cumsum(mean)
        running_avg = cumsum / (timesteps + 1) * 100
        ax.plot(timesteps, running_avg, label=name, color=colors[name], linewidth=2.5)
    ax.set_xlabel('Time Steps', fontsize=11)
    ax.set_ylabel('% Optimal Action', fontsize=11)
    ax.set_title('Optimal Action Selection Rate', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    
    # Cumulative Regret
    ax = axes[1, 1]
    for name, results in comparison_data.items():
        mean = results['regrets_mean']
        std = results['regrets_std']
        ax.plot(timesteps, mean, label=name, color=colors[name], linewidth=2.5)
        ax.fill_between(timesteps, mean - std, mean + std, 
                        alpha=0.2, color=colors[name])
    ax.set_xlabel('Time Steps', fontsize=11)
    ax.set_ylabel('Cumulative Regret', fontsize=11)
    ax.set_title('Cumulative Regret Comparison', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Best Parameters Comparison: ε-greedy(ε={best_epsilon}) vs UCB(c={best_ucb_c})', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(FIGURES_DIR / 'best_overlay.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print final statistics
    print("\n" + "="*60)
    print("FINAL PERFORMANCE COMPARISON")
    print("="*60)
    for name, results in comparison_data.items():
        final_cumulative = results['cumulative_rewards_mean'][-1]
        final_regret = results['regrets_mean'][-1]
        final_optimal_rate = np.mean(results['optimal_actions_mean'][-100:]) * 100
        
        print(f"\n{name}:")
        print(f"  Final Cumulative Reward: {final_cumulative:.2f} ± {results['cumulative_rewards_std'][-1]:.2f}")
        print(f"  Final Cumulative Regret: {final_regret:.2f} ± {results['regrets_std'][-1]:.2f}")
        print(f"  Final Optimal Action Rate: {final_optimal_rate:.2f}%")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*60)
    print("EXPERIMENTS COMPLETED!")
    print("="*60)
    print(f"\nGenerated files:")
    print(f"\nFigures ({FIGURES_DIR}):")
    print(f"  Individual experiment folders:")
    for exp_name in list(epsilon_exp_names.values()) + list(ucb_exp_names.values()):
        print(f"    - {exp_name}/ (4 plots)")
    print(f"  Combined plots:")
    print(f"    - combined_rewards.png")
    print(f"    - combined_optimal.png")
    print(f"    - combined_regret.png")
    print(f"    - combined_q_error.png")
    print(f"  Group plots:")
    print(f"    - group_epsilon_greedy.png")
    print(f"    - group_ucb.png")
    print(f"  Comparison:")
    print(f"    - best_overlay.png")
    
    print(f"\nLogs ({LOGS_DIR}):")
    for exp_name in list(epsilon_exp_names.values()) + list(ucb_exp_names.values()):
        print(f"  - {exp_name}/")
        print(f"      {exp_name}_log.json")
        print(f"      {exp_name}_results.csv")
    
    print("\n" + "="*60)
    plt.close('all')
