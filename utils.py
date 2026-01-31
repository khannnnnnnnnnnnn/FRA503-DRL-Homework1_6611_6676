import numpy as np
from bandit import Bandit
from agents import EpsilonGreedyAgent, UCBAgent

def run_single_experiment(bandit, agent, n_steps):
    actions = np.zeros(n_steps, dtype=np.int32)
    rewards = np.zeros(n_steps, dtype=np.float64)
    
    for step in range(n_steps):
        # Agent selects action
        action = agent.select_action()
        
        # Environment returns reward
        reward = bandit.pull(action)
        
        # Agent updates its knowledge
        agent.update(action, reward)
        
        # Record what happened
        actions[step] = action
        rewards[step] = reward
    
    return actions, rewards


def run_multiple_experiments(bandit_probs, agent_type, n_experiments, n_steps, **agent_params):
    n_arms = len(bandit_probs)
    optimal_arm = np.argmax(bandit_probs)
    
    # Accumulators
    total_rewards = np.zeros(n_steps)
    optimal_actions = np.zeros(n_steps)
    all_rewards = np.zeros((n_experiments, n_steps))
    
    for exp in range(n_experiments):
        # Create fresh bandit and agent for each experiment
        bandit = Bandit(bandit_probs)
        
        if agent_type == 'epsilon-greedy':
            agent = EpsilonGreedyAgent(n_arms, agent_params['epsilon'])
        elif agent_type == 'ucb':
            agent = UCBAgent(n_arms, agent_params.get('c', 2.0))
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        # Run experiment
        actions, rewards = run_single_experiment(bandit, agent, n_steps)
        
        # Accumulate results
        total_rewards += rewards
        optimal_actions += (actions == optimal_arm)
        all_rewards[exp, :] = rewards
    
    # Calculate averages
    avg_rewards = total_rewards / n_experiments
    optimal_action_pct = (optimal_actions / n_experiments) * 100
    
    return {
        'avg_rewards': avg_rewards,
        'optimal_action_pct': optimal_action_pct,
        'all_rewards': all_rewards
    }

def calculate_cumulative_regret(rewards, optimal_reward):
    regret_per_step = optimal_reward - rewards
    cumulative_regret = np.cumsum(regret_per_step)
    return cumulative_regret


def calculate_convergence_step(rewards, optimal_reward, threshold=0.95, window=50):
    target_reward = threshold * optimal_reward
    
    # Check if we have enough data
    if len(rewards) < window:
        return len(rewards)
    
    # Calculate moving average
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    
    # Find first timestep where moving average >= target
    converged_indices = np.where(moving_avg >= target_reward)[0]
    
    if len(converged_indices) > 0:
        # Add window size because moving_avg is shorter
        return converged_indices[0] + window
    else:
        return len(rewards)


def calculate_statistics(results, optimal_reward, n_steps):
    avg_rewards = results['avg_rewards']
    optimal_pct = results['optimal_action_pct']
    
    cumulative_regret = calculate_cumulative_regret(avg_rewards, optimal_reward)
    convergence_step = calculate_convergence_step(avg_rewards, optimal_reward)
    
    # Calculate final optimal action percentage (last 100 steps)
    final_optimal_pct = np.mean(optimal_pct[-100:]) if len(optimal_pct) >= 100 else optimal_pct[-1]
    
    return {
        'total_reward': np.sum(avg_rewards),
        'avg_reward_per_step': np.mean(avg_rewards),
        'final_regret': cumulative_regret[-1],
        'convergence_step': convergence_step,
        'final_optimal_pct': final_optimal_pct,
        'cumulative_regret': cumulative_regret,
        'avg_rewards': avg_rewards,
        'optimal_action_pct': optimal_pct
    }


def print_experiment_results(param_name, param_value, stats, verbose=True):
    if verbose:
        print(f"\n{param_name} = {param_value}")
        print("-" * 70)
        print(f"  Total Reward:        {stats['total_reward']:.2f}")
        print(f"  Avg Reward/Step:     {stats['avg_reward_per_step']:.4f}")
        print(f"  Final Regret:        {stats['final_regret']:.2f}")
        print(f"  Convergence Step:    {stats['convergence_step']}")
        print(f"  Final Optimal %:     {stats['final_optimal_pct']:.2f}%")
    else:
        print(f"  {param_name}={param_value}: Reward={stats['total_reward']:.2f}, "
              f"Convergence={stats['convergence_step']}")


def print_summary_table(all_results, param_name, n_steps):
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    print(f"{'Parameter':<15} {'Total Reward':<15} {'Avg/Step':<12} {'Final Regret':<15} "
          f"{'Convergence':<15} {'Optimal %':<12}")
    print("-" * 100)
    
    for param_value in sorted(all_results.keys()):
        stats = all_results[param_value]
        print(f"{param_name}={param_value:<12.2f} "
              f"{stats['total_reward']:<15.2f} "
              f"{stats['avg_reward_per_step']:<12.4f} "
              f"{stats['final_regret']:<15.2f} "
              f"{stats['convergence_step']:<15} "
              f"{stats['final_optimal_pct']:<12.2f}%")
    
    print("=" * 100)


def find_best_parameter(all_results, metric='total_reward'):
    if metric == 'final_regret':
        # Lower is better for regret
        return min(all_results.keys(), key=lambda k: all_results[k][metric])
    elif metric == 'convergence_step':
        # Lower is better for convergence
        return min(all_results.keys(), key=lambda k: all_results[k][metric])
    else:
        # Higher is better for reward metrics
        return max(all_results.keys(), key=lambda k: all_results[k][metric])

def print_config(bandit_probs, n_experiments, n_steps):
    print("=" * 70)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 70)
    print(f"  Bandit arms:          {len(bandit_probs)}")
    print(f"  Arm probabilities:    {bandit_probs}")
    print(f"  Optimal arm:          Arm {np.argmax(bandit_probs) + 1} (p={np.max(bandit_probs):.2f})")
    print(f"  Experiments:          {n_experiments}")
    print(f"  Steps per experiment: {n_steps}")
    print("=" * 70)
