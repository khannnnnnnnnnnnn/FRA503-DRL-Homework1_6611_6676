"""
agents.py

Agent classes for multi-armed bandit problem.
Includes Epsilon-Greedy and UCB algorithms.

FRA 503 Homework 1
"""
import numpy as np


class EpsilonGreedyAgent:
    """
    Agent using Epsilon-Greedy algorithm.
    
    With probability epsilon, the agent explores (random action).
    With probability (1-epsilon), the agent exploits (best known action).
    """
    
    def __init__(self, n_actions, epsilon):
        """
        Initialize the epsilon-greedy agent.
        
        Args:
            n_actions (int): Number of available actions (arms)
            epsilon (float): Exploration probability [0, 1]
                           0.0 = pure exploitation (greedy)
                           1.0 = pure exploration (random)
        
        Example:
            agent = EpsilonGreedyAgent(n_actions=10, epsilon=0.1)
        """
        self.n_actions = n_actions
        self.epsilon = epsilon
        
        # Learnable parameters
        self.Q = np.zeros(n_actions, dtype=np.float64)  # Action-value estimates
        self.N = np.zeros(n_actions, dtype=np.int32)    # Action counts
    
    def select_action(self):
        """
        Select an action using epsilon-greedy policy.
        
        Returns:
            int: Selected action index
        
        Policy:
            - With probability epsilon: select random action (explore)
            - With probability (1-epsilon): select best action (exploit)
        """
        if np.random.random() < self.epsilon:
            # Explore: choose random action
            return np.random.randint(self.n_actions)
        else:
            # Exploit: choose best action (break ties randomly)
            max_value = self.Q.max()
            best_actions = np.flatnonzero(self.Q == max_value)
            return np.random.choice(best_actions)
    
    def update(self, action, reward):
        """
        Update action-value estimates using incremental average.
        
        Args:
            action (int): Action that was taken
            reward (float): Reward that was received
        
        Update rule:
            Q(a) ← Q(a) + 1/N(a) * (reward - Q(a))
        
        This is the incremental formula for computing the average reward.
        """
        self.N[action] += 1
        alpha = 1.0 / self.N[action]  # Step size (decreases over time)
        self.Q[action] += alpha * (reward - self.Q[action])
    
    def get_Q_values(self):
        """Get current Q-value estimates."""
        return self.Q.copy()
    
    def get_action_counts(self):
        """Get count of times each action was selected."""
        return self.N.copy()
    
    def __str__(self):
        """String representation of the agent."""
        return f"EpsilonGreedyAgent(n_actions={self.n_actions}, epsilon={self.epsilon})"


class UCBAgent:
    """
    Agent using Upper Confidence Bound (UCB) algorithm.
    
    UCB balances exploration and exploitation by adding an uncertainty bonus
    to each action's value estimate. Actions that have been tried less often
    receive a larger bonus.
    """
    
    def __init__(self, n_actions, c=2.0):
        """
        Initialize the UCB agent.
        
        Args:
            n_actions (int): Number of available actions (arms)
            c (float): Exploration parameter (controls exploration vs exploitation)
                      Higher c = more exploration
                      Typical values: 1.0 to 3.0
        
        Example:
            agent = UCBAgent(n_actions=10, c=2.0)
        """
        self.n_actions = n_actions
        self.c = c
        
        # Learnable parameters
        self.Q = np.zeros(n_actions, dtype=np.float64)  # Action-value estimates
        self.N = np.zeros(n_actions, dtype=np.int32)    # Action counts
        self.t = 0  # Total timesteps
    
    def select_action(self):
        """
        Select an action using UCB policy.
        
        Returns:
            int: Selected action index
        
        UCB Formula:
            action = argmax[Q(a) + c * sqrt(log(t) / N(a))]
            
            where:
            - Q(a): estimated value of action a
            - c: exploration parameter
            - t: total timesteps
            - N(a): number of times action a was selected
        
        Policy:
            1. First, try each action at least once
            2. Then, select action with highest UCB value
        """
        self.t += 1
        
        # Phase 1: Try each action at least once
        untried_actions = np.flatnonzero(self.N == 0)
        if len(untried_actions) > 0:
            return np.random.choice(untried_actions)
        
        # Phase 2: Use UCB formula
        # UCB = Q(a) + c * sqrt(log(t) / N(a))
        ucb_values = self.Q + self.c * np.sqrt(np.log(self.t) / self.N)
        
        # Select action with highest UCB (break ties randomly)
        max_ucb = ucb_values.max()
        best_actions = np.flatnonzero(ucb_values == max_ucb)
        return np.random.choice(best_actions)
    
    def update(self, action, reward):
        """
        Update action-value estimates using incremental average.
        
        Args:
            action (int): Action that was taken
            reward (float): Reward that was received
        
        Update rule:
            Q(a) ← Q(a) + 1/N(a) * (reward - Q(a))
        
        Note: The update rule is the same as epsilon-greedy.
        The difference is in how actions are selected.
        """
        self.N[action] += 1
        alpha = 1.0 / self.N[action]  # Step size (decreases over time)
        self.Q[action] += alpha * (reward - self.Q[action])
    
    def get_Q_values(self):
        """Get current Q-value estimates."""
        return self.Q.copy()
    
    def get_action_counts(self):
        """Get count of times each action was selected."""
        return self.N.copy()
    
    def get_ucb_values(self):
        """
        Get current UCB values for all actions.
        
        Returns:
            array: UCB values for each action
        """
        if self.t == 0:
            return np.zeros(self.n_actions)
        
        ucb_values = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            if self.N[a] == 0:
                ucb_values[a] = np.inf  # Untried actions have infinite UCB
            else:
                ucb_values[a] = self.Q[a] + self.c * np.sqrt(np.log(self.t) / self.N[a])
        
        return ucb_values
    
    def __str__(self):
        """String representation of the agent."""
        return f"UCBAgent(n_actions={self.n_actions}, c={self.c})"
