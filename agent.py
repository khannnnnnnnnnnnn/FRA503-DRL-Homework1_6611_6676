"""
Multi-Armed Bandit Framework
Agent Class Implementation
"""

import numpy as np


class Agent:
    def __init__(self, n_actions, algorithm='epsilon-greedy', epsilon=0.1, 
                 ucb_c=2.0, initial_value=0.0, step_size=None):
        """
        Initialize an agent with learnable parameters.
        
        Parameters:
        -----------
        n_actions : int
            Number of available actions (bandit arms)
        algorithm : str
            Learning algorithm ('epsilon-greedy' or 'ucb')
        epsilon : float
            Exploration rate for epsilon-greedy (0 to 1)
        ucb_c : float
            Confidence parameter for UCB algorithm
        initial_value : float
            Initial Q-value estimates for all actions
        step_size : float or None
            Fixed step size for updates. If None, uses sample average (1/n)
        """
        self.n_actions = n_actions
        self.algorithm = algorithm
        self.epsilon = epsilon
        self.ucb_c = ucb_c
        self.step_size = step_size
        
        # Initialize Q-value estimates
        self.Q = np.ones(n_actions) * initial_value
        
        # Initialize action counts
        self.N = np.zeros(n_actions)
        
        # Total steps taken
        self.total_steps = 0
        
        # History tracking
        self.reward_history = []
        self.action_history = []
    
    def select_action(self):
        """
        Select an action based on the algorithm.
        
        Returns:
        --------
        action : int
            Index of the selected action
        """
        self.total_steps += 1
        
        if self.algorithm == 'epsilon-greedy':
            return self._epsilon_greedy()
        elif self.algorithm == 'ucb':
            return self._ucb()
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def _epsilon_greedy(self):
        """
        Epsilon-greedy action selection.
        
        With probability epsilon, select a random action (exploration).
        With probability 1-epsilon, select the greedy action (exploitation).
        """
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.n_actions)
        else:
            # Exploit: greedy action
            return np.argmax(self.Q)
    
    def _ucb(self):
        """
        Upper Confidence Bound action selection.
        
        Select action that maximizes: Q(a) + c * sqrt(ln(t) / N(a))
        
        For actions that haven't been tried yet (N(a) = 0), 
        we give them maximum priority by returning inf for their UCB value.
        """
        # First, ensure all actions are tried at least once
        if np.any(self.N == 0):
            # Return the first untried action
            return np.where(self.N == 0)[0][0]
        
        # Calculate UCB values for all actions
        ucb_values = self.Q + self.ucb_c * np.sqrt(
            np.log(self.total_steps) / self.N
        )
        
        # Select action with highest UCB value
        return np.argmax(ucb_values)
    
    def update(self, action, reward):
        """
        Update agent's Q-values and action counts based on received reward.
        
        Parameters:
        -----------
        action : int
            Action that was taken
        reward : float
            Reward received from the action
        """
        # Update action count
        self.N[action] += 1
        
        # Update Q-value estimate
        if self.step_size is None:
            # Sample average (incremental implementation)
            alpha = 1.0 / self.N[action]
        else:
            # Fixed step size
            alpha = self.step_size
        
        # Q(a) = Q(a) + alpha * [R - Q(a)]
        self.Q[action] += alpha * (reward - self.Q[action])
        
        # Track history
        self.reward_history.append(reward)
        self.action_history.append(action)
    
    def get_action_percentages(self):
        """
        Get the percentage of times each action was selected.
        
        Returns:
        --------
        percentages : np.array
            Percentage for each action
        """
        if self.total_steps == 0:
            return np.zeros(self.n_actions)
        return self.N / self.total_steps * 100
    
    def reset(self):
        """Reset the agent to initial state."""
        self.Q = np.ones(self.n_actions) * 0.0
        self.N = np.zeros(self.n_actions)
        self.total_steps = 0
        self.reward_history = []
        self.action_history = []
    
    def get_cumulative_reward(self):
        """Get cumulative reward over all steps."""
        return np.sum(self.reward_history)
    
    def get_average_reward(self):
        """Get average reward per step."""
        if len(self.reward_history) == 0:
            return 0.0
        return np.mean(self.reward_history)
