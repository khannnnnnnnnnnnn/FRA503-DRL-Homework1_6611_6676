"""
Multi-Armed Bandit Framework
Bandit Class Implementation
"""

import numpy as np


class Bandit:
    """
    Multi-armed bandit with hidden reward distributions.
    """
    
    def __init__(self, n_bandits, reward_type='gaussian', seed=None):
        """
        Initialize n bandits with hidden reward distributions.
        
        Parameters:
        -----------
        n_bandits : int
            Number of bandit arms
        reward_type : str
            Type of reward distribution ('gaussian' or 'bernoulli')
        seed : int
            Random seed for reproducibility
        """
        self.n_bandits = n_bandits
        self.reward_type = reward_type
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize hidden reward distributions
        if reward_type == 'gaussian':
            # Each bandit has a true mean from a normal distribution
            self.true_means = np.random.randn(n_bandits)
            self.true_stds = np.ones(n_bandits)  # Unit variance
            
        elif reward_type == 'bernoulli':
            # Each bandit has a probability of success
            self.true_probs = np.random.uniform(0, 1, n_bandits)
        
        # Track the optimal bandit
        if reward_type == 'gaussian':
            self.optimal_bandit = np.argmax(self.true_means)
            self.optimal_value = np.max(self.true_means)
        else:
            self.optimal_bandit = np.argmax(self.true_probs)
            self.optimal_value = np.max(self.true_probs)
    
    def pull(self, action):
        """
        Pull an arm and return a reward signal.
        
        Parameters:
        -----------
        action : int
            Index of the bandit arm to pull
            
        Returns:
        --------
        reward : float
            Reward received from pulling the arm
        """
        if action < 0 or action >= self.n_bandits:
            raise ValueError(f"Action must be between 0 and {self.n_bandits - 1}")
        
        if self.reward_type == 'gaussian':
            # Sample from Gaussian distribution
            reward = np.random.normal(self.true_means[action], 
                                     self.true_stds[action])
        elif self.reward_type == 'bernoulli':
            # Sample from Bernoulli distribution
            reward = np.random.binomial(1, self.true_probs[action])
        
        return reward
    
    def get_optimal_action(self):
        """Return the index of the optimal bandit."""
        return self.optimal_bandit
    
    def get_optimal_value(self):
        """Return the expected value of the optimal bandit."""
        return self.optimal_value
    
    def get_true_values(self):
        """Return the true expected values of all bandits."""
        if self.reward_type == 'gaussian':
            return self.true_means.copy()
        else:
            return self.true_probs.copy()
