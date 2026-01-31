"""
bandit.py

Multi-armed bandit environment class.
Each arm has a hidden Bernoulli reward distribution.

FRA 503 Homework 1
"""
import numpy as np


class Bandit:
    """
    Multi-armed bandit environment.
    
    Each arm has a hidden reward distribution (Bernoulli).
    When an arm is pulled, it returns reward 1 with its success probability,
    and reward 0 otherwise.
    """
    
    def __init__(self, probs):
        """
        Initialize the bandit environment.
        
        Args:
            probs (list or array): Success probabilities for each arm.
                                   Each value should be between 0 and 1.
        
        Example:
            bandit = Bandit([0.1, 0.5, 0.8])  # 3 arms with different probabilities
        """
        self.probs = np.array(probs, dtype=np.float64)
        self.n_arms = len(probs)
        self.optimal_arm = np.argmax(probs)
        self.optimal_prob = np.max(probs)
    
    def pull(self, action):
        """
        Pull an arm and return a stochastic reward.
        
        Args:
            action (int): Index of the arm to pull (0 to n_arms-1)
        
        Returns:
            int: Reward (1 for success, 0 for failure)
        
        Raises:
            ValueError: If action is out of valid range
        
        Example:
            reward = bandit.pull(2)  # Pull arm 2
        """
        if action < 0 or action >= self.n_arms:
            raise ValueError(f"Invalid action {action}. Must be in range [0, {self.n_arms-1}]")
        
        # Bernoulli reward: return 1 with probability probs[action]
        random_value = np.random.random()
        return 1 if random_value < self.probs[action] else 0
    
    def get_optimal_arm(self):
        """
        Get the index of the optimal arm (highest probability).
        
        Returns:
            int: Index of optimal arm
        """
        return self.optimal_arm
    
    def get_optimal_prob(self):
        """
        Get the success probability of the optimal arm.
        
        Returns:
            float: Probability of optimal arm
        """
        return self.optimal_prob
    
    def get_arm_prob(self, action):
        """
        Get the success probability of a specific arm.
        
        Args:
            action (int): Index of the arm
        
        Returns:
            float: Probability of the arm
        """
        return self.probs[action]
    
    def __str__(self):
        """String representation of the bandit."""
        return f"Bandit(n_arms={self.n_arms}, probs={self.probs}, optimal_arm={self.optimal_arm})"
    
    def __repr__(self):
        """Detailed representation of the bandit."""
        return self.__str__()
