import numpy as np

class Bandit:
    def __init__(self, probs):
        self.probs = np.array(probs, dtype=np.float64)
        self.n_arms = len(probs)
        self.optimal_arm = np.argmax(probs)
        self.optimal_prob = np.max(probs)
    
    def pull(self, action):
        if action < 0 or action >= self.n_arms:
            raise ValueError(f"Invalid action {action}. Must be in range [0, {self.n_arms-1}]")
        # Bernoulli reward: return 1 with probability probs[action]
        random_value = np.random.random()
        return 1 if random_value < self.probs[action] else 0
    
    def get_optimal_arm(self):
        return self.optimal_arm
    
    def get_optimal_prob(self):
        return self.optimal_prob
    
    def get_arm_prob(self, action):

        return self.probs[action]
    
    def __str__(self):
        return f"Bandit(n_arms={self.n_arms}, probs={self.probs}, optimal_arm={self.optimal_arm})"
    
    def __repr__(self):
        """Detailed representation of the bandit."""
        return self.__str__()
