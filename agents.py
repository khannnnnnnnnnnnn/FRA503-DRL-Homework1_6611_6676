import numpy as np


class EpsilonGreedyAgent:    
    def __init__(self, n_actions, epsilon):
        self.n_actions = n_actions
        self.epsilon = epsilon
        # Learnable parameters
        self.Q = np.zeros(n_actions, dtype=np.float64)  # Action-value estimates
        self.N = np.zeros(n_actions, dtype=np.int32)    # Action counts
    
    def select_action(self):
        if np.random.random() < self.epsilon:
            # Explore: choose random action
            return np.random.randint(self.n_actions)
        else:
            # Exploit: choose best action (break ties randomly)
            max_value = self.Q.max()
            best_actions = np.flatnonzero(self.Q == max_value)
            return np.random.choice(best_actions)
    
    def update(self, action, reward):
        self.N[action] += 1
        alpha = 1.0 / self.N[action]  # Step size (decreases over time)
        self.Q[action] += alpha * (reward - self.Q[action])
    
    def get_Q_values(self):
        return self.Q.copy()
    
    def get_action_counts(self):
        return self.N.copy()
    
    def __str__(self):
        return f"EpsilonGreedyAgent(n_actions={self.n_actions}, epsilon={self.epsilon})"


class UCBAgent:
    def __init__(self, n_actions, c=2.0):
        self.n_actions = n_actions
        self.c = c
        
        # Learnable parameters
        self.Q = np.zeros(n_actions, dtype=np.float64)  # Action-value estimates
        self.N = np.zeros(n_actions, dtype=np.int32)    # Action counts
        self.t = 0  # Total timesteps
    
    def select_action(self):
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
        self.N[action] += 1
        alpha = 1.0 / self.N[action]  # Step size (decreases over time)
        self.Q[action] += alpha * (reward - self.Q[action])
    
    def get_Q_values(self):
        return self.Q.copy()
    
    def get_action_counts(self):
        return self.N.copy()
    
    def get_ucb_values(self):
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
        return f"UCBAgent(n_actions={self.n_actions}, c={self.c})"
