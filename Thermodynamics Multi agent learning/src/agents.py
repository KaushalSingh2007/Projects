"""
Agent implementation with policy gradient learning.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple

class Agent:
    """
    Single agent with stochastic policy over discrete strategies.
    
    Strategies:
    - 0: Cooperate
    - 1: Defect
    - 2: Neutral
    """
    
    STRATEGY_NAMES = {0: 'Cooperate', 1: 'Defect', 2: 'Neutral'}
    
    def __init__(self, 
                 agent_id: int, 
                 n_actions: int = 3, 
                 init_std: float = 0.1, 
                 rng: np.random.Generator = None):
        """
        Initialize agent.
        
        Args:
            agent_id: Unique agent identifier
            n_actions: Number of strategies (default 3)
            init_std: Standard deviation for initialization of policy logits
            rng: Random number generator instance
        """
        self.agent_id = agent_id
        self.n_actions = n_actions
        self.rng = rng if rng is not None else np.random.default_rng()
        
        # Policy parameters (logits) - detached from gradient computation
        init_logits = self.rng.normal(0, init_std, size=(n_actions,)).astype(np.float32)
        self.theta = torch.tensor(init_logits, dtype=torch.float32, requires_grad=False)
        
        # Value function baseline
        self.value = torch.tensor(0.0, dtype=torch.float32, requires_grad=False)
        
        # History tracking
        self.reward_history = []
        self.action_history = []
        self.value_history = []

    def policy(self) -> np.ndarray:
        """
        Compute softmax action probabilities.
        
        Returns:
            Array of shape (n_actions,) with action probabilities
        """
        logits = self.theta.detach()
        probs = F.softmax(logits, dim=0)
        return probs.numpy()

    def sample_action(self) -> int:
        """
        Sample action from current stochastic policy.
        
        Returns:
            Action index in {0, 1, 2}
        """
        probs = self.policy()
        action = int(np.random.choice(self.n_actions, p=probs))
        self.action_history.append(action)
        return action

    def update_policy(self, action: int, advantage: float, lr: float) -> None:
        """
        Policy gradient update with advantage weighting.
        
        Update rule:
            θ_k ← θ_k + lr * A * [I(a=k) - π(k)]
        
        Args:
            action: Action that was taken
            advantage: Advantage estimate (R - baseline)
            lr: Policy learning rate
        """
        probs = F.softmax(self.theta.detach(), dim=0).numpy()
        
        # Compute gradient: [I(a=k) - π(k)] for each action k
        grad = np.zeros(self.n_actions, dtype=np.float32)
        for k in range(self.n_actions):
            indicator = 1.0 if k == action else 0.0
            grad[k] = indicator - probs[k]
        
        # Parameter update (numpy)
        self.theta.data = self.theta.data + lr * advantage * torch.tensor(grad, dtype=torch.float32)

    def update_value(self, reward: float, lr: float) -> None:
        """
        Update value baseline towards observed reward.
        
        Update rule:
            v ← v + lr * (R - v)
        
        Args:
            reward: Observed reward
            lr: Value function learning rate
        """
        delta = float(reward) - self.value.item()
        self.value.data = self.value.data + lr * delta
        self.value_history.append(self.value.item())

    def strategy_probs(self) -> np.ndarray:
        """
        Get current strategy probabilities.
        
        Returns:
            Array of probabilities for each strategy
        """
        return self.policy()

    def get_theta(self) -> np.ndarray:
        """Get policy parameters."""
        return self.theta.detach().numpy().copy()

    def get_value(self) -> float:
        """Get current value baseline."""
        return float(self.value.item())

    def entropy(self) -> float:
        """
        Compute policy entropy (exploration measure).
        
        Returns:
            Entropy value in [0, log(n_actions)]
        """
        probs = self.policy()
        probs = np.clip(probs, 1e-12, 1.0)
        return float(-np.sum(probs * np.log(probs)))

    def reset_history(self) -> None:
        """Clear history buffers."""
        self.reward_history = []
        self.action_history = []
        self.value_history = []

    def __repr__(self) -> str:
        theta_str = ', '.join([f'{x:.3f}' for x in self.get_theta()])
        return f"Agent({self.agent_id}, θ=[{theta_str}], v={self.get_value():.3f})"