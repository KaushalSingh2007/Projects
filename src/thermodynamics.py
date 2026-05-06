"""
Thermodynamic quantities and metrics.
"""

import numpy as np
from typing import List, Tuple

def compute_energy(rewards: np.ndarray) -> float:
    """
    Compute system energy (negative sum of rewards).
    
    Physical interpretation:
    - High payoff → Low energy → Favorable state
    - Low payoff → High energy → Unfavorable state
    
    Args:
        rewards: Array of agent rewards
    
    Returns:
        Float, system energy E = -Σ R_i
    """
    return -np.sum(rewards)

def compute_entropy(agent_probs: List[np.ndarray]) -> float:
    """
    Compute population entropy from strategy distributions.
    
    Formula:
        S = -Σ_a p_a * log(p_a)
    
    where p_a is the population mean probability of action a
    
    Args:
        agent_probs: List of policy arrays (one per agent, shape (3,))
    
    Returns:
        Float, entropy in [0, log(3)] ≈ [0, 1.099]
    """
    # Stack into (n_agents, n_actions) matrix
    probs = np.stack(agent_probs)
    
    # Compute population mean policy
    mean_probs = np.mean(probs, axis=0)
    
    # Avoid log(0)
    mean_probs = np.clip(mean_probs, 1e-12, 1.0)
    
    # Shannon entropy
    entropy = -np.sum(mean_probs * np.log(mean_probs))
    return float(entropy)

def compute_cooperation_fraction(actions: np.ndarray) -> float:
    """
    Compute fraction of agents choosing Cooperate (action 0).
    
    Args:
        actions: Array of action indices
    
    Returns:
        Float in [0, 1]
    """
    if len(actions) == 0:
        return 0.0
    return float(np.mean(np.array(actions) == 0))

def compute_strategy_distribution(agent_probs: List[np.ndarray]) -> np.ndarray:
    """
    Compute empirical strategy distribution (population fractions).
    
    Args:
        agent_probs: List of policy arrays
    
    Returns:
        Array of shape (3,) with population fractions [p_C, p_D, p_N]
    """
    probs = np.stack(agent_probs)
    return np.mean(probs, axis=0)

def compute_free_energy(energy: float, 
                       entropy: float, 
                       temperature: float = 1.0) -> float:
    """
    Compute free energy-like quantity: F = E - T*S
    
    Balances energy minimization and entropy maximization
    
    Args:
        energy: System energy
        entropy: System entropy
        temperature: Inverse of exploration parameter
    
    Returns:
        Float, free energy F
    """
    return energy - temperature * entropy

class ThermodynamicsTracker:
    """
    Track thermodynamic quantities over simulation.
    """
    
    def __init__(self, max_steps: int = 10000):
        """
        Initialize tracker.
        
        Args:
            max_steps: Maximum number of steps to track
        """
        self.max_steps = max_steps
        self.energy_history = np.zeros(max_steps)
        self.entropy_history = np.zeros(max_steps)
        self.cooperation_history = np.zeros(max_steps)
        self.step_count = 0
    
    def record(self, 
               rewards: np.ndarray, 
               agent_probs: List[np.ndarray], 
               actions: np.ndarray) -> None:
        """
        Record metrics at current step.
        
        Args:
            rewards: Array of agent rewards
            agent_probs: List of agent policies
            actions: Array of agent actions
        """
        if self.step_count >= self.max_steps:
            return
        
        self.energy_history[self.step_count] = compute_energy(rewards)
        self.entropy_history[self.step_count] = compute_entropy(agent_probs)
        self.cooperation_history[self.step_count] = compute_cooperation_fraction(actions)
        self.step_count += 1
    
    def get_history(self) -> dict:
        """
        Get recorded history up to current step.
        
        Returns:
            Dict with 'energy', 'entropy', 'cooperation' arrays
        """
        return {
            'energy': self.energy_history[:self.step_count].copy(),
            'entropy': self.entropy_history[:self.step_count].copy(),
            'cooperation': self.cooperation_history[:self.step_count].copy(),
        }