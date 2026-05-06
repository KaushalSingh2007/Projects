"""
Configuration and hyperparameter settings.
"""

from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class SimulationConfig:
    """Configuration for simulation parameters."""
    
    # Network parameters
    n_agents: int = 50
    topology: str = 'small-world'  # 'random', 'small-world', 'scale-free'
    
    # Training parameters
    n_steps: int = 10000
    learning_rate: float = 0.05
    value_lr: float = 0.1
    
    # Game theory parameters
    alpha: float = 1.0  # global cooperation incentive strength
    beta: float = 1.0   # weight of global signal
    noise_std: float = 0.1
    
    # Randomness
    seed: Optional[int] = None
    
    # Payoff matrix for 3-strategy game (Cooperate, Defect, Neutral)
    payoff_matrix: Optional[List[List[int]]] = None
    
    def __post_init__(self):
        if self.payoff_matrix is None:
            # Prisoner's Dilemma variant with Neutral strategy
            self.payoff_matrix = [
                [3, 0, 1],  # agent plays Cooperate (0)
                [5, 1, 2],  # agent plays Defect (1)
                [1, 2, 2]   # agent plays Neutral (2)
            ]

@dataclass
class GraphConfig:
    """Configuration for graph topology."""
    
    n_agents: int = 50
    topology: str = 'small-world'
    
    # Random Erdős-Rényi: edge probability
    random_p: float = 0.1
    
    # Small-world Watts-Strogatz
    smallworld_k: int = 4  # neighbors per node
    smallworld_p: float = 0.1  # rewiring probability
    
    # Scale-free Barabási-Albert
    scalefree_k: int = 3  # attachment degree

@dataclass
class ExperimentConfig:
    """Configuration for experiments."""
    
    n_agents: int = 50
    n_steps: int = 10000
    n_runs: int = 3
    n_parallel_workers: int = 4

# Preset configurations
CONFIGS = {
    'baseline': SimulationConfig(
        n_agents=50,
        topology='random',
        n_steps=10000,
        alpha=1.0,
        beta=1.0
    ),
    'high_cooperation': SimulationConfig(
        n_agents=50,
        topology='small-world',
        n_steps=10000,
        alpha=2.0,
        beta=2.0
    ),
    'defection_dominated': SimulationConfig(
        n_agents=50,
        topology='random',
        n_steps=10000,
        alpha=0.1,
        beta=0.1
    ),
    'scale_free': SimulationConfig(
        n_agents=50,
        topology='scale-free',
        n_steps=10000,
        alpha=1.5,
        beta=1.5
    ),
}