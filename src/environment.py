"""
Multi-agent environment with game-theoretic interactions.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, List
from src.agents import Agent

class MultiAgentEnvironment:
    """
    Environment orchestrating agent interactions on a graph.
    
    - Agents play pairwise games with neighbors
    - Rewards combine local payoffs and global cooperation incentives
    - Policies and values updated via RL
    """
    
    STRATEGIES = {0: 'Cooperate', 1: 'Defect', 2: 'Neutral'}
    COOPERATE = 0  # Strategy index for cooperation
    
    def __init__(self,
                 agents: List[Agent],
                 G,
                 payoff_matrix: np.ndarray = None,
                 noise_std: float = 0.1,
                 alpha: float = 1.0,
                 beta: float = 1.0,
                 learning_rate: float = 0.05,
                 value_lr: float = 0.1,
                 rng: np.random.Generator = None):
        """
        Initialize environment.
        
        Args:
            agents: List of Agent objects
            G: NetworkX undirected graph
            payoff_matrix: 3x3 payoff matrix for game
            noise_std: Standard deviation for observation noise
            alpha: Strength of global cooperation incentive
            beta: Weight of global reward signal
            learning_rate: Policy learning rate
            value_lr: Value function learning rate
            rng: Random number generator
        """
        self.agents = agents
        self.G = G
        self.n_agents = len(agents)
        self.noise_std = noise_std
        self.alpha = alpha
        self.beta = beta
        self.learning_rate = learning_rate
        self.value_lr = value_lr
        self.rng = rng if rng is not None else np.random.default_rng()
        
        # Default 3x3 payoff matrix (Prisoner's Dilemma variant)
        if payoff_matrix is None:
            self.P = np.array([
                [3.0, 0.0, 1.0],  # agent plays Cooperate (0)
                [5.0, 1.0, 2.0],  # agent plays Defect (1)
                [1.0, 2.0, 2.0]   # agent plays Neutral (2)
            ], dtype=float)
        else:
            self.P = np.array(payoff_matrix, dtype=float)
        
        # Precompute edge list for efficient sampling
        self.edges = list(self.G.edges())
        if len(self.edges) == 0:
            raise ValueError("Graph must have at least one edge")
        
        # Tracking
        self.step_count = 0
        self.last_step_info = {}
        
        # Storage for rewards (for computing energy)
        self.all_rewards = np.zeros(self.n_agents)

    def compute_population_cooperation(self) -> float:
        """
        Compute fraction of agents with high cooperation probability.
        
        Returns:
            Float in [0, 1]
        """
        coop_probs = np.array([ag.policy()[0] for ag in self.agents])
        return float(np.mean(coop_probs >= 1.0/3.0))

    def step(self) -> Dict:
        """
        Execute one environment step.
        
        1. Sample random edge (i, j)
        2. Both agents sample actions from policies
        3. Compute rewards (local + global)
        4. Update policies and value functions
        5. Return step statistics
        
        Returns:
            Dictionary with step information
        """
        self.step_count += 1
        
        # 1. Sample random edge
        i, j = self.edges[self.rng.integers(0, len(self.edges))]
        agent_i = self.agents[i]
        agent_j = self.agents[j]
        
        # 2. Action sampling from policies
        a_i = agent_i.sample_action()
        a_j = agent_j.sample_action()
        
        # 3. Local payoffs from game matrix
        R_i_local = float(self.P[a_i, a_j])
        R_j_local = float(self.P[a_j, a_i])
        
        # 4. Global cooperation signal
        C_pop = self.compute_population_cooperation()
        
        # 5. Total rewards with cooperation-specific global incentive
        # Only agents who cooperate receive the global reward
        R_i_global = self.alpha * C_pop if a_i == self.COOPERATE else 0.0
        R_j_global = self.alpha * C_pop if a_j == self.COOPERATE else 0.0
        
        R_i = R_i_local + self.beta * R_i_global
        R_j = R_j_local + self.beta * R_j_global
        
        # Store for energy computation
        self.all_rewards[i] = R_i
        self.all_rewards[j] = R_j
        
        # 6. Advantage estimates
        v_i = agent_i.get_value()
        v_j = agent_j.get_value()
        A_i = R_i - v_i
        A_j = R_j - v_j
        
        # 7. Policy updates
        agent_i.update_policy(a_i, A_i, lr=self.learning_rate)
        agent_j.update_policy(a_j, A_j, lr=self.learning_rate)
        
        # 8. Value function updates
        agent_i.update_value(R_i, lr=self.value_lr)
        agent_j.update_value(R_j, lr=self.value_lr)
        
        # Store step information
        info = {
            'agent_i': i,
            'agent_j': j,
            'action_i': a_i,
            'action_j': a_j,
            'reward_i': R_i,
            'reward_j': R_j,
            'reward_i_local': R_i_local,
            'reward_j_local': R_j_local,
            'cooperation_level': C_pop,
            'strategy_names': (self.STRATEGIES[a_i], self.STRATEGIES[a_j])
        }
        
        self.last_step_info = info
        return info

    def get_agent_policies(self) -> np.ndarray:
        """
        Get all agent policies.
        
        Returns:
            Array of shape (n_agents, 3) with action probabilities
        """
        return np.array([ag.policy() for ag in self.agents])

    def get_agent_values(self) -> np.ndarray:
        """Get all agent value baselines."""
        return np.array([ag.get_value() for ag in self.agents])

    def reset(self) -> None:
        """Reset environment statistics."""
        self.step_count = 0
        self.last_step_info = {}
        self.all_rewards = np.zeros(self.n_agents)
        for agent in self.agents:
            agent.reset_history()