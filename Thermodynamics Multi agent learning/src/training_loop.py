"""
Main training loop and simulation runner.
"""

import numpy as np
from tqdm import tqdm
from typing import Tuple, Dict, List
import networkx as nx

from src.agents import Agent
from src.environment import MultiAgentEnvironment
from src.graph_model import build_graph
from src.thermodynamics import ThermodynamicsTracker

def run_simulation(
    n_agents: int = 50,
    topology: str = 'random',
    n_steps: int = 10000,
    payoff_matrix: np.ndarray = None,
    alpha: float = 1.0,
    beta: float = 1.0,
    noise_std: float = 0.1,
    learning_rate: float = 0.05,
    value_lr: float = 0.1,
    seed: int = None,
    verbose: bool = True
) -> Tuple[Dict, List[Agent], nx.Graph]:
    """
    Run full simulation of thermodynamic multi-agent RL system.
    
    Args:
        n_agents: Number of agents
        topology: Graph topology ('random', 'small-world', 'scale-free')
        n_steps: Number of interaction steps
        payoff_matrix: 3x3 game payoff matrix
        alpha: Global cooperation incentive strength
        beta: Weight of global reward signal
        noise_std: Observation noise standard deviation
        learning_rate: Policy gradient learning rate
        value_lr: Value function learning rate
        seed: Random seed for reproducibility
        verbose: Print progress bar and metrics
    
    Returns:
        Tuple of:
        - history: Dict with time series of 'energy', 'entropy', 'cooperation'
        - agents: List of final agent states
        - G: Interaction graph used
    
    Example:
        >>> history, agents, G = run_simulation(
        ...     n_agents=50, topology='small-world', n_steps=5000
        ... )
        >>> print(f"Final cooperation: {history['cooperation'][-1]:.3f}")
    """
    
    # Initialize
    rng = np.random.default_rng(seed)
    G = build_graph(n_agents, topology=topology, seed=seed)
    agents = [Agent(i, rng=rng) for i in range(n_agents)]
    
    # Environment
    env = MultiAgentEnvironment(
        agents=agents,
        G=G,
        payoff_matrix=payoff_matrix,
        noise_std=noise_std,
        alpha=alpha,
        beta=beta,
        learning_rate=learning_rate,
        value_lr=value_lr,
        rng=rng
    )
    
    # Tracker
    tracker = ThermodynamicsTracker(max_steps=n_steps)
    
    # Simulation loop
    iterator = tqdm(range(n_steps), disable=not verbose, desc='Simulation')
    
    for step in iterator:
        # Environment step
        info = env.step()
        
        # Collect metrics
        agent_probs = env.get_agent_policies()
        rewards = env.get_agent_values()
        actions = np.array([
            ag.action_history[-1] if ag.action_history else 0 
            for ag in agents
        ])
        
        # Record
        tracker.record(rewards, agent_probs, actions)
        
        # Progress update
        if verbose and (step + 1) % 1000 == 0:
            history = tracker.get_history()
            S = history['entropy'][-1] if len(history['entropy']) > 0 else 0
            E = history['energy'][-1] if len(history['energy']) > 0 else 0
            C = history['cooperation'][-1] if len(history['cooperation']) > 0 else 0
            iterator.set_postfix({
                'S': f'{S:.3f}',
                'E': f'{E:.1f}',
                'C': f'{C:.3f}'
            })
    
    history = tracker.get_history()
    return history, agents, G

def run_experiment_sweep(
    param_name: str,
    param_values: List[float],
    base_config: Dict,
    n_seeds: int = 3,
    verbose: bool = True
) -> Dict[float, List[Dict]]:
    """
    Run simulation across parameter range.
    
    Args:
        param_name: Name of parameter to vary
        param_values: List of values to test
        base_config: Base simulation config dict
        n_seeds: Number of random seeds per config
        verbose: Print progress
    
    Returns:
        Dict mapping param_value -> list of histories
    
    Example:
        >>> results = run_experiment_sweep(
        ...     param_name='alpha',
        ...     param_values=[0, 0.5, 1.0, 1.5, 2.0],
        ...     base_config={'n_agents': 50, 'n_steps': 5000}
        ... )
    """
    
    results = {}
    
    for param_val in (tqdm(param_values) if verbose else param_values):
        config = base_config.copy()
        config[param_name] = param_val
        
        histories = []
        for seed in range(n_seeds):
            config['seed'] = seed
            history, _, _ = run_simulation(**config, verbose=False)
            histories.append(history)
        
        results[param_val] = histories
    
    return results