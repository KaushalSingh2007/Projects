"""
Main experiment runner with multiple configurations.
"""

import numpy as np
from typing import Dict, List
import pandas as pd
from src.training_loop import run_simulation
from src.utils import compute_statistics, equilibrium_analysis

def run_topology_comparison(n_agents: int = 50,
                           n_steps: int = 10000,
                           n_runs: int = 3,
                           **kwargs) -> pd.DataFrame:
    """
    Compare all three topologies.
    
    Args:
        n_agents: Number of agents
        n_steps: Number of simulation steps
        n_runs: Number of random seeds
        **kwargs: Additional config arguments
    
    Returns:
        DataFrame with results
    """
    
    topologies = ['random', 'small-world', 'scale-free']
    results = []
    
    for topology in topologies:
        for run in range(n_runs):
            history, agents, G = run_simulation(
                n_agents=n_agents,
                topology=topology,
                n_steps=n_steps,
                seed=run,
                verbose=False,
                **kwargs
            )
            
            stats = compute_statistics(history)
            results.append({
                'topology': topology,
                'run': run,
                **{f'{k}_{m}': v for k, v in stats.items()
                   for m in ['mean', 'std', 'final', 'change']}
            })
    
    return pd.DataFrame(results)

def run_parameter_sweep(param_name: str,
                       param_values: List[float],
                       n_runs: int = 3,
                       **base_config) -> pd.DataFrame:
    """
    Sweep single parameter.
    
    Args:
        param_name: Name of parameter to vary
        param_values: List of values to test
        n_runs: Number of random seeds
        **base_config: Base configuration
    
    Returns:
        DataFrame with results
    """
    
    results = []
    
    for param_val in param_values:
        for run in range(n_runs):
            config = base_config.copy()
            config[param_name] = param_val
            config['seed'] = run
            config['verbose'] = False
            
            history, agents, G = run_simulation(**config)
            stats = compute_statistics(history)
            
            results.append({
                param_name: param_val,
                'run': run,
                **{f'{k}_{m}': v for k, v in stats.items()
                   for m in ['mean', 'std', 'final', 'change']}
            })
    
    return pd.DataFrame(results)