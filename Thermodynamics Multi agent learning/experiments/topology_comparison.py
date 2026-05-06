"""
Compare behavior across different network topologies.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.training_loop import run_simulation
from src.visualization import plot_multiple_runs

def compare_topologies(n_agents: int = 50,
                      n_steps: int = 5000,
                      n_runs: int = 3,
                      alpha: float = 1.0,
                      beta: float = 1.0):
    """
    Compare all three topologies side-by-side.
    
    Args:
        n_agents: Number of agents
        n_steps: Simulation steps
        n_runs: Number of runs per topology
        alpha: Global cooperation incentive
        beta: Global signal weight
    
    Returns:
        Dict mapping topology -> list of histories
    """
    
    topologies = ['random', 'small-world', 'scale-free']
    all_histories = {topo: [] for topo in topologies}
    
    for topology in topologies:
        print(f"Running {topology} topology...")
        for seed in range(n_runs):
            history, _, _ = run_simulation(
                n_agents=n_agents,
                topology=topology,
                n_steps=n_steps,
                alpha=alpha,
                beta=beta,
                seed=seed,
                verbose=False
            )
            all_histories[topology].append(history)
    
    # Plot averaged results per topology
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (topology, histories) in enumerate(all_histories.items()):
        ax = axes[idx]
        
        # Average
        avg_entropy = np.mean([h['entropy'] for h in histories], axis=0)
        avg_coop = np.mean([h['cooperation'] for h in histories], axis=0)
        
        ax_twin = ax.twinx()
        
        line1 = ax.plot(avg_entropy, 'b-', linewidth=2, label='Entropy')
        line2 = ax_twin.plot(avg_coop, 'r-', linewidth=2, label='Cooperation')
        
        ax.set_xlabel('Step', fontsize=11)
        ax.set_ylabel('Entropy', color='b', fontsize=11)
        ax_twin.set_ylabel('Cooperation', color='r', fontsize=11)
        ax.set_title(f'{topology.title()} Network', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Color y-axis labels
        ax.tick_params(axis='y', labelcolor='b')
        ax_twin.tick_params(axis='y', labelcolor='r')
    
    plt.suptitle('Topology Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return all_histories

if __name__ == '__main__':
    compare_topologies()