"""
Study phase transitions as a function of parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.training_loop import run_simulation
from src.utils import compute_statistics

def study_alpha_phase_transition(n_agents: int = 25,
                                 n_steps: int = 2000,
                                 n_runs: int = 2):
    """
    Vary alpha (global incentive) and observe cooperation phase transition.
    
    Args:
        n_agents: Number of agents
        n_steps: Simulation steps
        n_runs: Number of runs per configuration
    
    Returns:
        Tuple of (alphas, cooperations, stds)
    """
    
    alphas = np.linspace(0, 3, 10)
    final_cooperations = []
    cooperation_stds = []
    
    for alpha in alphas:
        coop_values = []
        
        for seed in range(n_runs):
            history, _, _ = run_simulation(
                n_agents=n_agents,
                topology='small-world',
                n_steps=n_steps,
                alpha=alpha,
                beta=1.0,
                seed=seed,
                verbose=False
            )
            coop_values.append(history['cooperation'][-1])
        
        final_cooperations.append(np.mean(coop_values))
        cooperation_stds.append(np.std(coop_values))
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(alphas, final_cooperations, yerr=cooperation_stds,
                marker='o', linewidth=2, markersize=8, capsize=5, color='#2E86AB')
    plt.xlabel('Alpha (Global Incentive Strength)', fontsize=12)
    plt.ylabel('Final Cooperation Fraction', fontsize=12)
    plt.title('Phase Transition: Global Incentive Effect', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim([-0.05, 1.05])
    plt.tight_layout()
    plt.show()
    
    return alphas, np.array(final_cooperations), np.array(cooperation_stds)

def study_beta_effect(n_agents: int = 25,
                     n_steps: int = 2000,
                     n_runs: int = 2):
    """
    Vary beta (weight of global signal) and observe cooperation.
    
    Args:
        n_agents: Number of agents
        n_steps: Simulation steps
        n_runs: Number of runs per configuration
    """
    
    betas = np.linspace(0, 2, 10)
    final_cooperations = []
    final_entropies = []
    
    for beta in betas:
        coop_values = []
        entropy_values = []
        
        for seed in range(n_runs):
            history, _, _ = run_simulation(
                n_agents=n_agents,
                topology='small-world',
                n_steps=n_steps,
                alpha=1.0,
                beta=beta,
                seed=seed,
                verbose=False
            )
            coop_values.append(history['cooperation'][-1])
            entropy_values.append(history['entropy'][-1])
        
        final_cooperations.append(np.mean(coop_values))
        final_entropies.append(np.mean(entropy_values))
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(betas, final_cooperations, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax1.set_xlabel('Beta (Global Weight)', fontsize=11)
    ax1.set_ylabel('Final Cooperation', fontsize=11)
    ax1.set_title('Cooperation vs Global Weight')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(betas, final_entropies, 'o-', linewidth=2, markersize=8, color='#F18F01')
    ax2.set_xlabel('Beta (Global Weight)', fontsize=11)
    ax2.set_ylabel('Final Entropy', fontsize=11)
    ax2.set_title('Entropy vs Global Weight')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    print("Running phase transition studies...")
    study_alpha_phase_transition()
    study_beta_effect()