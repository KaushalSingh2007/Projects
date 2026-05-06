"""
Ablation studies to understand component importance.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.training_loop import run_simulation
from src.utils import compute_statistics

def ablation_global_reward(n_agents: int = 25,
                          n_steps: int = 2000,
                          n_runs: int = 2):
    """
    Study effect of global vs local rewards.
    
    Args:
        n_agents: Number of agents
        n_steps: Simulation steps
        n_runs: Runs per configuration
    
    Returns:
        DataFrame with results
    """
    
    configs = [
        {'name': 'Local Only', 'alpha': 0.0, 'beta': 1.0},
        {'name': 'Local + Weak Global', 'alpha': 0.5, 'beta': 0.5},
        {'name': 'Local + Medium Global', 'alpha': 1.0, 'beta': 1.0},
        {'name': 'Local + Strong Global', 'alpha': 2.0, 'beta': 2.0},
    ]
    
    results = []
    
    for config in configs:
        coop_finals = []
        entropy_finals = []
        
        for seed in range(n_runs):
            history, _, _ = run_simulation(
                n_agents=n_agents,
                topology='small-world',
                n_steps=n_steps,
                alpha=config['alpha'],
                beta=config['beta'],
                seed=seed,
                verbose=False
            )
            coop_finals.append(history['cooperation'][-1])
            entropy_finals.append(history['entropy'][-1])
        
        results.append({
            'config': config['name'],
            'coop_mean': np.mean(coop_finals),
            'coop_std': np.std(coop_finals),
            'entropy_mean': np.mean(entropy_finals),
            'entropy_std': np.std(entropy_finals),
        })
    
    df = pd.DataFrame(results)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    x = np.arange(len(df))
    ax1.bar(x, df['coop_mean'], yerr=df['coop_std'], capsize=5, alpha=0.7)
    ax1.set_xlabel('Configuration', fontsize=11)
    ax1.set_ylabel('Final Cooperation', fontsize=11)
    ax1.set_title('Global Reward Ablation', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['config'], rotation=15, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    ax2.bar(x, df['entropy_mean'], yerr=df['entropy_std'], capsize=5, alpha=0.7, color='orange')
    ax2.set_xlabel('Configuration', fontsize=11)
    ax2.set_ylabel('Final Entropy', fontsize=11)
    ax2.set_title('Entropy by Configuration', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['config'], rotation=15, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    return df

def ablation_learning_rate(n_agents: int = 25,
                          n_steps: int = 2000,
                          n_runs: int = 2):
    """
    Study effect of learning rate on convergence.
    
    Args:
        n_agents: Number of agents
        n_steps: Simulation steps
        n_runs: Runs per configuration
    """
    
    learning_rates = [0.01, 0.02, 0.05, 0.1, 0.2]
    convergence_speeds = []
    final_cooperations = []
    
    for lr in learning_rates:
        speed_values = []
        coop_values = []
        
        for seed in range(n_runs):
            history, _, _ = run_simulation(
                n_agents=n_agents,
                topology='small-world',
                n_steps=n_steps,
                learning_rate=lr,
                alpha=1.0,
                beta=1.0,
                seed=seed,
                verbose=False
            )
            
            # Convergence speed: steps to reach 90% of final cooperation
            final_coop = history['cooperation'][-1]
            threshold = 0.9 * final_coop
            convergence_idx = np.where(history['cooperation'] >= threshold)[0]
            if len(convergence_idx) > 0:
                speed_values.append(convergence_idx[0])
            else:
                speed_values.append(n_steps)
            
            coop_values.append(final_coop)
        
        convergence_speeds.append(np.mean(speed_values))
        final_cooperations.append(np.mean(coop_values))
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(learning_rates, convergence_speeds, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax1.set_xlabel('Learning Rate', fontsize=11)
    ax1.set_ylabel('Steps to 90% Final Cooperation', fontsize=11)
    ax1.set_title('Convergence Speed vs Learning Rate', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(learning_rates, final_cooperations, 'o-', linewidth=2, markersize=8, color='#F18F01')
    ax2.set_xlabel('Learning Rate', fontsize=11)
    ax2.set_ylabel('Final Cooperation', fontsize=11)
    ax2.set_title('Final Cooperation vs Learning Rate', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    print("Running ablation studies...")
    df = ablation_global_reward()
    print(df)
    ablation_learning_rate()