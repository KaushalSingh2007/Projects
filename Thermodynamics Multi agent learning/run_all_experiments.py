"""
Run all experiments and save visualizations without popup windows.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from src.training_loop import run_simulation

def run_all_experiments():
    """
    Run all experiments and save results without showing plots.
    """
    print("🚀 Running All Experiments - No Popups Mode")
    print("=" * 60)
    
    # 1. Final Phase Transition Study
    print("\n1️⃣ Running Final Phase Transition Study...")
    print("-" * 40)
    
    alphas = np.linspace(0, 3, 10)
    n_agents = 25
    n_steps = 3000
    n_runs = 2
    
    final_cooperations = []
    cooperation_stds = []
    
    for i, alpha in enumerate(alphas):
        print(f"  α={alpha:.2f} ({i+1}/{len(alphas)})...", end=" ")
        
        coop_values = []
        for run in range(n_runs):
            try:
                history, _, _ = run_simulation(
                    n_agents=n_agents,
                    topology='random',
                    n_steps=n_steps,
                    alpha=alpha,
                    beta=1.0,
                    seed=run,
                    verbose=False
                )
                
                final_coop = history['cooperation'][-1]
                coop_values.append(final_coop)
                
            except Exception as e:
                print(f"ERROR: {e}", end=" ")
                coop_values.append(0.0)
        
        mean_coop = np.mean(coop_values)
        std_coop = np.std(coop_values)
        final_cooperations.append(mean_coop)
        cooperation_stds.append(std_coop)
        
        print(f"C={mean_coop:.3f}±{std_coop:.3f}")
    
    # Save phase transition plot
    plt.figure(figsize=(12, 8))
    plt.errorbar(alphas, final_cooperations, yerr=cooperation_stds, 
                marker='o', linewidth=2, markersize=8, capsize=5,
                color='#2E86AB', ecolor='#A23B72')
    plt.xlabel('Alpha (Global Incentive Strength)', fontsize=12)
    plt.ylabel('Final Cooperation Fraction', fontsize=12)
    plt.title('Phase Transition: Global Incentive Effect', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim([-0.05, 1.05])
    plt.tight_layout()
    plt.savefig('phase_transition_final.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: phase_transition_final.png")
    
    # 2. Topology Comparison
    print("\n2️⃣ Running Topology Comparison...")
    print("-" * 40)
    
    topologies = ['random', 'small-world', 'scale-free']
    topology_results = {}
    
    for topology in topologies:
        print(f"  Testing {topology}...", end=" ")
        
        coop_values = []
        for run in range(2):
            try:
                history, _, _ = run_simulation(
                    n_agents=25,
                    topology=topology,
                    n_steps=2000,
                    alpha=1.0,
                    beta=1.0,
                    seed=run,
                    verbose=False
                )
                
                final_coop = history['cooperation'][-1]
                coop_values.append(final_coop)
                
            except Exception as e:
                print(f"ERROR: {e}", end=" ")
                coop_values.append(0.0)
        
        mean_coop = np.mean(coop_values)
        topology_results[topology] = mean_coop
        print(f"C={mean_coop:.3f}")
    
    # Save topology comparison plot
    plt.figure(figsize=(10, 6))
    names = list(topology_results.keys())
    values = list(topology_results.values())
    colors = ['#2E86AB', '#F18F01', '#A23B72']
    
    bars = plt.bar(names, values, color=colors, alpha=0.7)
    plt.ylabel('Final Cooperation', fontsize=12)
    plt.title('Topology Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('topology_comparison_final.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: topology_comparison_final.png")
    
    # 3. Ablation Study
    print("\n3️⃣ Running Ablation Study...")
    print("-" * 40)
    
    configs = [
        {'name': 'Local Only', 'alpha': 0.0, 'beta': 1.0},
        {'name': 'Local + Weak Global', 'alpha': 0.5, 'beta': 0.5},
        {'name': 'Local + Medium Global', 'alpha': 1.0, 'beta': 1.0},
        {'name': 'Local + Strong Global', 'alpha': 2.0, 'beta': 2.0},
    ]
    
    ablation_results = []
    
    for config in configs:
        print(f"  Testing {config['name']}...", end=" ")
        
        coop_values = []
        for run in range(2):
            try:
                history, _, _ = run_simulation(
                    n_agents=25,
                    topology='small-world',
                    n_steps=2000,
                    alpha=config['alpha'],
                    beta=config['beta'],
                    seed=run,
                    verbose=False
                )
                
                final_coop = history['cooperation'][-1]
                coop_values.append(final_coop)
                
            except Exception as e:
                print(f"ERROR: {e}", end=" ")
                coop_values.append(0.0)
        
        mean_coop = np.mean(coop_values)
        ablation_results.append({'name': config['name'], 'coop': mean_coop})
        print(f"C={mean_coop:.3f}")
    
    # Save ablation study plot
    plt.figure(figsize=(12, 6))
    names = [r['name'] for r in ablation_results]
    values = [r['coop'] for r in ablation_results]
    
    bars = plt.bar(names, values, color='#2E86AB', alpha=0.7)
    plt.ylabel('Final Cooperation', fontsize=12)
    plt.title('Global Reward Ablation Study', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=15, ha='right')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('ablation_study_final.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: ablation_study_final.png")
    
    # Summary Report
    print("\n" + "=" * 60)
    print("📊 FINAL EXPERIMENT SUMMARY")
    print("=" * 60)
    
    print(f"\n🎯 Phase Transition Study:")
    print(f"  Best Alpha: {alphas[np.argmax(final_cooperations)]:.2f}")
    print(f"  Best Cooperation: {np.max(final_cooperations):.3f}")
    print(f"  Mean Cooperation: {np.mean(final_cooperations):.3f}")
    
    print(f"\n🌐 Topology Comparison:")
    best_topology = max(topology_results.keys(), key=lambda k: topology_results[k])
    print(f"  Best Topology: {best_topology}")
    print(f"  Best Cooperation: {topology_results[best_topology]:.3f}")
    
    print(f"\n🔬 Ablation Study:")
    best_config = max(ablation_results, key=lambda r: r['coop'])
    print(f"  Best Config: {best_config['name']}")
    print(f"  Best Cooperation: {best_config['coop']:.3f}")
    
    print(f"\n📁 Generated Files:")
    print(f"  ✅ phase_transition_final.png")
    print(f"  ✅ topology_comparison_final.png")
    print(f"  ✅ ablation_study_final.png")
    
    print("\n🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == '__main__':
    run_all_experiments()
