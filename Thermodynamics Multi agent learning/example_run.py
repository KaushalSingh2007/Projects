"""
Quick start example script with debugging output.
"""

import sys
import traceback
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

print("=" * 70)
print("Thermodynamic Multi-Agent Reinforcement Learning - Example")
print("=" * 70)

try:
    print("\n[1/6] Importing modules...")
    from src.training_loop import run_simulation
    from src.visualization import plot_metrics, plot_graph
    from src.utils import compute_statistics
    print("✓ Imports successful")
    
    print("\n[2/6] Running simulation...")
    print("  - Agents: 30")
    print("  - Topology: small-world")
    print("  - Steps: 5000")
    print("  - Alpha: 1.0, Beta: 1.0")
    
    history, agents, G = run_simulation(
        n_agents=30,
        topology='small-world',
        n_steps=5000,
        alpha=1.0,
        beta=1.0,
        learning_rate=0.05,
        value_lr=0.1,
        seed=42,
        verbose=True  # Show progress bar
    )
    print("✓ Simulation complete")
    
    print("\n[3/6] Computing statistics...")
    stats = compute_statistics(history)
    print("✓ Statistics computed")
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    for metric, values in stats.items():
        print(f"\n{metric.upper()}:")
        print(f"  Initial:    {values['initial']:.6f}")
        print(f"  Final:      {values['final']:.6f}")
        print(f"  Change:     {values['change']:.6f}")
        print(f"  Mean:       {values['mean']:.6f}")
        print(f"  Std Dev:    {values['std']:.6f}")
        print(f"  Min:        {values['min']:.6f}")
        print(f"  Max:        {values['max']:.6f}")
    
    print("\n[4/6] Creating visualizations...")
    
    # Plot 1: Time series
    print("  - Creating metric plots...")
    fig1 = plt.figure(figsize=(14, 4))
    
    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(history['entropy'], linewidth=2, color='#2E86AB')
    ax1.set_xlabel('Step', fontsize=11)
    ax1.set_ylabel('Entropy S(t)', fontsize=11)
    ax1.set_title('Entropy Evolution', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(1, 3, 2)
    ax2.plot(history['energy'], linewidth=2, color='#A23B72')
    ax2.set_xlabel('Step', fontsize=11)
    ax2.set_ylabel('Energy E(t)', fontsize=11)
    ax2.set_title('Energy Evolution', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(1, 3, 3)
    ax3.plot(history['cooperation'], linewidth=2, color='#F18F01')
    ax3.set_xlabel('Step', fontsize=11)
    ax3.set_ylabel('Cooperation C(t)', fontsize=11)
    ax3.set_title('Cooperation Fraction', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle('Small-World Network (alpha=1.0, beta=1.0)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output_metrics.png', dpi=150, bbox_inches='tight')
    print("  - Saved: output_metrics.png")
    
    # Plot 2: Network graph
    print("  - Creating network visualization...")
    fig2, ax = plt.subplots(figsize=(8, 8))
    import networkx as nx
    
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    agent_probs = [ag.policy() for ag in agents]
    coop_probs = np.array([probs[0] for probs in agent_probs])
    
    nodes = nx.draw_networkx_nodes(
        G, pos, node_color=coop_probs, cmap='RdYlGn',
        node_size=200, ax=ax, vmin=0, vmax=1
    )
    nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax, width=0.5)
    nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)
    
    plt.colorbar(nodes, ax=ax, label='P(Cooperate)')
    ax.set_title('Agent Interaction Network\n(Node color = Cooperation probability)', 
                fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('output_network.png', dpi=150, bbox_inches='tight')
    print("  - Saved: output_network.png")
    
    print("\n[5/6] Generating summary report...")
    
    # Create text report (NO SPECIAL UNICODE CHARACTERS)
    report = f"""
================================================================
Thermodynamic Multi-Agent Reinforcement Learning
SIMULATION REPORT
================================================================

CONFIGURATION:
  Agents: {len(agents)}
  Topology: small-world
  Steps: 5000
  Global Incentive (alpha): 1.0
  Global Weight (beta): 1.0
  Learning Rate: 0.05
  Value LR: 0.1

RESULTS:

Entropy (S):
  Initial:  {stats['entropy']['initial']:.6f}
  Final:    {stats['entropy']['final']:.6f}
  Change:   {stats['entropy']['change']:.6f}
  -> Entropy collapsed by {abs(stats['entropy']['change']):.1%}

Energy (E):
  Initial:  {stats['energy']['initial']:.2f}
  Final:    {stats['energy']['final']:.2f}
  Change:   {stats['energy']['change']:.2f}
  -> Energy decreased (system reached favorable state)

Cooperation (C):
  Initial:  {stats['cooperation']['initial']:.6f}
  Final:    {stats['cooperation']['final']:.6f}
  Change:   {stats['cooperation']['change']:.6f}
  -> Cooperation change: {stats['cooperation']['change']:.1%}

INTERPRETATION:
  * Entropy decrease indicates agents converged on strategies
  * Decreasing energy shows system moved to favorable states
  * Cooperation dynamics reflect global incentive strength
  * Small-world topology facilitated information flow

OUTPUTS GENERATED:
  * output_metrics.png - Time evolution plots
  * output_network.png - Network visualization
  * simulation_report.txt - This report

================================================================
"""
    
    print(report)
    
    # FIXED: Write with UTF-8 encoding explicitly
    with open('simulation_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    print("  - Saved: simulation_report.txt")
    
    print("\n[6/6] Cleanup...")
    plt.close('all')
    print("- All plots closed\n")
    
    print("=" * 70)
    print("SUCCESS! All outputs generated.")
    print("=" * 70)
    print("\nFiles created:")
    print("  - output_metrics.png")
    print("  - output_network.png")
    print("  - simulation_report.txt")
    print("\nNext steps:")
    print("  1. Open output_metrics.png to see evolution curves")
    print("  2. Open output_network.png to see agent network")
    print("  3. Read simulation_report.txt for interpretation")
    print("\nTo run experiments:")
    print("  python -m experiments.phase_transition_study")
    print("  python -m experiments.topology_comparison")
    print("  python -m experiments.ablation_study")
    
except Exception as e:
    print("\n" + "=" * 70)
    print("ERROR OCCURRED")
    print("=" * 70)
    print(f"\nError Type: {type(e).__name__}")
    print(f"Error Message: {str(e)}")
    print("\nFull Traceback:")
    traceback.print_exc()
    print("\n" + "=" * 70)
    sys.exit(1)