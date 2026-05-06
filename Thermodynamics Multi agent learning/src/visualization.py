"""
Visualization utilities for analysis and presentation.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Dict, List, Optional, Tuple

sns.set_style("darkgrid")

def plot_metrics(history: Dict,
                figsize: Tuple[int, int] = (12, 4),
                title: str = "Thermodynamic Evolution",
                save_path: str = None) -> None:
    """
    Plot energy, entropy, and cooperation over time.
    
    Args:
        history: Dict with 'energy', 'entropy', 'cooperation' time series
        figsize: Figure size
        title: Plot title
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Entropy
    ax = axes[0]
    ax.plot(history['entropy'], linewidth=2, color='#2E86AB')
    ax.set_xlabel('Step', fontsize=11)
    ax.set_ylabel('Entropy S(t)', fontsize=11)
    ax.set_title('Entropy Evolution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Energy
    ax = axes[1]
    ax.plot(history['energy'], linewidth=2, color='#A23B72')
    ax.set_xlabel('Step', fontsize=11)
    ax.set_ylabel('Energy E(t)', fontsize=11)
    ax.set_title('Energy Evolution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Cooperation
    ax = axes[2]
    ax.plot(history['cooperation'], linewidth=2, color='#F18F01')
    ax.set_xlabel('Step', fontsize=11)
    ax.set_ylabel('Cooperation C(t)', fontsize=11)
    ax.set_title('Cooperation Fraction', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_multiple_runs(histories: List[Dict],
                      labels: List[str] = None,
                      figsize: Tuple[int, int] = (12, 4),
                      save_path: str = None) -> None:
    """
    Compare multiple simulation runs.
    
    Args:
        histories: List of history dicts
        labels: Legend labels
        figsize: Figure size
        save_path: Optional save path
    """
    
    if labels is None:
        labels = [f'Run {i+1}' for i in range(len(histories))]
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
    
    for idx, (history, label, color) in enumerate(zip(histories, labels, colors)):
        axes[0].plot(history['entropy'], label=label, alpha=0.7, color=color)
        axes[1].plot(history['energy'], label=label, alpha=0.7, color=color)
        axes[2].plot(history['cooperation'], label=label, alpha=0.7, color=color)
    
    axes[0].set_title('Entropy')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('S(t)')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title('Energy')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('E(t)')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_title('Cooperation')
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('C(t)')
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_graph(G: nx.Graph,
              agent_probs: Optional[List[np.ndarray]] = None,
              figsize: Tuple[int, int] = (8, 8),
              color_by: str = 'cooperation',
              save_path: str = None) -> None:
    """
    Visualize agent interaction network.
    
    Args:
        G: NetworkX graph
        agent_probs: List of agent policy arrays
        figsize: Figure size
        color_by: 'cooperation', 'entropy', or 'none'
        save_path: Optional save path
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Node colors
    if color_by == 'cooperation' and agent_probs is not None:
        coop_probs = np.array([probs[0] for probs in agent_probs])
        node_colors = coop_probs
        cmap = 'RdYlGn'
        label = 'P(Cooperate)'
    
    elif color_by == 'entropy' and agent_probs is not None:
        entropies = np.array([
            -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)))
            for probs in agent_probs
        ])
        node_colors = entropies
        cmap = 'viridis'
        label = 'Policy Entropy'
    
    else:
        node_colors = 'lightblue'
        label = None
    
    # Draw network
    nodes = nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, cmap=cmap,
        node_size=200, ax=ax, vmin=0
    )
    nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax, width=0.5)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    
    ax.set_title('Agent Interaction Network', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    if label:
        cbar = plt.colorbar(nodes, ax=ax, label=label)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_phase_diagram(param_values: np.ndarray,
                      final_cooperation: np.ndarray,
                      param_name: str = 'Parameter',
                      figsize: Tuple[int, int] = (8, 5),
                      save_path: str = None) -> None:
    """
    Plot phase transition diagram.
    
    Args:
        param_values: Parameter values
        final_cooperation: Final cooperation at each value
        param_name: Name of parameter
        figsize: Figure size
        save_path: Optional save path
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(param_values, final_cooperation, 'o-', linewidth=2.5,
            markersize=10, color='#2E86AB', label='Final Cooperation')
    
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel('Cooperation Fraction', fontsize=12)
    ax.set_title('Phase Transition Diagram', fontsize=14, fontweight='bold')
    ax.set_ylim([-0.05, 1.05])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_heatmap(param1_values: np.ndarray,
                param2_values: np.ndarray,
                results_matrix: np.ndarray,
                param1_name: str = 'Parameter 1',
                param2_name: str = 'Parameter 2',
                metric_name: str = 'Cooperation',
                figsize: Tuple[int, int] = (10, 8),
                save_path: str = None) -> None:
    """
    Plot 2D parameter sweep heatmap.
    
    Args:
        param1_values: First parameter values
        param2_values: Second parameter values
        results_matrix: Shape (len(param1), len(param2))
        param1_name: Name of first parameter
        param2_name: Name of second parameter
        metric_name: Name of metric
        figsize: Figure size
        save_path: Optional save path
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(results_matrix, aspect='auto', cmap='RdYlGn',
                   extent=[param2_values[0], param2_values[-1],
                          param1_values[-1], param1_values[0]])
    
    ax.set_xlabel(param2_name, fontsize=12)
    ax.set_ylabel(param1_name, fontsize=12)
    ax.set_title(f'{metric_name} Phase Diagram', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric_name, fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()