"""
Graph construction utilities for network topologies.
"""

import networkx as nx
from typing import Optional, Dict

def build_graph(n_agents: int,
                topology: str = 'random',
                p: float = 0.1,
                k: int = 4,
                seed: Optional[int] = None) -> nx.Graph:
    """
    Generate interaction graph with specified topology.
    
    Args:
        n_agents: Number of nodes (agents)
        topology: Graph type - 'random', 'small-world', or 'scale-free'
        p: 
            - Random: edge probability (Erdős-Rényi)
            - Small-world: rewiring probability (Watts-Strogatz)
        k:
            - Small-world: number of neighbors per node
            - Scale-free: new nodes attach to k existing nodes
        seed: Random seed for reproducibility
    
    Returns:
        NetworkX Graph object
    
    Raises:
        ValueError: If topology not recognized
    """
    
    if topology == 'random':
        # Erdős-Rényi random graph
        G = nx.erdos_renyi_graph(n_agents, p, seed=seed)
    
    elif topology == 'small-world':
        # Watts-Strogatz small-world (ring + rewiring)
        G = nx.watts_strogatz_graph(n_agents, k, p, seed=seed)
    
    elif topology == 'scale-free':
        # Barabási-Albert preferential attachment
        G = nx.barabasi_albert_graph(n_agents, k, seed=seed)
    
    else:
        raise ValueError(f"Unknown topology: {topology}. "
                        "Choose: 'random', 'small-world', 'scale-free'")
    
    # Ensure connectivity
    if not nx.is_connected(G):
        # Get largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    
    return G

def get_network_statistics(G: nx.Graph) -> Dict:
    """
    Compute network topology statistics.
    
    Args:
        G: NetworkX graph
    
    Returns:
        Dictionary with network metrics
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    stats = {
        'n_nodes': n,
        'n_edges': m,
        'density': nx.density(G),
        'avg_degree': 2 * m / n if n > 0 else 0,
        'clustering_coeff': nx.average_clustering(G),
    }
    
    if nx.is_connected(G):
        stats['diameter'] = nx.diameter(G)
        stats['avg_shortest_path'] = nx.average_shortest_path_length(G)
    else:
        stats['diameter'] = float('inf')
        stats['avg_shortest_path'] = float('inf')
    
    return stats