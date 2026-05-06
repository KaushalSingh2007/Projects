"""
Utility functions for analysis and data handling.
"""

import numpy as np
from typing import Dict, List, Tuple
import json
from pathlib import Path

def save_history(history: Dict, filepath: str) -> None:
    """
    Save history dictionary to JSON file.
    
    Args:
        history: Dict with numpy arrays
        filepath: Output file path
    """
    data = {k: v.tolist() if isinstance(v, np.ndarray) else v 
            for k, v in history.items()}
    with open(filepath, 'w') as f:
        json.dump(data, f)

def load_history(filepath: str) -> Dict:
    """
    Load history from JSON file.
    
    Args:
        filepath: Input file path
    
    Returns:
        Dict with numpy arrays
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    return {k: np.array(v) for k, v in data.items()}

def compute_statistics(history: Dict) -> Dict:
    """
    Compute summary statistics from history.
    
    Args:
        history: History dict
    
    Returns:
        Dict with statistics for each metric
    """
    
    stats = {}
    for key in ['entropy', 'energy', 'cooperation']:
        if key in history:
            data = history[key]
            stats[key] = {
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'final': float(data[-1]),
                'initial': float(data[0]),
                'change': float(data[-1] - data[0]),
            }
    
    return stats

def detect_phase_transition(data: np.ndarray,
                           window_size: int = 100) -> Tuple[int, float]:
    """
    Detect sudden phase transition in time series.
    
    Uses rolling variance of differences to find abrupt changes.
    
    Args:
        data: Time series data
        window_size: Rolling window size
    
    Returns:
        Tuple of (step_index, magnitude)
    """
    if len(data) < window_size:
        return 0, 0.0
    
    # Compute rolling variance of differences
    rolling_var = np.convolve(
        np.abs(np.diff(data)), 
        np.ones(window_size)/window_size, 
        mode='valid'
    )
    
    # Find maximum variance point
    max_idx = np.argmax(rolling_var) + window_size
    max_magnitude = rolling_var[max(0, max_idx - window_size)]
    
    return max_idx, float(max_magnitude)

def equilibrium_analysis(history: Dict,
                        burnin: int = 1000) -> Dict:
    """
    Analyze equilibrium properties (post-burnin phase).
    
    Args:
        history: History dict
        burnin: Number of initial steps to discard
    
    Returns:
        Dict with equilibrium statistics
    """
    
    analysis = {}
    for key in history.keys():
        data = history[key][burnin:]
        if len(data) > 0:
            mean = np.mean(data)
            analysis[key] = {
                'eq_mean': float(mean),
                'eq_std': float(np.std(data)),
                'eq_min': float(np.min(data)),
                'eq_max': float(np.max(data)),
                'cv': float(np.std(data) / mean) if mean != 0 else 0.0
            }
    
    return analysis
    