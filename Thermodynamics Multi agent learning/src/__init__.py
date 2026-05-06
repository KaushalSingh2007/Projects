"""
Thermodynamic Multi-Agent Reinforcement Learning Package.

Main modules:
- agents: Agent implementation with policy gradient learning
- environment: Multi-agent interaction environment
- graph_model: Network topology generation
- thermodynamics: Energy and entropy computation
- training_loop: Main simulation runner
- visualization: Plotting and analysis tools
"""

__version__ = "0.1.0"
__author__ = "Kaushal Singh"

from src.agents import Agent
from src.environment import MultiAgentEnvironment
from src.graph_model import build_graph
from src.training_loop import run_simulation

__all__ = [
    'Agent',
    'MultiAgentEnvironment',
    'build_graph',
    'run_simulation',
]