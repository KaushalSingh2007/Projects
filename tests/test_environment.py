"""
Unit tests for MultiAgentEnvironment class.
"""

import numpy as np
import networkx as nx
import pytest
import torch
from src.agents import Agent
from src.environment import MultiAgentEnvironment
from src.graph_model import build_graph

class TestEnvironment:
    
    @pytest.fixture
    def env(self):
        """Create test environment."""
        rng = np.random.default_rng(42)
        n_agents = 10
        agents = [Agent(i, rng=rng) for i in range(n_agents)]
        G = build_graph(n_agents, topology='random', seed=42)
        return MultiAgentEnvironment(agents, G, rng=rng)
    
    def test_initialization(self, env):
        """Test environment setup."""
        assert env.n_agents == 10
        assert len(env.edges) > 0
        assert env.P.shape == (3, 3)
    
    def test_step(self, env):
        """Test single step execution."""
        info = env.step()
        
        assert 'agent_i' in info
        assert 'agent_j' in info
        assert 'reward_i' in info
        assert 'reward_j' in info
        assert 0 <= info['cooperation_level'] <= 1
        assert 'strategy_names' in info
    
    def test_agent_policies(self, env):
        """Test getting agent policies."""
        policies = env.get_agent_policies()
        assert policies.shape == (10, 3)
        assert np.allclose(np.sum(policies, axis=1), 1.0)
    
    def test_agent_values(self, env):
        """Test getting agent values."""
        values = env.get_agent_values()
        assert len(values) == 10
    
    def test_cooperation_computation(self):
        """Test cooperation level computation."""
        rng = np.random.default_rng(42)
        n_agents = 100
        agents = [Agent(i, rng=rng) for i in range(n_agents)]
        G = build_graph(n_agents, topology='random', seed=42)
        env = MultiAgentEnvironment(agents, G, rng=rng)
        
        # Force all agents to high cooperation probability
        for agent in agents:
            agent.theta.data = torch.tensor([10.0, -10.0, -10.0], dtype=torch.float32)
        
        coop = env.compute_population_cooperation()
        assert coop > 0.9
    
    def test_multiple_steps(self, env):
        """Test multiple consecutive steps."""
        for _ in range(100):
            info = env.step()
            assert 'reward_i' in info
        
        assert env.step_count == 100

if __name__ == '__main__':
    pytest.main([__file__, '-v'])