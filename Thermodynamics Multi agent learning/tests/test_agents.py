"""
Unit tests for Agent class.
"""

import numpy as np
import pytest
import torch
from src.agents import Agent

class TestAgent:
    
    @pytest.fixture
    def agent(self):
        """Create test agent."""
        rng = np.random.default_rng(42)
        return Agent(0, rng=rng)
    
    def test_initialization(self, agent):
        """Test agent creation."""
        assert agent.agent_id == 0
        assert agent.n_actions == 3
        assert agent.theta.shape == (3,)
        assert agent.get_value() == 0.0
    
    def test_policy_softmax(self, agent):
        """Test policy is valid probability distribution."""
        policy = agent.policy()
        assert len(policy) == 3
        assert np.allclose(np.sum(policy), 1.0)
        assert np.all(policy >= 0)
        assert np.all(policy <= 1)
    
    def test_action_sampling(self, agent):
        """Test action sampling."""
        actions = [agent.sample_action() for _ in range(100)]
        assert all(a in [0, 1, 2] for a in actions)
        assert len(agent.action_history) == 100
    
    def test_policy_update(self, agent):
        """Test policy gradient update."""
        theta_before = agent.get_theta().copy()
        agent.update_policy(action=0, advantage=1.0, lr=0.1)
        theta_after = agent.get_theta()
        assert not np.allclose(theta_before, theta_after)
    
    def test_value_update(self, agent):
        """Test value function update."""
        v_before = agent.get_value()
        agent.update_value(reward=5.0, lr=0.1)
        v_after = agent.get_value()
        assert v_after > v_before
        assert np.isclose(v_after, v_before + 0.1 * (5.0 - v_before))
    
    def test_entropy(self, agent):
        """Test policy entropy computation."""
        entropy = agent.entropy()
        assert 0 <= entropy <= np.log(3) + 1e-6
    
    def test_strategy_probs(self, agent):
        """Test strategy probabilities."""
        probs = agent.strategy_probs()
        assert len(probs) == 3
        assert np.allclose(np.sum(probs), 1.0)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])