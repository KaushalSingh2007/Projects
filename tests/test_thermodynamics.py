"""
Unit tests for thermodynamic functions.
"""

import numpy as np
import pytest
from src.thermodynamics import (
    compute_energy, compute_entropy, compute_cooperation_fraction,
    compute_strategy_distribution, compute_free_energy,
    ThermodynamicsTracker
)

class TestThermodynamics:
    
    def test_energy_computation(self):
        """Test energy calculation."""
        rewards = np.array([1.0, 2.0, 3.0])
        energy = compute_energy(rewards)
        assert energy == -6.0
    
    def test_energy_empty(self):
        """Test energy with empty array."""
        rewards = np.array([])
        energy = compute_energy(rewards)
        assert energy == 0.0
    
    def test_entropy_uniform(self):
        """Test entropy of uniform distribution."""
        # Uniform distribution: S = log(3)
        probs = [np.array([1/3, 1/3, 1/3]) for _ in range(10)]
        entropy = compute_entropy(probs)
        assert np.isclose(entropy, np.log(3), rtol=1e-5)
    
    def test_entropy_deterministic(self):
        """Test entropy of deterministic distribution."""
        # Deterministic: S = 0
        probs = [np.array([1.0, 0.0, 0.0]) for _ in range(10)]
        entropy = compute_entropy(probs)
        assert np.isclose(entropy, 0.0, atol=1e-10)
    
    def test_cooperation_fraction(self):
        """Test cooperation computation."""
        actions = np.array([0, 0, 1, 2])
        coop = compute_cooperation_fraction(actions)
        assert coop == 0.5
    
    def test_strategy_distribution(self):
        """Test strategy distribution."""
        probs = [np.array([0.5, 0.3, 0.2]) for _ in range(5)]
        dist = compute_strategy_distribution(probs)
        assert len(dist) == 3
        assert np.isclose(np.sum(dist), 1.0)
        assert np.allclose(dist, [0.5, 0.3, 0.2])
    
    def test_free_energy(self):
        """Test free energy calculation."""
        E = 10.0
        S = 1.0
        T = 1.0
        F = compute_free_energy(E, S, T)
        assert F == 9.0
    
    def test_tracker(self):
        """Test thermodynamics tracker."""
        tracker = ThermodynamicsTracker(max_steps=100)
        
        for step in range(50):
            rewards = np.random.rand(10)
            probs = [np.random.dirichlet([1, 1, 1]) for _ in range(10)]
            actions = np.random.choice([0, 1, 2], 10)
            
            tracker.record(rewards, probs, actions)
        
        history = tracker.get_history()
        assert len(history['energy']) == 50
        assert len(history['entropy']) == 50
        assert len(history['cooperation']) == 50
    
    def test_tracker_overflow(self):
        """Test tracker respects max_steps."""
        tracker = ThermodynamicsTracker(max_steps=50)
        
        for step in range(100):
            rewards = np.random.rand(10)
            probs = [np.random.dirichlet([1, 1, 1]) for _ in range(10)]
            actions = np.random.choice([0, 1, 2], 10)
            tracker.record(rewards, probs, actions)
        
        history = tracker.get_history()
        assert len(history['energy']) == 50

if __name__ == '__main__':
    pytest.main([__file__, '-v'])