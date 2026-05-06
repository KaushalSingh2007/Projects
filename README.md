# Thermodynamic Multi-Agent Reinforcement Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A research-grade implementation combining **evolutionary multi-agent reinforcement learning** with **thermodynamic principles**.

## 🎯 Overview

This project studies how agents in a network learn cooperative strategies through reinforcement learning while their collective behavior is governed by thermodynamic constraints.

**Key Insight:** Both learning systems and physical systems are dynamical systems of interacting entities governed by energy and entropy.

### Features

✅ **Discrete Strategy Space**: Cooperate, Defect, Neutral with stochastic policies  
✅ **Policy Gradient Learning**: Actor-critic with value baselines  
✅ **Graph-based Interactions**: Multiple topologies (random, small-world, scale-free)  
✅ **Real-time Thermodynamics**: Energy and entropy tracking  
✅ **Phase Transition Analysis**: Detect cooperation emergence  
✅ **Comprehensive Visualization**: Plots, graphs, heatmaps  
✅ **Experiment Framework**: Parameter sweeps, ablations, comparisons  
✅ **Unit Tests**: Full test coverage  

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/KaushalSingh2007/thermodynamic-marl.git
cd thermodynamic-marl
pip install -r requirements.txt
```

### Basic Usage

```python
from src.training_loop import run_simulation
from src.visualization import plot_metrics

# Run simulation
history, agents, G = run_simulation(
    n_agents=50,
    topology='small-world',
    n_steps=10000,
    alpha=1.0,      # global cooperation incentive
    beta=1.0        # weight of global signal
)

# Plot results
plot_metrics(history)
```

## 📊 System Architecture

### Agent State
Each agent $i$ maintains:
- **Policy parameters** $\theta_i \in \mathbb{R}^3$: Softmax logits for 3 strategies
- **Value baseline** $v_i$: For advantage estimation
- **Strategy** $s_i \in \{0,1,2\}$: Cooperate, Defect, Neutral

### Learning Rule
**Policy Gradient with Advantage:**
```
θ_k ← θ_k + η·A·[I(a=k) - π(k)]
```
where:
- $A = R - v$ (advantage)
- $R = R_{local} + β·R_{global}$ (mixed reward signal)
- $η$ is learning rate

### Thermodynamic Quantities

**Energy:**
$$\mathcal{H} = -\sum_i R_i$$

**Entropy:**
$$S = -\sum_a p_a \log p_a$$

where $p_a$ is the population fraction with action $a$.

**Free Energy (optional):**
$$F = E - T·S$$

## 🔬 Experiments

### 1. Phase Transition Study

```python
from experiments.phase_transition_study import study_alpha_phase_transition

alphas, cooperations, stds = study_alpha_phase_transition(
    n_agents=50,
    n_steps=5000,
    n_runs=3
)
```

Observe cooperation emergence as global incentive $\alpha$ increases.

### 2. Topology Comparison

```python
from experiments.topology_comparison import compare_topologies

histories = compare_topologies(
    n_agents=50,
    n_steps=5000,
    n_runs=3,
    alpha=1.0,
    beta=1.0
)
```

Compare random, small-world, and scale-free networks.

### 3. Parameter Sweep

```python
from experiments.experiment_runner import run_parameter_sweep

results = run_parameter_sweep(
    param_name='beta',
    param_values=[0, 0.5, 1.0, 1.5, 2.0],
    n_agents=50,
    n_steps=5000,
    n_runs=3
)
```

## 📈 Key Results

The system exhibits:

1. **Cooperation Phases**
   - Low $\alpha, \beta$: Defection dominates
   - High $\alpha, \beta$: Cooperation emerges

2. **Entropy Dynamics**
   - Initial exploration → high entropy
   - Learning → entropy collapse (consensus) or plateau (mixed)
   - Possible oscillations at phase transitions

3. **Energy Evolution**
   - Decreases over time (like dissipative systems)
   - Analogous to Lyapunov function behavior

4. **Topology Effects**
   - Scale-free hubs may stabilize cooperation
   - Small-world networks show fastest convergence
   - Random networks serve as baseline

## 📚 Documentation

- [System Design](docs/DESIGN.md) - Detailed architecture and theory
- [API Reference](docs/API.md) - Module documentation
- [Experiments](docs/EXPERIMENTS.md) - Experiment details and results

## 🧪 Testing

```bash
pytest tests/ -v
pytest tests/ --cov=src     # with coverage
```

## 📝 Citation

If you use this code in research, please cite:

```bibtex
@software{singh2024thermodynamic,
  title={Thermodynamic Multi-Agent Reinforcement Learning},
  author={Singh, Kaushal},
  year={2024},
  url={https://github.com/KaushalSingh2007/thermodynamic-marl}
}
```

## 🔗 Research Context

This work connects:
- **Evolutionary Game Theory**: Strategic adaptation in populations
- **Thermodynamic Dynamical Systems**: Energy/entropy as order parameters
- **Physics-Informed ML**: Physical constraints in learning systems

## 🚧 Future Extensions

- [ ] Dynamic graphs (co-evolving networks)
- [ ] Heterogeneous agents (different learning rates)
- [ ] Non-Markovian memory models
- [ ] Mean-field theoretical analysis
- [ ] GPU acceleration
- [ ] Distributed training

## 📄 License

MIT License - see LICENSE file for details.

## 👤 Author

**Kaushal Singh** - [@KaushalSingh2007](https://github.com/KaushalSingh2007)

---

**Last Updated**: March 2026
