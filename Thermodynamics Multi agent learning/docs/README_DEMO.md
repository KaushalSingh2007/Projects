# Phase Transition in Multi-Agent Learning via Thermodynamic Incentives

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A research-grade demonstration of **engineered phase transitions** in multi-agent reinforcement learning systems through **thermodynamic incentive design**.

## 🎯 Problem Statement

Multi-agent systems often fail to achieve cooperation despite individual learning. Traditional reward structures incentivize individual optimization, leading to collective behavior trapped in defective equilibria.

**Key Question**: Can we design incentives that create emergent cooperation through phase transitions?

## 💡 Key Innovation

### Original Reward (Failed Approach)
```
R_i = P[a_i, a_j] + β * α * C_pop
```
- All agents receive global incentive regardless of action
- Defectors get rewarded for others' cooperation
- System remains in defective phase

### Modified Reward (Breakthrough)
```
R_i = P[a_i, a_j] + β * α * C_pop * I(a_i == C)
```
- **Only cooperating agents receive global incentive**
- Defectors get no global reward
- Incentives properly aligned with collective behavior

## 🚀 Main Result - Phase Transition Achieved

![Phase Transition](phase_transition_final.png)

**Key Findings:**
- **Critical Point**: α_c ≈ 2.8
- **Cooperation Jump**: 26% → 98% at critical point
- **Clear Phase Separation**: Defection → Mixed → Cooperation

| Phase | α Range | Cooperation | Behavior |
|--------|----------|-------------|-----------|
| Defection | α < 2.8 | < 30% | Individual optimization dominates |
| Mixed | α ≈ 2.8 | 30-70% | Competing forces balance |
| Cooperation | α > 2.8 | > 70% | Collective behavior emerges |

## 🛠️ How to Run

### Installation
```bash
git clone https://github.com/KaushalSingh2007/thermodynamic-marl.git
cd thermodynamic-marl
pip install -r requirements.txt
```

### Quick Demo
```python
# Run phase transition study
python -m experiments.phase_transition_study

# Run all experiments (no popup windows)
python run_all_experiments.py

# Basic simulation
python example_run.py
```

### Key Scripts
- `run_all_experiments.py` - Complete experiment suite
- `experiments/phase_transition_study.py` - Phase transition analysis
- `example_run.py` - Basic simulation demonstration

## 📊 Example Outputs

### Phase Transition Diagram
![Phase Transition](phase_transition_final.png)
Shows cooperation vs global incentive strength with clear critical point

### Network Comparison
![Topology Comparison](topology_comparison_final.png)
Random networks achieve highest cooperation (20%)

### Thermodynamic Evolution
![System Evolution](output_metrics.png)
Energy decreases while entropy collapses as system orders

## 🔬 Key Insights

### 1. **Incentive Alignment is Crucial**
- Proper reward alignment creates dramatic behavioral changes
- Small modification (conditional reward) leads to phase transition

### 2. **Phase Transition Mechanism**
- **Low α**: Local payoffs dominate → Defection phase
- **High α**: Global incentives dominate → Cooperation phase
- **Critical α**: Competing forces balance → Sharp transition

### 3. **Thermodynamic Signatures**
- **Energy**: Lower in cooperative phase (more favorable)
- **Entropy**: Decreases as system orders (strategy convergence)
- **Cooperation**: Serves as order parameter

### 4. **Network Effects**
- **Random networks**: Most conducive to cooperation
- **Small-world**: Surprisingly low cooperation
- **Scale-free**: Intermediate performance

## 📈 Performance Metrics

### Cooperation Levels Achieved
- **Maximum**: 98% cooperation (α = 3.0)
- **Baseline**: 10-20% cooperation (α < 2.8)
- **Improvement**: 5-10x increase in cooperation

### Phase Transition Characteristics
- **Sharpness**: Discontinuous jump at critical point
- **Reproducibility**: Consistent across multiple runs
- **Tunability**: Critical point adjustable via parameters

## 🎯 Applications

This research enables:
- **Collective Intelligence Engineering**: Design systems that self-organize
- **Social Dynamics Modeling**: Understand incentive-driven behavior
- **Multi-Agent System Design**: Build cooperative AI systems
- **Critical Phenomena Study**: Explore phase transitions in learning

## 📚 Technical Details

### System Architecture
- **Agents**: Policy gradient learning with value baselines
- **Network**: Graph-based interactions (random, small-world, scale-free)
- **Game Theory**: 3-strategy payoff matrix (Cooperate, Defect, Neutral)
- **Thermodynamics**: Real-time energy and entropy tracking

### Learning Algorithm
- **Policy Updates**: REINFORCE-style gradient ascent
- **Value Functions**: Temporal difference learning
- **Exploration**: Stochastic action sampling
- **Convergence**: Monitored via thermodynamic metrics

## 🔮 Future Directions

- **Critical Phenomena Analysis**: Study universality classes
- **Network Dynamics**: Time-varying topologies
- **Multi-Objective**: Balance cooperation with efficiency
- **Real-World Validation**: Apply to practical systems

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

---

**Core Message**: *Align individual incentives with collective goals to engineer emergent cooperation through phase transitions.*
