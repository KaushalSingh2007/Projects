# Experiment Specifications and Results

## Experiment 1: Phase Transition Study

### Objective
Identify cooperation emergence as global incentive α increases.

### Setup
```python
from experiments.phase_transition_study import study_alpha_phase_transition

alphas, cooperations, stds = study_alpha_phase_transition(
    n_agents=50,
    n_steps=5000,
    n_runs=3
)
```

### Parameters
- n_agents: 50
- topology: small-world
- α: 0 to 3 (20 points)
- β: 1.0 (fixed)
- n_runs: 3

### Expected Results
1. α < 0.5: Defection dominates (C ≈ 0)
2. 0.5 < α < 1.5: Mixed phase, transition
3. α > 1.5: Cooperation dominates (C ≈ 1)

### Metrics
- Final cooperation C_final
- Cooperation variance
- Transition sharpness

---

## Experiment 2: Topology Comparison

### Objective
Compare convergence and equilibrium across topologies.

### Setup
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

### Parameters
- Topologies: random, small-world, scale-free
- n_agents: 50
- α=1.0, β=1.0 (fixed)
- Same number of edges (approximately)

### Expected Results

| Topology | Cooperation | Entropy | Convergence |
|----------|-------------|---------|-------------|
| Random   | ~0.5        | High    | Slow        |
| Small-W  | ~0.7        | Medium  | Fast        |
| Scale-F  | ~0.8        | Low     | Very fast   |

### Explanation
- Small-world balances local structure and global reach
- Scale-free hubs coordinate cooperation
- Random lacks structural advantage

---

## Experiment 3: Parameter Sweep (α, β)

### Objective
2D phase diagram showing cooperation as function of both parameters.

### Setup
```python
results = run_parameter_sweep(
    param_name='alpha',
    param_values=np.linspace(0, 2, 10),
    base_config={'n_agents': 50, 'n_steps': 5000, 'beta': 1.0},
    n_runs=3
)
```

### Grid
- α: 0 to 2 (10 points)
- β: 0 to 2 (10 points)
- Total: 100 configurations

### Expected Pattern
```
       β=0.0  β=0.5  β=1.0  β=1.5  β=2.0
α=0.0  0.0    0.1    0.2    0.3    0.4
α=0.5  0.1    0.3    0.5    0.6    0.7
α=1.0  0.2    0.5    0.7    0.8    0.9
α=1.5  0.3    0.6    0.8    0.9    0.95
α=2.0  0.4    0.7    0.9    0.95   1.0
```

---

## Experiment 4: Ablation Study

### Objective
Isolate importance of different components.

### Configurations Tested

1. **Local Only** (β=0)
   - Pure payoff-driven learning
   - Expected: Low cooperation

2. **Weak Global** (α=0.5, β=0.5)
   - Small global incentive
   - Expected: Mixed

3. **Medium Global** (α=1.0, β=1.0)
   - Balanced
   - Expected: Good cooperation

4. **Strong Global** (α=2.0, β=2.0)
   - High global pressure
   - Expected: Near-universal cooperation

### Expected Finding
Cooperation monotonically increases with α and β.

---

## Experiment 5: Learning Rate Sensitivity

### Objective
Find optimal learning rates for convergence.

### Parameters Tested
- Learning rates: 0.01, 0.02, 0.05, 0.1, 0.2

### Metrics
- Convergence speed (steps to 90% final cooperation)
- Final cooperation level
- Stability (oscillation magnitude)

### Expected Results
- Too slow (η=0.01): Slow convergence, high stability
- Optimal (η=0.05): Fast, stable
- Too fast (η=0.2): Oscillatory, unstable

---

## Running All Experiments

```bash
# Phase transitions
python experiments/phase_transition_study.py

# Topology comparison
python experiments/topology_comparison.py

# Ablation studies
python experiments/ablation_study.py
```

## Analysis Tools

```python
from src.utils import compute_statistics, equilibrium_analysis

# Summary stats
stats = compute_statistics(history)
print(f"Cooperation: {stats['cooperation']['final']:.3f}")

# Equilibrium properties (post-burnin)
eq = equilibrium_analysis(history, burnin=1000)
print(f"Eq. entropy: {eq['entropy']['eq_mean']:.3f}")
```

## Visualization

```python
from src.visualization import plot_metrics, plot_phase_diagram

# Time evolution
plot_metrics(history)

# Phase transition
plot_phase_diagram(alphas, cooperations)
```

## Expected Findings Summary

1. **Cooperation emergence** is a sharp phase transition
2. **Small-world networks** facilitate cooperation
3. **Global rewards** are critical for cooperation
4. **Entropy collapses** during learning
5. **Energy decreases** (dissipative system)
6. **Phase transitions** show critical phenomena

## Extensions

- Study entropy production rate
- Analyze mean-field equations
- Compare with theoretical predictions
- Study co-evolution dynamics
- Test heterogeneous agents