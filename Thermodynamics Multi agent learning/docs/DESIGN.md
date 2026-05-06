# System Design Document

## Overview

The Thermodynamic Multi-Agent Reinforcement Learning (MARL) system combines evolutionary game theory with thermodynamic principles to study how agents learn cooperative strategies in networks.

## Theoretical Foundation

### Game Theory Framework

Agents play repeated pairwise games with neighbors. Each game is defined by a 3×3 payoff matrix:

```
       C    D    N
   C [[3,   0,   1],
   D  [5,   1,   2],
   N  [1,   2,   2]]
```

Where:
- **C (Cooperate)**: Payoff 3 with C, 0 with D, 1 with N
- **D (Defect)**: Payoff 5 with C, 1 with D, 2 with N
- **N (Neutral)**: Payoff 1 with C, 2 with D, 2 with N

### Reinforcement Learning

Agents learn policies π_i(a|s) = softmax(θ_i,a) over strategies using policy gradient:

**Update Rule:**
```
θ_i,k ← θ_i,k + η·(R_i - v_i)·[I(a=k) - π_i(k)]
```

Where:
- η: learning rate
- R_i: reward (local + global)
- v_i: value baseline
- a: sampled action

### Thermodynamic Interpretation

**Energy:**
```
E = -Σ_i R_i
```

Low energy = favorable state (high rewards)

**Entropy:**
```
S = -Σ_a p_a·log(p_a)
```

Where p_a = fraction of population with action a

**Free Energy (optional):**
```
F = E - T·S
```

## Architecture

### Agents

Each agent maintains:
- Policy logits θ_i (3 dimensions)
- Value baseline v_i (scalar)
- Action history

**Agent behavior:**
1. Compute policy: π_i = softmax(θ_i)
2. Sample action: a_i ~ π_i
3. Receive reward: R_i = R_local + β·R_global
4. Update: θ_i ← θ_i + η·A_i·∇log π_i(a_i)
5. Update: v_i ← v_i + η_v·(R_i - v_i)

### Environment

Orchestrates agent interactions:
1. Sample random edge (i,j)
2. Agents play game
3. Compute rewards (local + global cooperation bonus)
4. Update both agents
5. Track metrics

### Graph Models

**Random (Erdős-Rényi):**
- Each edge with probability p
- Homogeneous, baseline

**Small-World (Watts-Strogatz):**
- Ring lattice with k neighbors
- Random rewiring probability p
- High clustering, short paths

**Scale-Free (Barabási-Albert):**
- Preferential attachment
- Power-law degree distribution
- Hubs emerge

## Reward Signal

**Local Payoff:**
```
R_i^local = P[a_i, a_j]
```

**Global Signal:**
```
R_i^global = α · C_pop
```

Where C_pop = fraction with high cooperation probability

**Total Reward:**
```
R_i = R_i^local + β·R_i^global
```

**Parameters:**
- α: strength of global cooperation incentive
- β: weight of global signal

## Dynamics

### Short-term (Single Step)
1. Edge (i,j) sampled
2. Actions determined by policies
3. Policies updated
4. Values updated

### Medium-term (Hundreds of Steps)
1. Cooperation level rises/falls
2. Entropy decreases (policies converge)
3. Energy stabilizes

### Long-term (Thousands of Steps)
1. System reaches equilibrium
2. Final cooperation level determined
3. Entropy plateau or oscillation

## Phase Transitions

As α or β increase, system transitions through phases:

1. **Defection Phase** (low α, β)
   - High defection
   - High entropy
   - Low energy (surprisingly high payoffs)

2. **Mixed Phase** (medium α, β)
   - Some cooperation
   - Moderate entropy
   - Unstable dynamics

3. **Cooperation Phase** (high α, β)
   - High cooperation
   - Low entropy (consensus)
   - Low energy (stable)

## Implementation Details

### Learning Rates
- Policy: η_learn ≈ 0.05
- Value: η_value ≈ 0.1

### Initialization
- Logits: θ_i ~ N(0, 0.1)
- Values: v_i = 0

### Numerical Stability
- Entropy clipping: p_a ∈ [1e-12, 1.0]
- Advantage scaling: implicit via learning rates

## Experimental Design

### Key Experiments

1. **Phase Transition Study**
   - Vary α from 0 to 3
   - Measure final cooperation
   - Look for sudden changes

2. **Topology Comparison**
   - Compare 3 topologies
   - Same α, β, n_agents
   - Measure convergence speed

3. **Parameter Sweep**
   - Vary learning rate, α, β
   - 2D heatmaps
   - Find optimal regions

4. **Ablation Study**
   - Remove global reward (β=0)
   - Vary learning rate
   - Isolate component effects

## Metrics

**Primary:**
- Cooperation fraction C(t)
- Entropy S(t)
- Energy E(t)

**Secondary:**
- Convergence speed
- Equilibrium variance
- Phase transition sharpness
- Network statistics (clustering, diameter)

## Extensions

Possible future work:
- Dynamic graphs (co-evolution)
- Heterogeneous agents
- Memory models
- Mean-field theory
- GPU acceleration