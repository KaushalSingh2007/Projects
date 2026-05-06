# API Reference

## Core Classes

### Agent

```python
class Agent:
    def __init__(agent_id, n_actions=3, init_std=0.1, rng=None)
    def policy() -> np.ndarray
    def sample_action() -> int
    def update_policy(action, advantage, lr)
    def update_value(reward, lr)
    def entropy() -> float
    def get_theta() -> np.ndarray
    def get_value() -> float
```

**Example:**
```python
from src.agents import Agent
agent = Agent(0)
action = agent.sample_action()
agent.update_policy(action, advantage=1.5, lr=0.05)
```

### MultiAgentEnvironment

```python
class MultiAgentEnvironment:
    def __init__(agents, G, payoff_matrix=None, alpha=1.0, beta=1.0, ...)
    def step() -> Dict
    def get_agent_policies() -> np.ndarray
    def compute_population_cooperation() -> float
    def reset()
```

**Example:**
```python
from src.environment import MultiAgentEnvironment
env = MultiAgentEnvironment(agents, G)
info = env.step()
policies = env.get_agent_policies()
```

## Graph Functions

### build_graph

```python
def build_graph(n_agents, topology='random', p=0.1, k=4, seed=None) -> nx.Graph
```

**Topologies:**
- `'random'`: Erdős-Rényi
- `'small-world'`: Watts-Strogatz
- `'scale-free'`: Barabási-Albert

**Example:**
```python
from src.graph_model import build_graph
G = build_graph(50, topology='small-world')
```

## Thermodynamic Functions

### compute_energy

```python
def compute_energy(rewards: np.ndarray) -> float
```

Returns negative sum of rewards.

### compute_entropy

```python
def compute_entropy(agent_probs: List[np.ndarray]) -> float
```

Returns Shannon entropy of population strategy distribution.

### compute_cooperation_fraction

```python
def compute_cooperation_fraction(actions: np.ndarray) -> float
```

Returns fraction of agents playing Cooperate.

### ThermodynamicsTracker

```python
class ThermodynamicsTracker:
    def record(rewards, agent_probs, actions)
    def get_history() -> Dict
```

## Training Functions

### run_simulation

```python
def run_simulation(
    n_agents=50,
    topology='random',
    n_steps=10000,
    payoff_matrix=None,
    alpha=1.0,
    beta=1.0,
    learning_rate=0.05,
    value_lr=0.1,
    seed=None,
    verbose=True
) -> Tuple[Dict, List[Agent], nx.Graph]
```

**Returns:**
- history: Dict with 'energy', 'entropy', 'cooperation'
- agents: Final agent states
- G: Interaction graph

**Example:**
```python
from src.training_loop import run_simulation
history, agents, G = run_simulation(
    n_agents=50,
    topology='small-world',
    n_steps=5000,
    alpha=1.5,
    beta=1.0
)
```

### run_experiment_sweep

```python
def run_experiment_sweep(
    param_name: str,
    param_values: List[float],
    base_config: Dict,
    n_seeds: int = 3
) -> Dict[float, List[Dict]]
```

**Example:**
```python
from src.training_loop import run_experiment_sweep
results = run_experiment_sweep(
    param_name='alpha',
    param_values=[0, 0.5, 1.0, 1.5, 2.0],
    base_config={'n_agents': 50, 'n_steps': 5000},
    n_seeds=3
)
```

## Visualization Functions

### plot_metrics

```python
def plot_metrics(history, figsize=(12,4), title="", save_path=None)
```

Plots energy, entropy, cooperation over time.

### plot_multiple_runs

```python
def plot_multiple_runs(histories, labels=None, figsize=(12,4))
```

Compare multiple runs side-by-side.

### plot_graph

```python
def plot_graph(G, agent_probs=None, color_by='cooperation')
```

Visualize network with agent coloring.

### plot_phase_diagram

```python
def plot_phase_diagram(param_values, final_cooperation, param_name)
```

Phase transition diagram.

## Configuration

### SimulationConfig

```python
@dataclass
class SimulationConfig:
    n_agents: int = 50
    topology: str = 'small-world'
    n_steps: int = 10000
    learning_rate: float = 0.05
    alpha: float = 1.0
    beta: float = 1.0
    ...
```

**Usage:**
```python
from src.config import SimulationConfig
config = SimulationConfig(n_agents=100, alpha=2.0)
```