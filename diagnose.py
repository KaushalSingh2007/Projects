import sys
print(f"Python version: {sys.version}")
print(f"Python path: {sys.executable}")

try:
    import numpy
    print(f"✓ NumPy {numpy.__version__}")
except:
    print("✗ NumPy failed")

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
except:
    print("✗ PyTorch failed")

try:
    import networkx
    print(f"✓ NetworkX {networkx.__version__}")
except:
    print("✗ NetworkX failed")

try:
    import matplotlib
    print(f"✓ Matplotlib {matplotlib.__version__}")
except:
    print("✗ Matplotlib failed")

try:
    from src.agents import Agent
    print("✓ Can import Agent")
except Exception as e:
    print(f"✗ Cannot import Agent: {e}")

try:
    from src.training_loop import run_simulation
    print("✓ Can import run_simulation")
except Exception as e:
    print(f"✗ Cannot import run_simulation: {e}")
