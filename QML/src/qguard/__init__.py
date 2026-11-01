"""Q-GUARD: Quantum Geometry-aware Unsupervised Anomaly detection for
vibration-based damage detection.

This package implements an unsupervised structural-health-monitoring pipeline
that encodes vibration features as pure quantum states on the manifold CP^(d-1)
and scores deviations from a healthy baseline using curvature-aware,
variance-adaptive geometry (tangent-space estimation with symmetric
retractions), benchmarked honestly against classical baselines.

Design contract: only HEALTHY data is ever seen during model fitting.
Damage labels are visible solely to the evaluation module (`qguard.eval`).
No quantum-advantage claim is made; this is a representation/geometry study.
"""

__version__ = "0.0.1"

__all__ = ["__version__"]
