"""Q-GUARD geometric core: pure-state manifold tangent tools + robust estimator.

Ports the machinery of Lumbreras & Tomamichel (2026): tangent projector,
symmetric retraction (curvature-cancelling), and variance-adaptive
Median-of-Means estimation on CP^(d-1).
"""

from .tangent import (  # noqa: F401
    density,
    retract_vector,
    symmetric_pair,
    tangent_from_phi,
    tangent_projector,
)
from .mom_estimator import median_of_means, variance_adaptive_mean  # noqa: F401
