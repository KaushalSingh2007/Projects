"""Deterministic preprocessing for Q-GUARD.

The pipeline is fit ONLY on the healthy training `Recording` and then applied to
both train and eval. Because `Pipeline.fit` receives a label-free `Recording`,
no preprocessing statistic (scaler mean/std, PCA basis) can be influenced by
evaluation data or labels. See `docs/preprocessing.md`.
"""

from .pipeline import (  # noqa: F401
    BandpassFilter,
    Detrend,
    PerChannelStandardizer,
    Pipeline,
    SpatialPCA,
    Taper,
    build_pipeline,
)
