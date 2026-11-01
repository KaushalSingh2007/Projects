"""Feature extraction: PSD (quantum path) and AR (Sohn baseline), + feature PCA.

Each extractor maps a label-free `Recording` -> X of shape (n_windows, n_features).
Stateful reducers (FeaturePCA) are fit on healthy-train features only.
"""

from .psd import PSDFeatures, welch_psd  # noqa: F401
from .ar import ARFeatures, ar_yule_walker  # noqa: F401
from .pca import FeaturePCA  # noqa: F401
