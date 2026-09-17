"""Optional GNN feature augmentation (gated behind USE_GNN + torch).

Builds a tiny cross-asset correlation graph over the core engineered
features and appends graph-diffused features. When ``torch`` (or
``torch_geometric``) is unavailable - or ``USE_GNN=false`` - this is
a passthrough returning the input DataFrame unchanged.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    import torch  # noqa: F401

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    _TORCH_AVAILABLE = False

try:  # pragma: no cover - optional dependency
    import torch_geometric  # noqa: F401

    _PYG_AVAILABLE = True
except Exception:  # pragma: no cover
    _PYG_AVAILABLE = False


def is_gnn_available() -> bool:
    """True when torch is importable (PyG optional but preferred)."""
    return _TORCH_AVAILABLE


def gnn_enabled() -> bool:
    """True when USE_GNN=true AND torch is installed."""
    return os.getenv("USE_GNN", "false").lower() == "true" and _TORCH_AVAILABLE


class GNNFeatureAugmenter:
    """Append k-step graph-diffused features.

    Nodes = feature columns; edges = |correlation| > ``corr_threshold``.
    Uses pure numpy diffusion so it works with torch alone (no hard
    dependency on torch_geometric).
    """

    def __init__(self, corr_threshold: float = 0.3, n_steps: int = 1) -> None:
        self.corr_threshold = float(corr_threshold)
        self.n_steps = int(n_steps)

    def augment(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Return ``df`` plus ``GNN_Diffused_<feat>`` columns."""
        if not gnn_enabled():
            return df
        try:
            cols = [c for c in feature_cols if c in df.columns]
            if len(cols) < 2:
                return df
            window = df[cols].tail(252).fillna(0.0).values
            corr = np.corrcoef(window, rowvar=False)
            corr = np.nan_to_num(corr, nan=0.0)
            adj = (np.abs(corr) > self.corr_threshold).astype(float)
            np.fill_diagonal(adj, 0.0)
            deg = adj.sum(axis=1, keepdims=True)
            deg[deg == 0.0] = 1.0
            norm_adj = adj / deg
            X = df[cols].fillna(0.0).values.astype(float)
            H = X
            for _ in range(max(1, self.n_steps)):
                H = 0.5 * H + 0.5 * (H @ norm_adj.T)
            out = df.copy()
            for j, c in enumerate(cols):
                out[f"GNN_Diffused_{c}"] = H[:, j]
            print(f"[GNN] Added {len(cols)} diffused features "
                  f"(torch={_TORCH_AVAILABLE}, pyg={_PYG_AVAILABLE})")
            return out
        except Exception as exc:
            print(f"[GNN] Augmentation skipped: {exc}")
            return df
