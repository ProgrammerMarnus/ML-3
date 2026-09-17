"""Optional RL policy layer (gated behind USE_RL + stable-baselines3).

The RL agent does NOT replace the supervised LightGBM forecast.
It learns a small residual position adjustment on top of the
rule-based sizing from ``market_predictor_ml.decision``.

When ``stable-baselines3``/``gymnasium`` are unavailable (or
``USE_RL=false``) the policy is a no-op passthrough returning the
base position unchanged.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

try:  # pragma: no cover - optional dependency
    import gymnasium as gym  # noqa: F401
    from stable_baselines3 import PPO  # noqa: F401

    _RL_AVAILABLE = True
except Exception:  # pragma: no cover
    _RL_AVAILABLE = False


def is_rl_available() -> bool:
    """True when stable-baselines3 + gymnasium are importable."""
    return _RL_AVAILABLE


def rl_enabled() -> bool:
    """True when USE_RL=true AND the RL libs are installed."""
    return os.getenv("USE_RL", "false").lower() == "true" and _RL_AVAILABLE


class RLPolicy:
    """Residual position-size policy.

    ``adjust`` maps (base_position, prediction) -> final position in
    ``[-max_position, max_position]``. Until a PPO model is trained and
    loaded, this applies a conservative 0.5x scale so paper trading
    stays safe by default.
    """

    def __init__(
        self,
        max_position: float = 1.0,
        model_path: Optional[str] = None,
    ) -> None:
        self.max_position = float(max_position)
        self.model = None
        if rl_enabled() and model_path:
            try:  # pragma: no cover
                from stable_baselines3 import PPO

                self.model = PPO.load(model_path)
            except Exception as exc:
                print(f"[RL] Could not load model {model_path}: {exc}")

    def adjust(self, base_position: float, prediction: float) -> float:
        """Adjust a base position with the RL residual."""
        base = float(np.clip(base_position, -self.max_position, self.max_position))
        if self.model is None:
            # Safe default: damp the supervised signal until RL is trained.
            return float(np.clip(0.5 * base, -self.max_position, self.max_position))
        try:  # pragma: no cover
            obs = np.array([base, float(prediction)], dtype=np.float32)
            action, _ = self.model.predict(obs, deterministic=True)
            residual = float(np.clip(action[0], -0.5, 0.5))
            return float(np.clip(base + residual, -self.max_position, self.max_position))
        except Exception:
            return base
