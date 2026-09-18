"""Regression tests for optional-dependency handling and Streamlit scripts.

Covers:
- The RL package must import cleanly without gymnasium and raise a clear
  error only when TradingEnv is actually instantiated.
- TradingEnv must run end-to-end when gymnasium is available.
- dashboard/paper_monitor.py must be importable without side effects and
  render without exceptions inside a Streamlit runtime.
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import market_predictor_ml.dashboard.paper_monitor as paper_monitor
from market_predictor_ml.rl.environment import GYMNASIUM_AVAILABLE, TradingEnv

PAPER_MONITOR_PATH = Path(paper_monitor.__file__).resolve()


def _import_env_without_gymnasium():
    """Import rl.environment with gymnasium hidden; restore state after."""
    import builtins

    env_mod_name = "market_predictor_ml.rl.environment"
    saved_env = sys.modules.get(env_mod_name)
    saved_gym = sys.modules.get("gymnasium")
    real_import = builtins.__import__

    def blocker(name, *args, **kwargs):
        if name == "gymnasium" or name.startswith("gymnasium."):
            raise ImportError("gymnasium blocked for test")
        return real_import(name, *args, **kwargs)

    sys.modules.pop(env_mod_name, None)
    sys.modules.pop("gymnasium", None)
    builtins.__import__ = blocker
    try:
        mod = importlib.import_module(env_mod_name)
        yield mod
    finally:
        builtins.__import__ = real_import
        sys.modules.pop(env_mod_name, None)
        if saved_env is not None:
            sys.modules[env_mod_name] = saved_env
        else:
            importlib.import_module(env_mod_name)
        if saved_gym is not None:
            sys.modules["gymnasium"] = saved_gym


def test_rl_module_imports_without_gymnasium():
    """rl.environment must import when gymnasium is absent (no hard failure)."""
    for mod in _import_env_without_gymnasium():
        assert mod.GYMNASIUM_AVAILABLE is False
        assert mod.gym is None
        assert mod.spaces is None
        # The class falls back to `object` instead of failing at definition time
        assert mod.TradingEnv.__bases__ == (object,)

        # ...and raises a clear, actionable ImportError only on instantiation
        try:
            mod.TradingEnv(None)
            raised = False
        except ImportError as exc:
            raised = True
            assert "pip install gymnasium" in str(exc)
        assert raised, "TradingEnv() must raise ImportError without gymnasium"


def test_trading_env_runs_end_to_end():
    """TradingEnv reset/step loop works when gymnasium is available."""
    if not GYMNASIUM_AVAILABLE:
        print("gymnasium not installed; skipping")
        return

    rng = np.random.default_rng(0)
    n = 40
    df = pd.DataFrame(
        {
            "Close": 100 * np.cumprod(1 + rng.normal(0, 0.01, n)),
            "Volume": rng.integers(1_000_000, 5_000_000, n),
            "regime": rng.choice([-1.0, 0.0, 1.0], n),
        }
    )

    env = TradingEnv(df, initial_balance=100_000.0)
    obs, _ = env.reset(seed=42)
    # [balance_norm, position, features(3), regime]
    assert obs.shape == (2 + 3 + 1,)

    steps, done = 0, False
    while not done:
        obs, reward, done, truncated, info = env.step(env.action_space.sample())
        steps += 1

    assert steps == n - 1
    assert len(env.trade_history) == steps
    assert info["portfolio_value"] > 0



def test_hybrid_strategy_direction_and_risk_rules():
    """HybridRLStrategy applies direction, vol scaling, and regime penalty."""
    from market_predictor_ml.rl import HybridRLStrategy

    class StubAgent:
        def predict(self, observation, deterministic=True):
            return np.array([0.6]), None

    obs = np.zeros(4, dtype=np.float32)
    strategy = HybridRLStrategy(StubAgent(), max_position_pct=0.5)

    assert strategy.get_position_size(obs, 0, 0.02) == 0.0  # hold -> flat
    # Clipped to max_position_pct (raw 0.6 * vol_scale ~1.0 -> 0.5 cap)
    assert float(strategy.get_position_size(obs, 1, 0.02)) == pytest.approx(0.5)
    # Recession regime halves the exposure: ~0.6 * 0.5 = ~0.3
    assert float(strategy.get_position_size(obs, 1, 0.02, regime=2)) == pytest.approx(
        0.3, abs=1e-6
    )


def test_agents_raise_clear_error_without_stable_baselines3():
    """Factory functions report the missing optional package clearly."""
    from market_predictor_ml.rl import agents as rl_agents

    if rl_agents.SB3_AVAILABLE:
        print("stable-baselines3 installed; skipping")
        return

    env = TradingEnv(pd.DataFrame({"Close": [100.0, 101.0]}))
    for factory in (rl_agents.create_ppo_agent, rl_agents.create_dqn_agent):
        try:
            factory(env)
            raised = False
        except ImportError as exc:
            raised = True
            assert "stable-baselines3" in str(exc)
        assert raised


def test_paper_monitor_import_is_side_effect_free():
    """Importing the Streamlit script must not execute the UI or raise."""
    import market_predictor_ml.dashboard.paper_monitor as pm

    assert callable(pm.main)
    assert callable(pm.load_events)
    # Re-import is a no-op (no repeated side effects)
    assert importlib.import_module("market_predictor_ml.dashboard.paper_monitor") is pm


def test_paper_monitor_load_events_handles_missing_log():
    """load_events() returns an empty frame when the log file does not exist."""
    if paper_monitor.LOG_PATH.exists():
        print(f"log file present ({paper_monitor.LOG_PATH}); skipping")
        return
    df = paper_monitor.load_events()
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_paper_monitor_renders_in_streamlit_runtime():
    """The script must run inside a Streamlit runtime without exceptions."""
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(PAPER_MONITOR_PATH), default_timeout=60)
    at.run()

    assert len(at.exception) == 0, [e.value for e in at.exception]
    assert at.title and at.title[0].value.startswith("Market Predictor")
    # Config metrics always render; live metrics render once trades exist
    assert len(at.metric) >= 3
