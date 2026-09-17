"""Dashboard module for Market Predictor ML.

``app.py`` is the full multi-page Streamlit dashboard (``run_dashboard``);
``paper_monitor.py`` is the live paper-trader monitor reading
``paper_trades.jsonl`` (``streamlit run .../paper_monitor.py``).
Imported lazily so plain library import does not require streamlit.
"""

try:
    from market_predictor_ml.dashboard.app import run_dashboard  # noqa: F401

    __all__ = ["run_dashboard"]
except Exception:
    __all__ = []
