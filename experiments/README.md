# experiments/ — research log

Outputs from the `EXECUTION_PLAN.md` validation phases. Nothing here is
production code; nothing here is imported by `market_predictor_ml`.

## Status

| Phase | Gate | Status | Date |
|---|---|---|---|
| 0 — environment | Gate 0 | **PASS** — 40 tests, smoke test completes | 2026-09-18 |
| 1 — real numbers | Gate 1 | **FAIL** — 0/5 tickers pass, both horizons | 2026-09-18 |
| 1c — signal test | — | **Pooled IC ≈ 0 (NO SIGNAL)**; one narrow h=1 signal in 2/5 tickers, smaller than costs | 2026-09-18 |
| 2 — paper trading | Gate 2-prep | **NOT AUTHORIZED** (Gate 1 failed) | — |
| 3 — stress tests | Gate 2 | **NOT RUN** (Gate 1 failed) | — |

**Read `phase1/GATE1_VERDICT.md` first.** It is the decision record; §9 is the
refined diagnosis.

## phase1/

| File | What it is |
|---|---|
| `GATE1_VERDICT.md` | The Gate 1 decision, the evidence, and the 7 findings (F1–F7). |
| `run_validation.py` | The plan's Step 1.1 script, run as written. Its numbers are **not economically interpretable** (see F1) and are kept only as evidence of the defect. |
| `run_horizon_matched.py` | The corrected measurement: P&L from the actual next-day return, at h=1 and h=21. This is the trustworthy artifact. |
| `summarize_gate1.py` | Prints the scorecards, cost attribution and the Sharpe-deflation diagnosis. |
| `check_determinism.py` | In-process repeatability (passes: bit-identical). |
| `check_thread_determinism.py` | Cross-process repeatability (fails: metrics vary run to run). |
| `check_data_stability.py` | Locates the cause — the feature/label hashes themselves differ between processes. |
| `run_signal_test.py` | **Phase 1c:** does the model know anything? IC / rank IC / hit rate / base-rate-robust skill / long-short spread, with F5 fixed and a random control. |
| `validation_*.json` | Raw metrics artifacts. |
| `signal_test_*.json` | Phase 1c artifact. |

## Headline numbers

Net-of-cost Sharpe, purged walk-forward, 2015–2024, 5 tickers:

- **h=1 (daily rebalance):** −2.23 (SPY) … +0.03 (MSFT). 0/5 pass.
- **h=21 (21-day hold):** −0.89 (SPY) … +0.56 (AAPL). 0/5 pass.
- Buy & hold on the identical fold dates: **+16.7% … +25.1% p.a.**

Cost drag is 2–12 bp/yr, so costs are not the binding constraint — the gross
signal is absent. See `GATE1_VERDICT.md` §4.

## Reproduce

```bash
cd /home/marnus/VS-Code/ML-3/ML-3
python experiments/phase1/run_horizon_matched.py
python experiments/phase1/summarize_gate1.py
python experiments/phase1/check_data_stability.py   # run twice; hashes differ
```
