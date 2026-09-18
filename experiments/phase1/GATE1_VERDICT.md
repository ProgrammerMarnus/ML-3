# GATE 1 VERDICT — Market Predictor ML

**Date:** 2026-09-18
**Plan:** `EXECUTION_PLAN.md` Phase 1 / Gate 1
**Codebase:** market_predictor_ml v0.4.0 @ `39b932b`

---

## THE DECISION

> ### GATE 1: **FAIL**
>
> **0 of 5 tickers pass** at the 1-day horizon and **0 of 5 pass** at the
> 21-day horizon, out-of-sample, net of costs.
>
> Per the plan's own rule — *"FAIL (0 tickers pass) → STOP. Go to Section 8
> (what NOT to do) and Section 9 (redesign options). Do NOT proceed to paper
> trading."* — **Phase 2 is not authorized. Paper trading is not authorized.**
>
> This is a contract with myself. No engineering work proceeds on the
> assumption of an edge that this measurement does not show.

**One qualification, stated up front:** the first Phase 1 run produced numbers
that are *not economically interpretable* (see §3). I re-measured correctly
before accepting the verdict. The corrected measurement is *worse*, not
better, so the verdict stands on both the original and the corrected numbers.

---

## 1. WHAT WAS MEASURED

| Item | Value |
|---|---|
| Universe | SPY, QQQ, AAPL, MSFT, GOOGL |
| Window | 2015-01-01 → 2024-12-31 (2 515 bars/ticker) |
| Validation | purged/embargoed walk-forward, `n_splits=5`, `test_size=252`, `purge=5`, `embargo=5` |
| Model | LightGBM (500 trees, lr 0.05, depth 6, 31 leaves, subsample/colsample 0.8, L1/L2 0.1) |
| Sizing | `volatility_adjusted` (target_vol 0.02, clip ±1.0) |
| Costs | 0.10% commission + 0.05% slippage = **0.15% per unit of turnover** |
| Label (h=1) | `Target_RiskAdj_1d` |
| Label (h=21) | `Target_RiskAdj_21d` |
| P&L basis | `position[t] * (close[t+1]/close[t] − 1) − abs(dposition) * 0.0015` |

No look-ahead in the splitter (train strictly precedes test, purge and embargo
non-zero). LightGBM early stopping is **not** applied inside the folds —
`run_walk_forward_backtest` calls `model.fit(X_train, y_train)` with no
`eval_set` — so no fold sees its own future.

---

## 2. CORRECTED RESULTS (horizon-matched, net of costs)

`experiments/phase1/validation_horizon_matched_2026-09-18.json`

### h = 1 — daily rebalance

| Ticker | Sharpe | PF (reported) | PF (true) | Win rate | MaxDD | Ann return | Turnover | Buy&Hold | Verdict |
|--------|-------:|--------------:|----------:|---------:|------:|-----------:|---------:|---------:|---------|
| SPY   | **−2.23** | 1.09 | 0.60 | 35.7% | 1.07% | −0.21% | 0.0031 | 17.17% | FAIL |
| QQQ   | **−0.63** | 1.37 | 0.87 | 38.8% | 0.30% | −0.06% | 0.0025 | 17.17% | FAIL |
| AAPL  | **−0.74** | 1.26 | 0.85 | 40.4% | 0.36% | −0.05% | 0.0020 | 25.04% | FAIL |
| MSFT  | **+0.03** | 1.39 | 1.01 | 41.9% | 0.23% | +0.00% | 0.0021 | 22.62% | FAIL |
| GOOGL | **−0.88** | 1.14 | 0.82 | 41.9% | 0.41% | −0.07% | 0.0018 | 21.53% | FAIL |

### h = 21 — signal refreshed every 21 days, position held in between

| Ticker | Sharpe | PF (reported) | PF (true) | Win rate | MaxDD | Ann return | Turnover | Buy&Hold | Verdict |
|--------|-------:|--------------:|----------:|---------:|------:|-----------:|---------:|---------:|---------|
| SPY   | **−0.89** | 0.89 | 0.80 | 47.5% | 4.99% | −0.97% | 0.0014 | 16.73% | FAIL |
| QQQ   | **−0.69** | 1.00 | 0.84 | 45.8% | 3.70% | −0.70% | 0.0011 | 18.32% | FAIL |
| AAPL  | **+0.56** | 1.05 | 1.13 | 51.8% | 0.86% | +0.23% | 0.0007 | 24.95% | FAIL |
| MSFT  | **−0.08** | 1.04 | 0.99 | 48.6% | 0.88% | −0.03% | 0.0006 | 25.11% | FAIL |
| GOOGL | **−0.75** | 0.94 | 0.86 | 47.5% | 3.89% | −0.54% | 0.0006 | 23.27% | FAIL |

Gate 1 criteria: `sharpe > 0.5 AND profit_factor > 1.2 AND max_drawdown > −25%`.
Only AAPL @ h=21 clears the Sharpe bar (0.56) and it misses on PF (1.13 < 1.2).
AAPL @ h=21 is **1 of 5** — the plan classifies that as MARGINAL at best, and
it is a single-ticker story, not a system.

**Buy & hold on the identical fold dates returned 16.7%–25.1% p.a.** The
strategy returned ≈0%. No benchmark-alpha analysis is needed to see there is
no alpha here.

## 3. WHY THE FIRST PHASE 1 NUMBERS WERE NOT TRUSTWORTHY

`experiments/phase1/run_validation.py` (the literal Step 1.1 script) produced
Sharpe −3.45 … +1.63 and annual returns of −111% for SPY. Those numbers came
from this pairing:

```python
pipeline.prepare_data(target_column='Target_RiskAdj_21d')
# backtest/engine.py:396
strategy_returns[1:] = positions[:-1] * y_test[1:]
```

`y` is the **21-day forward** risk-adjusted return
(`close[t+21]/close[t] − 1`, divided by annualised 21-day volatility —
`features/labels.py:235-244`). The engine multiplies the position by that
21-day quantity and treats it as a **daily** return, then compounds it.

Measured consequences:

| Ticker | Phase 1 Sharpe | lag-1 autocorr | Inflation factor | Deflated Sharpe |
|--------|---------------:|---------------:|-----------------:|----------------:|
| SPY   | −3.45 | 0.8941 | 3.87 | −0.89 |
| QQQ   | −2.17 | 0.9115 | 3.86 | −0.56 |
| AAPL  | +1.63 | 0.9078 | 3.37 | **+0.48** |
| MSFT  | +0.41 | 0.8100 | 3.34 | +0.12 |
| GOOGL | −2.67 | 0.9149 | 3.80 | −0.70 |

The deflation factor is the Newey-West/Hansen-Hodrick correction
`sqrt(1 + 2 * sum((1 - k/(K+1)) * rho_k))` with K=40. Overlapping 21-day
windows inflate the annualised Sharpe by ~3.3–3.9×. **AAPL's "PASS" (Sharpe
1.63, PF 1.20) deflates to Sharpe 0.48 — below the 0.5 Gate.** The one
apparent pass in Phase 1 was an artifact of overlapping labels.

---

## 4. COST ATTRIBUTION — the failure is NOT a cost problem

| Horizon | Gross Sharpe range | Net Sharpe range | Cost drag (annualised) |
|---|---|---|---|
| h=1 | −0.98 … +0.91 | −2.23 … +0.03 | 0.07% – 0.12% |
| h=21 | −0.85 … +0.62 | −0.89 … +0.56 | 0.02% – 0.04% |

At h=21 costs consume **2–4 basis points per year**. Four of five tickers are
still negative. The gross signal is absent; costs are not the binding
constraint. This is decision-relevant: it **rules out** the cost-motivated
redesign levers (raise threshold / trade less / lengthen horizon) as fixes.

At h=1, MSFT is the one informative case: gross Sharpe **+0.91** → net
**+0.03**. There, costs do erase the edge. So cost work is only worth doing
*if and when* a gross signal is first found to exist somewhere.

## 5. FINDINGS (defects uncovered by this validation)

| # | Finding | Evidence |
|---|---|---|
| F1 | **Label/horizon–P&L mismatch.** The engine credits a 21-day forward quantity as a daily return. Sharpe inflated ~3.4–3.9×. | `backtest/engine.py:396` + `features/labels.py:235-244` |
| F2 | **`profit_factor` is a payoff ratio, not a profit factor.** `profit_factor = -avg_win / avg_loss` ignores trade frequencies, so it can exceed 1.2 while the strategy loses money. True PF = `(win_rate*avg_win) / ((1-win_rate)*abs(avg_loss))`. SPY h=1: reported 1.09 → true **0.60**. | `backtest/engine.py:265` |
| F3 | **Gate 1's `PF > 1.2` criterion cannot be evaluated as written** until F2 is fixed — and with the current definition it gives false positives. | F2 |
| F4 | **Position sizing is off by a large factor.** `predictions` are dimensionless (return ÷ annualised vol) but `volatility_adjusted_position` divides by a *daily* vol and multiplies by `target_vol=0.02`. Realised book vol is ~0.5–1.5% p.a. instead of the intended risk target; positions are ~0.002–0.07, so the ±1.0 clip never binds. | `decision/sizing.py:88` + `pipeline.py:192-194` |
| F5 | **Fold early stopping never runs in the walk-forward.** `train_model()` (standalone) early-stopped at iteration **1** of 500 — validation RMSE never improved — while the fold loop trains all 500 trees with no `eval_set`. Two different models are used for the same claim. | `pipeline.py:293` vs `backtest/engine.py:378` |
| F6 | **The research result is not reproducible: market data is never pinned.** `download_stock_data` calls `yf.Ticker(ticker).history(...)` live on every invocation — no disk cache, no snapshot, no hash. Yahoo's adjusted prices are not bit-stable between fetches, so each process trains on slightly different data. | `data/loader.py:34-35` |
| F7 | **The resulting metric uncertainty exceeds the Gate margin.** Two identical `OMP_NUM_THREADS=1` processes produced AAPL h=1 net Sharpe **−0.4808** and **−0.6250** (range **0.14** = 28% of the 0.5 threshold); X/y hashes and even `y[0]` differ from the 7th significant digit between processes. In-process repeats are bit-identical, so the variance is data-driven, not model-driven. | `check_thread_determinism.py`, `check_data_stability.py` |

**Quantified uncertainty.** Observed AAPL metrics across five separate processes
on unchanged code:

| Run | AAPL h=1 Sharpe | AAPL h=1 PF | AAPL h=21 Sharpe |
|---|---|---|---|
| main horizon-matched run | −0.74 | 1.26 | +0.56 |
| determinism (in-process ×3) | −0.4876 | 1.3442 | — |
| threads=1 (run A) | −0.4808 | 1.2882 | +0.5611 |
| threads=1 (run B) | −0.6250 | 1.2622 | +0.5656 |
| threads=4 | −0.5175 | 1.2910 | +0.5649 |

h=1 swings by 0.14 Sharpe and 0.08 PF; h=21 is comparatively stable (0.005).
Because h=1's label is the noisiest, small data perturbations flip near-tie
tree splits. **Any h=1 metric quoted to two decimals in this project is
currently noise at that precision.** Gate thresholds cannot be evaluated
against a measurement with this much run-to-run variance.

Two consequences:
1. Before any further Gate decision, **freeze the data**: download once to a
   Parquet/CSV snapshot keyed by date range, hash it, and have the pipeline
   load that snapshot. Re-fetch never happens inside a measurement.
2. Report Gate metrics as a **distribution over repeated runs**, not a single
   point estimate.

---

## 6. WHAT THIS DOES AND DOES NOT AUTHORIZE

Per plan Section 8 — explicitly forbidden until gates pass:

- **NOT authorized:** wiring `USE_RL` / `USE_GNN`
- **NOT authorized:** running `hyperopt.py` (would tune noise)
- **NOT authorized:** adding data providers (sentiment, macro)
- **NOT authorized:** shortening the horizon to intraday
- **NOT authorized:** relaxing risk limits
- **NOT authorized:** scaling position size to compensate
- **NOT authorized:** Phase 2 (paper trading) or any `model_registry` training run for live use

**Authorized now:** fixing F1–F6, then a *re-measurement* at Gate 1 — not new
features.

---

## 7. NEXT ACTIONS (cheapest first)

0. **Freeze the data first (F6/F7).** One download to a hashed snapshot
   (`data/snapshots/<TICKER>_<start>_<end>.parquet` + `.sha256`), loaded by
   `download_stock_data` whenever a snapshot for that key exists. Nothing
   measured before this is reproducible, and the observed 0.14-Sharpe swing at
   h=1 is the same order as the Gate margin. Verify by re-running
   `check_data_stability.py` twice and confirming identical hashes.
1. **Fix F2** (true profit factor) — small, in `backtest/engine.py`. Gate 1
   cannot be decided without it. Add a regression test asserting
   `PF == gross_profit / gross_loss` against a hand-computed fixture.
2. **Fix F1** — either make the engine P&L from realised next-bar returns
   (as `run_horizon_matched.py` does) or make the label match the holding
   period. Add a guard that refuses to backtest when the label horizon does
   not match the holding period.
3. **Fix F4** — decide the units contract for predictions, then rescale
   `target_vol`. The current book runs at ~1% annualised vol, so every metric
   above is computed on a nearly-uninvested portfolio.
4. **Fix F5** — make `train_model()` and the fold loop train the same model;
   use an in-fold holdout for early stopping (never the test fold).
5. **Re-run Gate 1** with F1/F2/F4 fixed.
6. **Only if a gross signal appears**, apply plan Section 9 in order:
   R1 market-neutralise → R2 higher threshold → R3 different label
   (triple-barrier) → R4 different universe → R5 different horizon →
   R6 decorrelated ensemble. **R7 (accept the null) remains a legitimate and
   likely outcome** — the README says as much.

---

## 8. REPRODUCTION

```bash
cd /home/marnus/VS-Code/ML-3/ML-3

# Phase 0
python -m pytest market_predictor_ml/tests -q          # 40 passed
python market_predictor_ml/examples/run_pipeline.py    # smoke test

# Phase 1  (Step 1.1 as written - numbers are NOT interpretable, see section 3)
python experiments/phase1/run_validation.py

# Phase 1b (horizon-matched, economically valid)
python experiments/phase1/run_horizon_matched.py
python experiments/phase1/summarize_gate1.py

# Determinism
python experiments/phase1/check_determinism.py           # in-process, bit-identical
python experiments/phase1/check_thread_determinism.py    # cross-process, varies
python experiments/phase1/check_data_stability.py        # shows the X/y hashes differ
```

**Verdict recorded: FAIL. Do not build on an unproven edge.**

**Refinement (Phase 1c, same date):** the failure is *not* "no signal
anywhere". It is "one narrow 1-day signal in 2 of 5 tickers that is real but
smaller than the cost of harvesting it, an anti-predictive 21-day horizon, and
hit-rate evidence that was never evidence at all." See §9.

---

## 9. PHASE 1C — THE SIGNAL TEST (does the model know anything?)

Artifact: `experiments/phase1/signal_test_2026-09-18.json`
Script: `experiments/phase1/run_signal_test.py`

Same purged walk-forward, **with F5 fixed**: each fold carves an early-stopping
holdout off the end of its own training window (with an `h`-row gap so
overlapping labels cannot leak across the boundary). No hyperparameter tuned.
A seeded random prediction is used as an empirical noise band.

| Ticker | h | IC | rankIC | hit% | base% | dir_skill | t_eff | spread | spread t | +folds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SPY | 1 | −0.0606 | −0.0613 | 52.22% | 53.3% | −1.03% | −2.15 | −0.00605 | −0.78 | 1/5 |
| SPY | 21 | −0.0443 | +0.0087 | 54.44% | 67.9% | **−13.49%** | −0.34 | −0.08352 | **−5.32** | 3/5 |
| QQQ | 1 | +0.0285 | +0.0074 | 51.59% | 53.6% | −1.98% | +1.01 | −0.00096 | −0.20 | 4/5 |
| QQQ | 21 | −0.1237 | −0.1051 | 59.84% | 61.8% | −1.98% | −0.96 | −0.20780 | **−8.59** | 2/5 |
| **AAPL** | **1** | **+0.0840** | +0.0551 | 50.40% | 51.2% | −0.79% | **+2.99** | +0.00766 | +1.54 | **5/5** |
| AAPL | 21 | −0.1177 | −0.1001 | 58.10% | 59.8% | −1.75% | −0.92 | −0.05256 | −1.20 | 1/5 |
| **MSFT** | **1** | **+0.0866** | **+0.0635** | 52.38% | 51.3% | **+1.11%** | **+3.08** | +0.01394 | **+2.59** | **5/5** |
| MSFT | 21 | +0.1855 | +0.1765 | 54.29% | 59.8% | **−5.48%** | +1.46 | +0.00962 | +0.59 | 5/5 |
| GOOGL | 1 | −0.0174 | −0.0187 | 50.63% | 52.8% | −2.14% | −0.62 | −0.00316 | −0.63 | 2/5 |
| GOOGL | 21 | −0.1555 | −0.1118 | 52.22% | 58.6% | −6.35% | −1.22 | −0.11679 | **−5.25** | 2/5 |

| Aggregate | Value |
|---|---|
| **POOLED (group z-scored)** | **IC = −0.0135, t_eff = −1.51 → NO SIGNAL** |
| Random control | IC = −0.0042 (noise band confirmed) |
| Cells meeting criteria (IC>0, \|IC\|≥0.03, \|t_eff\|>2) | **2/10** |
| Cells with positive IC | **4/10** (coin-flip expectation 5/10) |

### What this establishes

1. **The pooled signal is zero.** An earlier version of this test pooled raw
   predictions across horizons and reported IC = **+0.10**. That was a scale
   artifact: h=1 and h=21 labels live on different scales, so pooling
   manufactures between-group correlation. Z-scoring within each
   (ticker, horizon) group collapses it to **−0.0135**. My own first test was
   wrong; the corrected one is in the artifact.

2. **Hit rate was never evidence.** MSFT h=21 has the largest IC in the table
   (+0.1855) yet is **5.48pp worse than always guessing "up"** (base rate
   59.8%). SPY h=21 is **13.49pp worse**. In a rising market, a model that
   predicts "up" scores the base rate for free. **This is the most likely
   explanation of the CHANGELOG's "win rate ~60%" claim.**

3. **The 21-day horizon is anti-predictive.** IC is negative for 4/5 tickers,
   and the long/short spread is significantly *negative* for SPY (−5.32),
   QQQ (−8.59) and GOOGL (−5.25). **Plan Section 9 R5 ("move to 5–21d where
   costs bite less") is now falsified** — the longer horizon is worse, not
   better, and it is not a cost problem.

4. **One narrow, real signal exists: MSFT at h=1** (IC +0.087, rankIC +0.064,
   **5/5 folds positive**, spread +0.0139 with t = +2.59, dir_skill +1.11pp),
   with **AAPL at h=1** close behind (IC +0.084, t_eff +2.99, 5/5 folds).
   Cross-referencing the corrected P&L test: MSFT h=1 gross Sharpe **+0.91** →
   net **+0.03**. So the signal is real and **costs consume all of it**.

5. **Early stopping corroborates this independently.** With F5 fixed, the
   chosen iteration is **1** in most folds — validation RMSE never improves
   after the first tree. The features carry almost nothing broadly learnable.
   Several folds did reach 200+, so the process is not degenerate.

### The refined diagnosis

> There is one narrow, real 1-day signal in **2 of 5 tickers** (MSFT, weakly
> AAPL) that is **smaller than the cost of trading it**; the 21-day horizon is
> **anti-predictive**; and the reported win rates were **market drift, not
> skill**. The system as configured has no net edge — and now we know exactly
> which of the plan's redesign levers that eliminates.

### Consequences for plan Section 9

| Lever | Status after Phase 1c |
|---|---|
| R2 higher threshold / trade less | **Only remaining cost-side lever.** MSFT h=1 gross +0.91 → net +0.03 means cost reduction is the binding constraint *there*. |
| R3 different label (triple-barrier) | Untested; plausible, since the 21-day return label is anti-predictive. |
| R5 different horizon (longer) | **Falsified.** h=21 is anti-predictive, at ~0 cost. |
| R1 market-neutralise | **Most promising untested lever** — removes base-rate/beta contamination, which §9.2 shows is what the "signal" mostly was. |
| R6 decorrelated ensemble | Premature; nothing positive to ensemble yet. |
| R7 accept the null | Still live, and now *quantified*: pooled IC ≈ 0. |

### 9.7 Why fixing the sizing (F4) does NOT rescue Gate 1

Precise magnitudes for the one surviving signal (MSFT, h=1):

| Quantity | Value |
|---|---|
| Gross annual return | **+0.0821%** |
| Net annual return | +0.0024% |
| Annualised strategy volatility | **0.0897%** |
| Average daily turnover | 0.00211 |
| Cost drag | **0.0797%** |
| Gross Sharpe | +0.91 |
| Net Sharpe | +0.03 |

Two things follow, and they point in opposite directions:

1. **Costs consume 97% of gross P&L** (0.0797% vs 0.0821%). The signal is real
   but the harvest is almost exactly cancelled by the cost of harvesting.
2. **Fixing F4 will not change this.** Sharpe is scale-invariant: scaling
   positions up so the book runs at a realistic risk level scales gross P&L,
   cost drag and volatility by the same factor, leaving net Sharpe at 0.03.
   F4 matters for the *absolute* interpretation of every number in this report
   (the book is roughly 0.4% invested) and for paper trading — but it does not
   move a Sharpe/PF gate. **The only variable that changes the ratio is
   turnover per unit of signal.**

That is what makes plan lever **R2 (trade less / higher threshold)** the single
remaining cost-side lever with evidence behind it, and it is testable:
a horizon sweep h ∈ {1,2,3,5} (h=21 is already falsified) crossed with a
prediction threshold, on MSFT+AAPL only, scored on **net** Sharpe and **net**
PF after the F2 fix.

**Pre-registered success criterion for that experiment:** net Sharpe > 0.5 AND
net PF > 1.2 out-of-sample on at least 2 of the 2 tickers. Anything less →
R7 (accept the null) and stop.