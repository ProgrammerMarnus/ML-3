#!/usr/bin/env python3
"""
Phase 1c - THE SIGNAL TEST (information, not P&L).

WHY THIS EXISTS
---------------
Gate 1 failed, but a P&L failure has many possible causes: costs, sizing units
(F4), label/P&L mismatch (F1), or overfitting from a fold loop that never
early-stops (F5). All of those are fixable *if and only if* the predictions
carry information. So the decisive question is upstream of every P&L metric:

    Do the model's out-of-sample predictions correlate with what happens next?

If the answer is no, then no amount of execution, sizing, cost or horizon work
can create an edge, and the honest outcome is plan Section 9 option R7
(accept the null). This is also what README prescribes: evaluate forecast
quality with IC, rank IC and hit rate BEFORE optimising reward.

WHAT IS MEASURED
----------------
Same purged/embargoed walk-forward as Gate 1, same features, same labels:
  * Pearson IC     : corr(prediction, realised forward return)
  * Rank IC        : Spearman corr (robust to outliers)
  * Hit rate       : fraction of correct directional calls
  * A seeded RANDOM prediction control, so the noise band is empirical rather
    than assumed. If the model's IC sits inside the random band, there is no
    signal to find.

F5 IS FIXED HERE: each fold carves an early-stopping holdout off the END of
its own training window (with an h-row gap so overlapping labels cannot leak
across the boundary). The holdout is never part of the test fold. This is the
only change from the Gate 1 fold loop; no hyperparameter is tuned.

PRE-REGISTERED INTERPRETATION (fixed before running):
    |IC| < 0.02 .......................... no exploitable signal
    0.02 <= |IC| < 0.03 .................. indeterminate, needs more data
    |IC| >= 0.03 with t_eff > 2 .......... candidate signal worth pursuing
    Hit-rate 95% CI containing 0.5 ....... no directional signal
    model IC inside the random band ...... no signal (decisive)
"""

from __future__ import annotations

import datetime
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats as sps

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = HERE
for _ in range(5):
    if os.path.isdir(os.path.join(REPO, "market_predictor_ml")):
        break
    REPO = os.path.dirname(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from market_predictor_ml.backtest.engine import WalkForwardSplit  # noqa: E402
from run_horizon_matched import (  # noqa: E402
    TICKERS,
    N_SPLITS,
    TEST_SIZE,
    PURGE,
    EMBARGO,
    build_panel,
    _new_model,
)

HORIZONS = [1, 21]
RANDOM_SEED = 12345
EARLY_STOP_HOLDOUT_FRACTION = 0.2
EARLY_STOP_ROUNDS = 50


def fold_predictions(X, y, horizon, seed_offset: int = 0) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Out-of-sample predictions from the purged walk-forward.

    Each fold trains on train_idx minus a holdout tail (gap = `horizon` rows to
    respect label overlap) and predicts test_idx. Returns (pred, y_test, fold_ids).
    """
    splitter = WalkForwardSplit(
        n_splits=N_SPLITS,
        test_size=TEST_SIZE,
        purge_size=PURGE,
        embargo_size=EMBARGO,
    )

    preds_all, y_all, fold_ids = [], [], []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        n_hold = max(int(len(train_idx) * EARLY_STOP_HOLDOUT_FRACTION), 50)
        fit_idx = train_idx[:-n_hold]
        hold_end = len(train_idx) - horizon          # gap for label overlap
        hold_idx = train_idx[len(train_idx) - n_hold:hold_end]
        if len(hold_idx) < 20:
            hold_idx = train_idx[len(train_idx) - n_hold:]
        if len(fit_idx) < 50:
            fit_idx = train_idx

        model = _new_model()
        model.early_stopping_rounds = EARLY_STOP_ROUNDS
        model.fit(
            X[fit_idx],
            y[fit_idx],
            eval_set=(X[hold_idx], y[hold_idx]),
        )
        pred = model.predict(X[test_idx])

        preds_all.append(pred)
        y_all.append(y[test_idx])
        fold_ids.extend([fold] * len(test_idx))

    return np.concatenate(preds_all), np.concatenate(y_all), fold_ids


def ic_metrics(pred: np.ndarray, actual: np.ndarray) -> Dict:
    """
    Pearson IC, Spearman rank IC, directional hit rate, and base-rate-robust
    skill measures.

    Hit rate alone is misleading: in a rising market a model that always
    predicts "up" scores at the base rate (fraction of positive actuals)
    without any skill. So we also report:
      * base_rate  - fraction of actuals > 0
      * dir_skill  - hit_rate minus the best constant-guess rate
                     (max(base_rate, 1-base_rate)); negative means the model is
                     worse than always guessing the majority direction.
      * spread     - mean(actual | pred > 0) - mean(actual | pred < 0), i.e. the
                     long/short spread the signal actually earns. This is
                     immune to the base rate and is the traded quantity.
    """
    ok = np.isfinite(pred) & np.isfinite(actual)
    pred, actual = pred[ok], actual[ok]
    n = len(pred)
    if n < 10 or np.std(pred) == 0 or np.std(actual) == 0:
        return {"n": n, "ic": None, "rank_ic": None, "hit_rate": None}

    ic = float(np.corrcoef(pred, actual)[0, 1])
    rank_ic = float(sps.spearmanr(pred, actual).correlation)
    hit = float(np.mean(np.sign(pred) == np.sign(actual)))
    base_rate = float(np.mean(actual > 0))
    dir_skill = float(hit - max(base_rate, 1.0 - base_rate))

    longs = actual[pred > 0]
    shorts = actual[pred < 0]
    if len(longs) >= 5 and len(shorts) >= 5:
        spread = float(longs.mean() - shorts.mean())
        se = float(
            np.sqrt(longs.var(ddof=1) / len(longs) + shorts.var(ddof=1) / len(shorts))
        )
        spread_t = float(spread / se) if se > 0 else 0.0
    else:
        spread, spread_t = None, None

    return {
        "n": n,
        "ic": ic,
        "rank_ic": rank_ic,
        "hit_rate": hit,
        "base_rate": base_rate,
        "dir_skill": dir_skill,
        "spread": spread,
        "spread_t": spread_t,
        "n_long": int(len(longs)),
        "n_short": int(len(shorts)),
    }


def hit_rate_ci(hits: int, n: int) -> Tuple[float, float]:
    """Wilson 95% interval for a hit rate."""
    if n == 0:
        return (0.0, 1.0)
    z = 1.959963985
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (float(centre - half), float(centre + half))


def effective_t(ic: float, n: int, horizon: int) -> float:
    """
    t-statistic for a correlation with an overlap correction.

    Overlapping h-day labels make the effective sample ~n/h, so the naive
    t = ic*sqrt(n-2)/sqrt(1-ic^2) is deflated by sqrt(horizon). Conservative
    by design: we would rather under-claim a signal than over-claim one.
    """
    if ic is None or n < 10 or abs(ic) >= 1.0:
        return 0.0
    t_naive = ic * np.sqrt(n - 2) / np.sqrt(1 - ic * ic)
    return float(t_naive / np.sqrt(max(1, horizon)))

def main() -> None:
    out_dir = HERE
    rng = np.random.default_rng(RANDOM_SEED)
    results: Dict = {}
    pooled_model_pred: List[np.ndarray] = []
    pooled_model_act: List[np.ndarray] = []
    pooled_rand_pred: List[np.ndarray] = []
    pooled_rand_act: List[np.ndarray] = []

    for ticker in TICKERS:
        per_ticker: Dict = {}
        for h in HORIZONS:
            print(f"[{ticker}] h={h} ...", flush=True)
            panel = build_panel(ticker, f"Target_RiskAdj_{h}d")
            X, y = panel["X"], panel["y"]

            pred, actual, fold_ids = fold_predictions(X, y, h)
            m = ic_metrics(pred, actual)
            hits = int(np.sum(np.sign(pred) == np.sign(actual)))
            lo, hi = hit_rate_ci(hits, m["n"])
            m["hit_count"] = hits
            m["hit_ci95"] = [lo, hi]
            m["t_eff"] = effective_t(m["ic"], m["n"], h)

            # Per-fold ICs, to see whether any fold carries the signal.
            per_fold = []
            fold_ids_arr = np.asarray(fold_ids)
            for fid in sorted(set(fold_ids)):
                sel = fold_ids_arr == fid
                fm = ic_metrics(pred[sel], actual[sel])
                per_fold.append({"fold": int(fid), "ic": fm["ic"],
                                 "hit_rate": fm["hit_rate"], "n": fm["n"]})
            m["per_fold"] = per_fold
            m["folds_with_positive_ic"] = int(sum(
                1 for f in per_fold if (f["ic"] or 0) > 0))

            # Random control with the same sample sizes.
            rpred = rng.standard_normal(len(pred))
            rm = ic_metrics(rpred, actual)
            rhits = int(np.sum(np.sign(rpred) == np.sign(actual)))
            rlo, rhi = hit_rate_ci(rhits, rm["n"])
            rm["hit_ci95"] = [rlo, rhi]
            m["random_control"] = rm
            m["beats_random_ic"] = (
                None if m["ic"] is None or rm["ic"] is None
                else bool(abs(m["ic"]) > abs(rm["ic"]))
            )

            # Z-score within this (ticker, horizon) group before pooling: h=1
            # and h=21 labels live on different scales, so pooling raw values
            # manufactures a between-group correlation that is not tradable.
            zp = (pred - pred.mean()) / (pred.std() or 1.0)
            za = (actual - actual.mean()) / (actual.std() or 1.0)
            zr = (rpred - rpred.mean()) / (rpred.std() or 1.0)
            pooled_model_pred.append(zp)
            pooled_model_act.append(za)
            pooled_rand_pred.append(zr)
            pooled_rand_act.append(za)

            per_ticker[f"h{h}"] = m

            print(
                f"    IC={m['ic']:+.4f} rankIC={m['rank_ic']:+.4f} "
                f"hit={m['hit_rate']*100:.2f}% base={m['base_rate']*100:.1f}% "
                f"skill={m['dir_skill']*100:+.2f}pp t_eff={m['t_eff']:+.2f} "
                f"| spread={m['spread']:+.5f} t={m['spread_t']:+.2f} "
                f"| rand IC={rm['ic']:+.4f} | n={m['n']}",
                flush=True,
            )

        results[ticker] = per_ticker

    # Pooled, group-standardised (meaningful) - see z-scoring above.
    pm = ic_metrics(np.concatenate(pooled_model_pred), np.concatenate(pooled_model_act))
    rm = ic_metrics(np.concatenate(pooled_rand_pred), np.concatenate(pooled_rand_act))
    pm["t_eff"] = effective_t(pm["ic"], pm["n"], 1)
    pooled = {"model_group_standardised": pm, "random_control": rm}

    stamp = datetime.date.today().isoformat()
    out_path = os.path.join(out_dir, f"signal_test_{stamp}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "generated_at": stamp,
                "purpose": "Phase 1c - out-of-sample forecast quality (IC / rank IC / hit rate)",
                "early_stopping": {
                    "applied": True,
                    "holdout_fraction": EARLY_STOP_HOLDOUT_FRACTION,
                    "gap_rows": "horizon",
                    "rounds": EARLY_STOP_ROUNDS,
                },
                "pre_registered_thresholds": {
                    "no_signal_abs_ic_below": 0.02,
                    "candidate_signal_abs_ic_at_least": 0.03,
                    "candidate_signal_t_eff_above": 2.0,
                },
                "pooled": pooled,
                "results": results,
            },
            fh,
            indent=2,
            default=str,
        )
    print(f"\nSaved: {out_path}")

    print("\n" + "=" * 96)
    print("SIGNAL TEST SUMMARY (out-of-sample, purged walk-forward, early-stopping ON)")
    print("=" * 96)
    print(f"{'Ticker':<7}{'h':>3}{'IC':>9}{'rankIC':>9}{'hit%':>7}{'base%':>7}"
          f"{'skill':>8}{'t_eff':>8}{'spread':>10}{'spread t':>10}{'+folds':>8}")
    for t in TICKERS:
        for h in HORIZONS:
            m = results[t][f"h{h}"]
            sp = m["spread"] if m["spread"] is not None else float("nan")
            spt = m["spread_t"] if m["spread_t"] is not None else float("nan")
            print(
                f"{t:<7}{h:>3}{m['ic']:>+9.4f}{m['rank_ic']:>+9.4f}"
                f"{m['hit_rate']*100:>6.2f}%{m['base_rate']*100:>6.1f}%"
                f"{m['dir_skill']*100:>+7.2f}%{m['t_eff']:>+8.2f}"
                f"{sp:>+10.5f}{spt:>+10.2f}{m['folds_with_positive_ic']:>6}/5"
            )
    print("-" * 96)
    print(f"POOLED (group z-scored)  IC={pm['ic']:+.4f}  rankIC={pm['rank_ic']:+.4f}  "
          f"hit={pm['hit_rate']*100:.2f}%  base={pm['base_rate']*100:.1f}%  "
          f"skill={pm['dir_skill']*100:+.2f}pp  t_eff={pm['t_eff']:+.2f}  n={pm['n']}")
    print(f"RANDOM CONTROL           IC={rm['ic']:+.4f}  rankIC={rm['rank_ic']:+.4f}  "
          f"hit={rm['hit_rate']*100:.2f}%  base={rm['base_rate']*100:.1f}%  "
          f"skill={rm['dir_skill']*100:+.2f}pp  n={rm['n']}")

    # Pre-registered decision, evaluated per ticker (the tradable unit) AND on
    # the group-standardised pool. Pooled alone is not sufficient: a single
    # ticker could carry a pool.
    cand = []
    for t in TICKERS:
        for h in HORIZONS:
            m = results[t][f"h{h}"]
            if m["ic"] is not None and m["ic"] > 0 and abs(m["ic"]) >= 0.03 \
                    and abs(m["t_eff"]) > 2.0:
                cand.append(f"{t} h={h} (IC={m['ic']:+.3f}, t={m['t_eff']:+.2f})")
    n_pos = sum(1 for t in TICKERS for h in HORIZONS
                if (results[t][f"h{h}"]["ic"] or 0) > 0)

    print()
    print(f"Per-ticker cells meeting criteria (IC>0, |IC|>=0.03, |t_eff|>2): "
          f"{len(cand)}/10")
    for c in cand:
        print(f"    {c}")
    print(f"Cells with positive IC: {n_pos}/10   "
          f"(coin-flip expectation is 5/10)")

    pooled_verdict = (
        "NO SIGNAL" if abs(pm["ic"]) < 0.02
        else "CANDIDATE SIGNAL" if abs(pm["ic"]) >= 0.03 and abs(pm["t_eff"]) > 2
        else "INDETERMINATE"
    )
    print(f"POOLED VERDICT: {pooled_verdict}  "
          f"(|IC|={abs(pm['ic']):.4f}, t_eff={pm['t_eff']:+.2f})")

    print()
    print("REFERENCE (pre-registered): |IC|<0.02 = no signal; |IC|>=0.03 with "
          "|t_eff|>2 = candidate")


if __name__ == "__main__":
    main()