#!/usr/bin/env python3
"""Summarise Phase 1 / Phase 1b artifacts into a Gate 1 report."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL"]


def find(name_prefix, exclude=None):
    hits = sorted(
        f for f in os.listdir(HERE)
        if f.startswith(name_prefix) and f.endswith(".json")
        and (exclude is None or exclude not in f)
    )
    return os.path.join(HERE, hits[-1]) if hits else None


def true_pf(m):
    """Standard profit factor = gross profit / gross loss."""
    wr = m.get("win_rate") or 0.0
    aw = m.get("avg_win") or 0.0
    al = m.get("avg_loss") or 0.0
    if al == 0 or wr >= 1.0:
        return None
    gross_profit = wr * aw
    gross_loss = (1.0 - wr) * abs(al)
    return gross_profit / gross_loss if gross_loss else None


def main():
    mm_path = find("validation_horizon_matched_")
    p1_path = find("validation_", exclude="horizon_matched")
    if not mm_path:
        print("no horizon-matched artifact found")
        return

    with open(mm_path) as fh:
        mm = json.load(fh)

    print("=" * 100)
    print("PHASE 1b - HORIZON-MATCHED (ECONOMICALLY VALID) RESULTS")
    print("=" * 100)
    print("artifact:", os.path.basename(mm_path))
    for h in ("h1", "h21"):
        print()
        print(f"--- horizon {h} : net of costs (commission 0.001 + slippage 0.0005) ---")
        print(f"{'Ticker':<7}{'Sharpe':>8}{'PF(reported)':>14}{'PF(true)':>10}"
              f"{'WinRate':>9}{'MaxDD':>9}{'AnnRet':>9}{'Turnover':>10}{'BH AnnRet':>11}  Verdict")
        for t in TICKERS:
            d = mm["results"][t]["horizons"].get(h, {})
            if "error" in d:
                print(f"{t:<7}ERROR {d['error'][:60]}")
                continue
            n = d["net"]
            tpf = true_pf(n)
            print(
                f"{t:<7}{n['sharpe_ratio']:>8.2f}"
                f"{n['profit_factor']:>14.2f}"
                f"{(f'{tpf:.2f}' if tpf is not None else 'n/a'):>10}"
                f"{n['win_rate']*100:>8.1f}%"
                f"{n['max_drawdown']*100:>8.2f}%"
                f"{n['annual_return']*100:>8.2f}%"
                f"{(n.get('turnover') or 0):>10.4f}"
                f"{d['buy_hold']['annual_return']*100:>10.2f}%"
                f"  {d['verdict']}"
            )

    print()
    print("=" * 100)
    print("COST ATTRIBUTION - gross (pre-cost) vs net (post-cost)")
    print("=" * 100)
    for h in ("h1", "h21"):
        print()
        print(f"--- horizon {h} ---")
        print(f"{'Ticker':<7}{'Gross Sharpe':>14}{'Net Sharpe':>12}{'Cost drag (ann)':>18}"
              f"{'Gross PF(true)':>16}{'Net PF(true)':>14}")
        for t in TICKERS:
            d = mm["results"][t]["horizons"].get(h, {})
            if "error" in d:
                continue
            g, n = d["gross"], d["net"]
            gpf, npf = true_pf(g), true_pf(n)
            print(
                f"{t:<7}{g['sharpe_ratio']:>14.2f}{n['sharpe_ratio']:>12.2f}"
                f"{d['cost_drag_ann']*100:>17.2f}%"
                f"{(f'{gpf:.2f}' if gpf is not None else 'n/a'):>16}"
                f"{(f'{npf:.2f}' if npf is not None else 'n/a'):>14}"
            )

    print()
    print("=" * 100)
    print("DIAGNOSIS - WHY THE PHASE 1 NUMBERS WERE NOT TRUSTWORTHY")
    print("=" * 100)
    print(f"{'Ticker':<7}{'Phase1 Sharpe':>15}{'lag1 autocorr':>15}"
          f"{'inflation x':>12}{'deflated Sharpe':>17}")
    for t in TICKERS:
        d = mm["results"][t].get("diagnosis", {})
        if "error" in d:
            print(f"{t:<7} error")
            continue
        print(
            f"{t:<7}{d['phase1_sharpe_reproduced']:>15.2f}"
            f"{(d['lag1_autocorr'] or 0):>15.4f}"
            f"{(d['inflation_factor'] or 0):>12.2f}"
            f"{d['deflated_sharpe']:>17.2f}"
        )

    print()
    print("=" * 100)
    print("ORIGINAL PHASE 1 ARTIFACT (as recorded) - Gate 1 result")
    print("=" * 100)
    if p1_path:
        with open(p1_path) as fh:
            p1 = json.load(fh)
        print("artifact:", os.path.basename(p1_path))
        print("warning:", (p1.get("measurement_warning") or "")[:150], "...")
        g = p1.get("gate1", {})
        print("pass_count:", g.get("pass_count"), "/ 5")
        for t, v in (g.get("per_ticker_pass") or {}).items():
            print(f"  {t}: {v}")
    else:
        print("(no phase 1 artifact found in this directory)")

    print()
    print("Gate 1 pass criteria: sharpe > 0.5 AND profit_factor > 1.2 AND max_drawdown > -0.25")
    print("pass_count_by_horizon:", mm.get("gate1", {}).get("pass_count_by_horizon"))


if __name__ == "__main__":
    sys.exit(main())
