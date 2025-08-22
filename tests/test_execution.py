import pandas as pd
from portfolio.execution import apply_execution
from portfolio.fees import apply_fees

def test_tplus1_and_spread():
    # Index
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2020-01-01","2020-01-02"]), ["SPY"]], names=["date","asset"]
    )
    # Preise: exec_ref_tplus1 für Tag 2020-01-01 ist 100 (wird am 02.01. ausgeführt)
    prices = pd.DataFrame({
        "exec_ref_tplus1": [100.0, 101.0],  # t, t+1 (für Sichtbarkeit)
        "spread_cs":       [0.002, 0.002],  # 20 bps
        "open":            [99.5, 100.5],
    }, index=idx)

    # Order am 01.01.: +10 Stück
    orders = pd.DataFrame({"delta_shares":[10.0, 0.0]}, index=idx)

    trades = apply_execution(prices, orders, use_tplus1=True, use_cs_spread=True)
    # p_exec = 100 * (1 + 0.5*0.002) = 100.1
    assert abs(trades.loc[("2020-01-01","SPY"), "p_exec"] - 100.1) < 1e-8
    # spread_cost = |10| * 100 * 0.5*0.002 = 1.0
    assert abs(trades.loc[("2020-01-01","SPY"), "spread_cost"] - 1.0) < 1e-8

def test_fees():
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2020-01-01"]), ["SPY"]], names=["date","asset"]
    )
    trades = pd.DataFrame({
        "q":[10.0],
        "p_ref":[100.0],
        "p_exec":[100.1],
        "notional_abs":[1001.0],
        "spread_cost":[1.0],
    }, index=idx)

    out = apply_fees(trades, commission_bps=5)  # 5 bps
    # fees = 1001 * 0.0005 = 0.5005
    assert abs(out.loc[idx[0], "fees"] - 0.5005) < 1e-8
    assert abs(out.loc[idx[0], "total_cost"] - (1.0 + 0.5005)) < 1e-8
