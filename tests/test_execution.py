"""
Tests für Ausführungs- und Fee-Berechnung.
Überprüft korrekte Anwendung von T+1-Logik, Spread-Kosten und Provisionen.
"""

# pandas für MultiIndex und DataFrame-Konstruktion in den Testfällen
import pandas as pd
# zu testende Funktion: berechnet Ausführungslogik inkl. T+1 und Spread
from portfolio.execution import apply_execution
# fügt Kommissionsgebühren zu Trades hinzu
from portfolio.fees import apply_fees

def test_tplus1_and_spread():
    """Ausführung am Folgetag mit Corwin-Schultz-Spread prüfen."""
    # Index
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2020-01-01","2020-01-02"]), ["SPY"]], names=["date","asset"]
    )  # MultiIndex repräsentiert zwei Handelstage für das Asset SPY
    # Preise: exec_ref_tplus1 für Tag 2020-01-01 ist 100 (wird am 02.01. ausgeführt)
    prices = pd.DataFrame({
        "exec_ref_tplus1": [100.0, 101.0],  # t, t+1 (für Sichtbarkeit)
        "spread_cs":       [0.002, 0.002],  # 20 bps
        "open":            [99.5, 100.5],
    }, index=idx)  # DataFrame mit notwendigen Spalten für Ausführung

    # Order am 01.01.: +10 Stück
    orders = pd.DataFrame({"delta_shares":[10.0, 0.0]}, index=idx)  # zweite Zeile = keine Order am 02.01.

    trades = apply_execution(prices, orders, use_tplus1=True, use_cs_spread=True)  # führt Ausführung mit Spread-Korrektur aus
    # p_exec = 100 * (1 + 0.5*0.002) = 100.1
    assert abs(trades.loc[("2020-01-01","SPY"), "p_exec"] - 100.1) < 1e-8  # Ausführungspreis mit halbem Spread-Anteil
    # spread_cost = |10| * 100 * 0.5*0.002 = 1.0
    assert abs(trades.loc[("2020-01-01","SPY"), "spread_cost"] - 1.0) < 1e-8  # erwartete Spread-Kosten

def test_fees():
    """Kommissionskosten auf bereits berechnete Trades anwenden."""
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2020-01-01"]), ["SPY"]], names=["date","asset"]
    )  # ein Handelstag für ein Asset
    trades = pd.DataFrame({
        "q":[10.0],  # gehandelte Stückzahl
        "p_ref":[100.0],  # Referenzpreis
        "p_exec":[100.1],  # Ausführungspreis
        "notional_abs":[1001.0],  # absoluter Geldbetrag
        "spread_cost":[1.0],  # bereits berechnete Spread-Kosten
    }, index=idx)  # vorbereitete Trades als Input

    out = apply_fees(trades, commission_bps=5)  # 5 bps
    # fees = 1001 * 0.0005 = 0.5005
    assert abs(out.loc[idx[0], "fees"] - 0.5005) < 1e-8  # korrekte Kommissionskosten
    assert abs(out.loc[idx[0], "total_cost"] - (1.0 + 0.5005)) < 1e-8  # Spread + Fees = total_cost
