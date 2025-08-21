import pandas as pd
import numpy as np
import pytest

# Versuche, die benötigten Funktionen zu importieren; falls das Modul
# (noch) nicht existiert, werden die Tests übersprungen.
exec_mod = pytest.importorskip("src.execution")
apply_execution = exec_mod.apply_execution
apply_fees = exec_mod.apply_fees


def test_apply_execution_and_fees():
    # Mini-DataFrame: 2 Tage, 1 Asset
    dates = pd.date_range("2020-01-01", periods=2, freq="D")
    df = pd.DataFrame(
        {
            "open": [90.0, 95.0],  # open_t ≠ exec_ref_tplus1
            "exec_ref_tplus1": [100.0, 101.0],
            # Spread des Folgetags (t+1) wird für Tag t verwendet
            "spread_cs": [np.nan, 0.002],
            # Handelsmenge: an Tag t +10 kaufen
            "q": [10.0, 0.0],
        },
        index=dates,
    )

    out = apply_execution(df.copy())

    t = dates[0]
    assert out.loc[t, "p_exec"] == pytest.approx(100.1)
    assert out.loc[t, "spread_cost"] == pytest.approx(1.0)
    # Sicherstellen, dass nicht der open-Preis von Tag t verwendet wurde
    assert out.loc[t, "p_exec"] != df.loc[t, "open"]

    out2 = apply_fees(out.copy(), commission_bps=5)
    assert out2.loc[t, "fees"] == pytest.approx(0.5005)
    assert out2.loc[t, "total_cost"] == pytest.approx(1.5005)

