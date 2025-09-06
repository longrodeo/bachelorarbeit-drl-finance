# tests/test_cvar_icvar.py
import math
import numpy as np
import pandas as pd
import pytest
import math

# Wir testen gegen eure Implementierung.
# Falls der Import (Name/Pfad) abweicht: hier anpassen.
try:
    from accounting.evaluator import mts_var_cvar_icvar  # as_series Flag: True->Serien, False->Skalare
except Exception as e:
    pytest.skip(f"icvar_mts nicht importierbar: {e}", allow_module_level=True)

# --- Kleine Hilfen nur für die Tests (keine Produktionsfunktionen) ---

def mts_cvar_loss_from_returns(r: np.ndarray, alpha: float):
    r = np.asarray(r, float)
    t = len(r)
    var_k = np.array([np.quantile(r[:k+1], alpha, method="linear") for k in range(t)])
    var_loss_k = -var_k
    L = -r
    excess = np.maximum(L - var_loss_k, 0.0)
    var_t_loss = var_loss_k[-1]
    cvar_t_loss = var_t_loss + excess.sum() / (alpha * t)
    return float(var_t_loss), float(cvar_t_loss)

def normal_loss_theory(alpha: float, mu: float, sigma: float):
    """
    Theoretische VaR/CVaR (VERLUST-Skala) bei X~N(mu,sigma^2), L=-X:
      VaR_loss = -(mu + sigma * z_alpha)
      CVaR_loss = -(mu + sigma * E[X | X <= z_alpha])
                = -mu + sigma * phi(z_alpha) / alpha
    wobei z_alpha = Φ^{-1}(alpha) (linkes Quantil).
    """
    def Phi(z):  # Standardnormal-CDF
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    # ppf via Bisection:
    lo, hi = -10.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if Phi(mid) < alpha:
            lo = mid
        else:
            hi = mid
    z = (lo + hi) / 2.0
    phi = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    var_loss = -(mu + sigma * z)
    cvar_loss = -mu + sigma * (phi / alpha)
    return var_loss, cvar_loss

# ----------------------------- Tests ------------------------------------

def test_handcrafted_tail_mean_matches():
    """
    Minibeispiel per Hand: r = [+1%, -2%, 0%, -5%], alpha=0.25.
    Erwartung: CVaR (loss) aus icvar_mts == Tail-Mittel-Formel (loss).
    """
    r = pd.Series([0.01, -0.02, 0.00, -0.05])
    alpha = 0.25

    # Serie bis t, ex-post, min_period=1 (damit alle Präfixe gerechnet werden)
    var_s, cvar_s, _ = mts_var_cvar_icvar(
        r, alpha=alpha, min_period=1, include_current=True, ewm_alpha=None, as_series=True
    )
    var_t = float(var_s.iloc[-1]); cvar_t = float(cvar_s.iloc[-1])

    var_loss_ref, cvar_loss_ref = mts_cvar_loss_from_returns(r.values, alpha)

    assert abs(var_t - var_loss_ref)  < 1e-12
    assert abs(cvar_t - cvar_loss_ref) < 1e-12

def test_ru_formula_equivalence_mts_discrete():
    """
    Prüft die diskrete MTS/Rockafellar-Uryasev-Gleichung:
      CVaR_{t} = VaR_{t} + (1/(α·t)) * Σ_{k≤t} max( L_k - VaR_k, 0 )
    mit VaR_k als Präfix-Quantil für jeden k.
    """
    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(0.0, 0.01, size=200))  # 200 Schritte
    alpha = 0.05

    var_s, cvar_s, _ = mts_var_cvar_icvar(
        r, alpha=alpha, min_period=20, include_current=True, ewm_alpha=None, as_series=True
    )
    # Hand-Rekonstruktion (identisch zur Formel)
    r_use = r
    n = np.arange(1, len(r_use)+1, dtype=float)
    var_ret = r_use.expanding(min_periods=20).apply(
        lambda a: np.quantile(a[np.isfinite(a)], alpha, method="linear"), raw=True
    )
    var_loss = -var_ret
    L = -r_use
    excess = (L - var_loss).clip(lower=0.0).fillna(0.0)
    cum_excess = excess.cumsum()
    cvar_formula = (var_loss + cum_excess / (alpha * n)).where(n >= 20)

    # Vergleich nur ab erstem gültigen Index
    start = cvar_s.first_valid_index()
    assert start is not None
    diff = (cvar_s.loc[start:] - cvar_formula.loc[start:]).abs().max()
    assert float(diff) < 1e-12

def test_normal_theory_matches_empirical_final_t():
    """
    Simulation aus N(0, sigma^2) und Vergleich mit Theorie:
      VaR_loss =  sigma * z_{1-α}  (entspricht -(mu + sigma*z_α) bei mu=0)
      CVaR_loss = sigma * φ(z_α) / α
    (wir nutzen z_α = Φ^{-1}(α) für linkes Quantil der Returns)
    """
    rng = np.random.default_rng(42)
    sigma = 0.01
    r = pd.Series(rng.normal(0.0, sigma, size=50_000))  # groß, aber schnell genug
    alpha = 0.05

    var_s, cvar_s, _ = mts_var_cvar_icvar(
        r, alpha=alpha, min_period=100, include_current=True, ewm_alpha=None, as_series=True
    )
    var_emp = float(var_s.iloc[-1]); cvar_emp = float(cvar_s.iloc[-1])

    var_th, cvar_th = normal_loss_theory(alpha=alpha, mu=0.0, sigma=sigma)

    # Toleranzen: stochastisch ~ O(sigma/sqrt(n)); hier sehr eng möglich
    assert abs(var_emp - var_th)  < 3e-4
    assert abs(cvar_emp - cvar_th) < 3e-4

def test_monotonicity_in_alpha():
    """
    α kleiner (z. B. 5% -> 1%) => Tail extremer => VaR/CVaR (loss) steigen.
    """
    rng = np.random.default_rng(3)
    r = pd.Series(rng.standard_t(df=5, size=5000) * 0.01)  # heavy tail
    alpha1, alpha2 = 0.10, 0.05

    var1, cvar1, _ = mts_var_cvar_icvar(r, alpha=alpha1, min_period=200, include_current=True, as_series=True)
    var2, cvar2, _ = mts_var_cvar_icvar(r, alpha=alpha2, min_period=200, include_current=True, as_series=True)

    v1, v2 = float(var1.iloc[-1]), float(var2.iloc[-1])
    c1, c2 = float(cvar1.iloc[-1]), float(cvar2.iloc[-1])

    assert v2 >= v1  # strengeres α → größerer Verlust-VaR
    assert c2 >= c1  # dito für CVaR

def test_translation_and_scaling_on_loss_scale():
    """
    Für Y = a*X + c (a>=0):
      L_Y = -Y = a*(-X) - c = a*L_X - c
      => CVaR_loss(Y) = a*CVaR_loss(X) - c
    Gleiches gilt für VaR_loss.
    """
    rng = np.random.default_rng(9)
    r = pd.Series(rng.normal(0.001, 0.01, size=5000))  # leicht positive Drift
    alpha = 0.05
    a, c = 1.7, 0.003

    var_x, cvar_x, _ = mts_var_cvar_icvar(r, alpha=alpha, min_period=200, include_current=True, as_series=True)
    var_y, cvar_y, _ = mts_var_cvar_icvar(a*r + c, alpha=alpha, min_period=200, include_current=True, as_series=True)

    vx, vy = float(var_x.iloc[-1]), float(var_y.iloc[-1])
    cx, cy = float(cvar_x.iloc[-1]), float(cvar_y.iloc[-1])

    assert abs(vy - (a*vx - c)) < 5e-4
    assert abs(cy - (a*cx - c)) < 5e-4

def test_expost_vs_exante_alignment():
    """
    ex-ante (bis t-1) muss zur ex-post-Serie um 1 nach links verschoben deckungsgleich sein:
      CVaR_ex_ante[t] == CVaR_ex_post[t-1]  (ab erstem gültigen Schritt)
    """
    rng = np.random.default_rng(13)
    r = pd.Series(rng.normal(0.0, 0.01, size=2000))
    alpha = 0.05
    mp = 60

    _, cvar_post, _ = mts_var_cvar_icvar(r, alpha=alpha, min_period=mp, include_current=True,  as_series=True)
    _, cvar_ante, _ = mts_var_cvar_icvar(r, alpha=alpha, min_period=mp, include_current=False, as_series=True)

    # ex-ante (bis t-1) muss ex-post (bis t) um 1 nach links entsprechen:
    # cvar_ante[t] == cvar_post[t-1]
    post_shifted = cvar_post.shift(1)  # ex-post um 1 verschoben
    diff = (cvar_ante - post_shifted).abs().dropna()  # nur gemeinsame gültige Stellen

    assert float(diff.max()) < 1e-12

def test_series_vs_scalar_last_values_match():
    """
    as_series=True (Serie) und as_series=False (Skalare) müssen am Ende dieselben Werte liefern.
    """
    rng = np.random.default_rng(21)
    r = pd.Series(rng.normal(0.0, 0.01, size=500))
    alpha, mp = 0.05, 60

    var_s, cvar_s, icvar_s = mts_var_cvar_icvar(
        r, alpha=alpha, min_period=mp, include_current=True, as_series=True
    )
    icvar_t, cvar_t, cvar_tm1 = mts_var_cvar_icvar(
        r, alpha=alpha, min_period=mp, include_current=True, as_series=False
    )

    # Serie -> letzte/letzte-1
    s = cvar_s.dropna()
    assert len(s) >= 2
    c_last, c_prev = float(s.iloc[-1]), float(s.iloc[-2])
    i_last = float((cvar_s - cvar_s.shift(1)).iloc[-1])

    assert abs(cvar_t - c_last)   < 1e-12
    assert abs(cvar_tm1 - c_prev) < 1e-12
    assert abs(icvar_t - i_last)  < 1e-12
