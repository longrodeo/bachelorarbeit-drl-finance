# src/env/smoke_test_env.py
# Kompakter, kommentierter Smoke-Test für TradingEnv (ohne Datei-I/O hier).
# Erwartet ein INJECT-Dict (wird von deiner *_run.py gebaut).
# Wir ändern NICHT deinen AccountingRecorder/Evaluator – nur Debug-Ausgaben.

from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd



# ===== Schalter / Einstellungen ===============================================
AUDIT_ASSET_INDEX: int = 0              # Welches Asset im t vs. t+1 Audit zeigen
MAX_STEPS: Optional[int] = 30           # Anzahl Schritte begrenzen (None = volle Episode)
STRICT_ASSERTS: bool = False            # True: harte Asserts, False: nur Warnungen
SHOW_MISSING_PRICE_ASSETS: bool = True  # Assets mit fehlenden adj_open/adj_close auflisten
# ==============================================================================


# --- kleine Helfer -------------------------------------------------------------
def _need(d: dict, key: str):
    """Pflichtfeld aus INJECT holen, sonst klarer Fehler."""
    if key in d and d[key] is not None:
        return d[key]
    raise NameError(f"Fehlt in INJECT: '{key}'")

def _opt(d: dict, key: str, default=None):
    """Optionales Feld aus INJECT holen, sonst Default."""
    return d[key] if key in d else default

def _shape(x) -> Optional[tuple]:
    return getattr(x, "shape", None)

def _warn(msg: str):
    print(f"[WARN] {msg}")

def _maybe_assert(cond: bool, msg: str):
    """Je nach STRICT_ASSERTS hart abbrechen oder nur warnen."""
    if STRICT_ASSERTS and not cond:
        raise AssertionError(msg)
    if not cond:
        _warn(msg)


# --- Haupteinstieg -------------------------------------------------------------
def main(INJECT: Dict[str, Any]) -> None:
    """
    Smoke-Test für TradingEnv mit gut lesbaren Audits.

    INJECT muss liefern:
      - panel_clean : MultiIndex-DataFrame [date, asset]
      - dates       : Datumsindex oder Liste
      - assets      : Liste der Assets (Reihenfolge)
      - spec        : StateSpec (mit per_asset_features)
      - state_builder: Modul/Funktion (build_state_for_date)
      - portfolio   : PortfolioLite-Instanz
      - rf_factor   : täglicher risikofreier Faktor (np.array/Series)

    Optional:
      - rf_rate, evaluator, reward_scaler, recorder, reward_kind, icvar-Settings, …
    """
    # 1) Abhängigkeiten aus INJECT
    panel_clean   = _need(INJECT, "panel_clean")
    panel_features = _need(INJECT, "panel_features")
    dates         = _opt(INJECT, "dates")
    assets        = _opt(INJECT, "assets")
    spec          = _need(INJECT, "spec")
    state_builder = _need(INJECT, "state_builder")
    portfolio     = _need(INJECT, "portfolio")
    rf_factor     = _need(INJECT, "rf_factor")

    rf_rate       = _opt(INJECT, "rf_rate")
    evaluator     = _opt(INJECT, "evaluator")
    reward_scaler = _opt(INJECT, "reward_scaler")
    recorder      = _opt(INJECT, "recorder")

    reward_kind      = _opt(INJECT, "reward_kind", "log")
    icvar_alpha      = float(_opt(INJECT, "icvar_alpha", 0.05))
    icvar_min_period = int(_opt(INJECT, "icvar_min_period", 1))
    icvar_ewm_alpha  = _opt(INJECT, "icvar_ewm_alpha", None)
    icvar_mode       = _opt(INJECT, "icvar_mode", "ex_post")
    lambda_icvar     = float(_opt(INJECT, "lambda_icvar", 1.0))
    gamma_dd         = float(_opt(INJECT, "gamma_dd", 0.0))
    start_idx        = int(_opt(INJECT, "start_idx", 0))
    end_idx_excl     = _opt(INJECT, "end_idx_exclusive", None)

    # Fallbacks, falls nicht injiziert
    if dates is None:
        dates = panel_clean.index.get_level_values(0).unique().sort_values()
    if assets is None:
        assets = list(panel_clean.index.get_level_values(1).unique())
    A = len(assets)

    # 2) Env bauen (nur per Dependency Injection – keine neuen Komponenten)
    from src.env.trading_env import TradingEnv
    env = TradingEnv(
        panel_clean=panel_clean,
        panel_features=panel_features,
        dates=dates,
        assets=assets,
        spec=spec,
        state_builder=state_builder,
        portfolio=portfolio,
        # Genug Start-Cash wählen, damit Lots-Rundung nicht alles auf 0 rundet:
        initial_cash=float(_opt(INJECT, "initial_cash", 1_000_000.0)),
        rf_factor=rf_factor,
        rf_rate=rf_rate,
        reward_kind=reward_kind,
        reward_scaler=reward_scaler,
        evaluator=evaluator,
        icvar_alpha=icvar_alpha,
        icvar_min_period=icvar_min_period,
        icvar_ewm_alpha=icvar_ewm_alpha,
        icvar_mode=icvar_mode,
        lambda_icvar=lambda_icvar,
        gamma_dd=gamma_dd,
        start_idx=start_idx,
        end_idx_exclusive=end_idx_excl,
        recorder=recorder,
    )

    # 3) Reset & Grundausgaben
    obs, info = env.reset()
    t = int(env.t)
    date_t  = env.dates[t]
    date_t1 = env.dates[t + 1]

    # 3a) Welche Features @t gehen in den State? (Leakage-Sanity)
    from src.state.state_builder import build_state_for_date
    snap0 = {"weights": pd.Series(0.0, index=assets), "cash": float(env.cash)}
    s_dbg = build_state_for_date(panel_clean=panel_clean, date=date_t,
                                 spec=spec, assets_order=assets,
                                 portfolio_snapshot=snap0, riskfree=float(env.RF_RATE[t]))

    print("[STATE] Datum (t) =", date_t)
    print("[STATE] Feature-Namen =", s_dbg["features"][:10], "... gesamt:", len(s_dbg["features"]))
    print("[TRADE] Datum (t+1) =", date_t1)

    print("RESET:")
    print("  Shapes:",
          "X", _shape(obs.get("X")),
          "g_scalars", _shape(obs.get("g_scalars")),
          "g_weights", _shape(obs.get("g_weights")),
          "position", _shape(obs.get("position")))
    print("  Info: date", info.get("date"),
          "value", info.get("value"),
          "cash", info.get("cash"),
          "equity", info.get("equity"))

    # 3b) Ein Asset detailliert vergleichen: Features@t vs. Preise@t+1
    if A > 0:
        asset_idx = int(np.clip(AUDIT_ASSET_INDEX, 0, A - 1))
        asset = assets[asset_idx]
        df_t  = panel_features.xs(date_t,  level=0).loc[asset, spec.per_asset_features]
        df_t1 = panel_features.xs(date_t1, level=0).loc[asset, spec.per_asset_features]
        print(f"[AUDIT] asset={asset}  t={date_t}  t+1={date_t1}")
        print("[AUDIT] features@t (erste 8):", dict(df_t.head(8)))
        print("[AUDIT] features@t+1 (erste 8):", dict(df_t1.head(8)))
        px_t1 = panel_clean.xs(date_t1, level=0).loc[asset]
        print("[AUDIT] adj_open@t+1:", float(px_t1.get("adj_open", float("nan"))),
              "adj_close@t+1:", float(px_t1.get("adj_close", float("nan"))),
              "spread(cs)@t+1:", float(px_t1.get("bid_ask_spread_corwin_schultz", 0.0)))


    max_steps = MAX_STEPS if MAX_STEPS is not None else (env.last_step - env.start_idx + 1)
    max_steps = int(min(max_steps, (env.last_step - env.start_idx + 1)))

    did_exec_audit = False


    for k in range(max_steps):
        # WICHTIG: Trade-Datum (t+1) VOR dem step merken (vermeidet Off-by-one)
        t_before = int(env.t)
        trade_date = env.dates[env.t + 1]

        # === TEST-ACTION: ab erstem handelbaren Schritt (t+1 >= 20) ===
        A = getattr(env, "A", len(getattr(env, "assets", [])))  # Anzahl Assets
        if env.t >= 19:
            a = np.zeros(A + 1)  # +1 für Cash
            a[:A] = 0.30 / A  # 30% gleichverteilt auf Assets
            a[-1] = 1.0 - a[:A].sum()  # Rest in Cash
        else:
            a = np.zeros(A + 1);
            a[:A] = 0.30 / A


        obs, reward, terminated, truncated, info = env.step(a)

        # 4a) Fehlende Preise an trade_date melden (z. B. ETH vor 2015)
        if SHOW_MISSING_PRICE_ASSETS:
            px_t1 = panel_clean.xs(trade_date, level=0)
            bad = px_t1[["adj_open_raw", "adj_close_raw"]].isna().any(axis=1)
            missing = px_t1.index[bad].tolist()
            if missing:
                print(f"[AUDIT] fehlende adj_open/adj_close @ {trade_date}: {missing}")

        # 4b) Einmalig: Ausführung ALLER gehandelten Assets detailliert zeigen
        if not did_exec_audit:
            exec_df = info.get("trades")
            fees_df = info.get("fees_detail")
            ctrl    = (info.get("controls", {}) or {})
            if exec_df is not None and hasattr(exec_df, "empty") and not exec_df.empty:
                nz = exec_df[exec_df["q"] != 0]
                if not nz.empty:
                    print(f"[EXEC] Datum (t+1) = {trade_date}")
                    for a, row in nz.iterrows():
                        fee = fees_df.loc[a]
                        p_ref_panel = float(panel_clean.xs(trade_date, level=0).loc[a, "adj_open_raw"])
                        _maybe_assert(abs(row["adj_open"] - p_ref_panel) < 1e-9,
                                      f"adj_open != adj_open(t+1) für {a}")

                        # Spread-Kontrolle: cs/2 pro Seite
                        cs = float(panel_clean.xs(trade_date, level=0).loc[a, "bid_ask_spread_corwin_schultz_raw"])
                        expected_spread = abs(row["q"]) * row["adj_open"] * (cs / 2.0)
                        _maybe_assert(abs(fee["spread_cost"] - expected_spread) < 1e-6,
                                      f"Spread-Abweichung für {a}")

                        per_share_spread = row["adj_open"] * (cs / 2.0)
                        if row["q"] > 0:
                            _maybe_assert(abs(row["p_exec"] - (row["adj_open"] + per_share_spread)) < 1e-6,
                                          f"p_exec (Kauf) passt nicht für {a}")
                        elif row["q"] < 0:
                            _maybe_assert(abs(row["p_exec"] - (row["adj_open"] - per_share_spread)) < 1e-6,
                                          f"p_exec (Verkauf) passt nicht für {a}")

                        print(f"   asset={a:>10s}  q={row['q']:.12f}  adj_open={row['adj_open']:.12f}  "
                              f"p_exec={row['p_exec']:.12f}  spread_cost={fee['spread_cost']:.12f}  "
                              f"fees={fee['fees']:.12f}  total_cost={fee['total_cost']:.12f}")

                    # --- Cash-Identität mit Toleranz (Rundungs-robust) ---
                    notional = float((exec_df["q"] * exec_df["p_exec"]).sum())
                    fees_total = float(fees_df["total_cost"].sum())

                    # cash_before aus der Recorder-Zeile VOR diesem Schritt
                    if getattr(env, "recorder", None) and getattr(env.recorder, "rows", None) and len(
                            env.recorder.rows) >= 2:
                        cash_before = float(env.recorder.rows[-2]["cash"])
                    else:
                        # Fallback (kommt praktisch nicht mehr vor)
                        cash_before = float(info.get("cash", float("nan"))) + notional + fees_total

                    # Risk-free-Aufzinsung zwischen t und t+1
                    rf_fac = float(rf_factor[t_before + 1])

                    cash_after_calc = cash_before - notional - fees_total
                    cash_after_calc *= rf_fac
                    cash_info = float(info.get("cash"))

                    delta_cash = abs(cash_after_calc - cash_info)
                    eps_cash   = max(0.05, 1e-8 * max(1.0, abs(cash_before)))
                    _maybe_assert(delta_cash <= eps_cash,
                                  f"Cash-Abweichung {delta_cash:.4f} > Toleranz {eps_cash:.4f}")
                    print(f"[CHECK] rf_fac={rf_fac:.10f} cash_before_eff={cash_before:.2f} "
                          f"notional={notional:.2f} fees_total={fees_total:.2f} "
                          f"cash_after_calc={cash_after_calc:.2f} cash_info={cash_info:.2f} Δ={delta_cash:.4f}")


                    # --- Turnover-Rekonstruktion mit Toleranz (pre vs post Norm) ---
                    if getattr(env, "recorder", None) and getattr(env.recorder, "rows", None) and len(
                            env.recorder.rows) >= 2:
                        w_prev = np.array(env.recorder.rows[-2]["w_post"], dtype=float)
                        w_post = np.array(env.recorder.rows[-1]["w_post"], dtype=float)
                        l1_exec = float(np.abs(w_post - w_prev).sum())
                        tv_info = float(info.get("turnover", np.nan))
                        delta_tv = abs(l1_exec - tv_info)
                        eps_tv = 1e-3  # sehr streng, aber nicht 1e-9
                        _maybe_assert(delta_tv <= eps_tv,
                                      f"Turnover-Abweichung {delta_tv:.8f} > Toleranz {eps_tv:.8f}")
                        print(f"[CHECK] l1_exec={l1_exec:.12f} info.turnover={tv_info:.12f} Δ={delta_tv:.8f}")

                    print(f"[EXEC] turnover_req={ctrl.get('l1_distance')}  "
                          f"turnover_exec={ctrl.get('l1_step')}  applied_scale={ctrl.get('applied_scale')}  "
                          f"acted={ctrl.get('acted')}")
                    if "exec_date" in info:
                        print("[EXEC] exec_date aus Env:", info["exec_date"])

                    did_exec_audit = True

        # 4c) Letzte Recorder-Zeile für CSV (ListRecorder in *_run.py) anreichern
        if getattr(env, "recorder", None) and getattr(env.recorder, "rows", None):
            row = env.recorder.rows[-1]
            row.update({
                "reward": float(reward),
                "value": float(info.get("value", float("nan"))),
                "fees": float(info.get("fees", 0.0)) if info.get("fees") is not None else 0.0,
                "turnover": float(info.get("turnover", float("nan"))),
                "turnover_req": float(info.get("turnover_req", np.nan)),
                "acted": bool((info.get("controls", {}) or {}).get("acted", False)),
            })

        print(f"STEP {k+1:02d}: reward={reward:.6f} "
              f"terminated={terminated} truncated={truncated} "
              f"fees={info.get('fees')} turnover={info.get('turnover')} value={info.get('value')}")

        if terminated or truncated:
            break

    # 5) Einfache Sanity-Checks
    assert obs["position"].shape[0] == A + 1, "Obs['position']-Länge != A+1."
    print("SANITY: OK")


if __name__ == "__main__":
    # Direkter Aufruf nur sinnvoll, wenn man unten selbst ein INJECT baut.
    # Üblich: main(INJECT) aus deiner *_run.py aufrufen.
    raise SystemExit("Bitte über *_run.py mit main(INJECT) starten.")
