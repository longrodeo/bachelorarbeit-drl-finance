# src/validation/check_cpcv_data.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd

@dataclass
class CheckResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warns:  List[str] = field(default_factory=list)

    def require(self, cond: bool, msg_ok: str, msg_fail: str, hard: bool = True):
        if cond:
            return True
        (self.errors if hard else self.warns).append(msg_fail)
        self.ok = self.ok and (not hard)  # hard Fehler → ok bleibt ggf. False
        return False

# ---------- Helpers für Index ----------
def to_daily_utc_index(df: pd.DataFrame, date_level: Optional[str] = None) -> pd.DataFrame:
    """Erzwinge UTC-tagesbasierten Index (00:00:00), sortiert & ohne Duplikate."""
    out = df.copy()
    if isinstance(out.index, pd.MultiIndex):
        # Datums-Level bestimmen
        lvl = date_level
        if lvl is None:
            for n in out.index.names:
                if n and n.lower() in {"date","datetime","timestamp","time"}:
                    lvl = n
                    break
            if lvl is None:
                lvl = out.index.names[0]
        # Datums-Level normalisieren
        dt = pd.to_datetime(out.index.get_level_values(lvl), utc=True).normalize()
        parts = [dt if n == lvl else out.index.get_level_values(n) for n in out.index.names]
        out.index = pd.MultiIndex.from_arrays(parts, names=out.index.names)
    else:
        out.index = pd.to_datetime(out.index, utc=True).normalize()

    out = out.sort_index()
    if out.index.has_duplicates:
        raise AssertionError("Duplikate im (normalisierten) Index.")
    return out

def normalize_datetime_index(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index(), None
    if isinstance(df.index, pd.MultiIndex):
        names = list(df.index.names)
        # a) schon datetime
        for i, name in enumerate(names):
            if isinstance(df.index.get_level_values(i), pd.DatetimeIndex):
                return df.sort_index(), name or names[i]
        # b) Level mit date-ähnlichem Namen parsen
        for i, name in enumerate(names):
            if name and name.lower() in {"date","datetime","timestamp","time"}:
                tmp = df.reset_index()
                tmp[name] = pd.to_datetime(tmp[name], utc=True, errors="coerce")
                tmp = tmp.set_index(names).sort_index()
                return tmp, name
        # c) Fallback: erste Ebene parsen
        first_name = names[0] or "date"
        tmp = df.reset_index()
        tmp[first_name] = pd.to_datetime(tmp[first_name], utc=True, errors="coerce")
        tmp = tmp.set_index(names).sort_index()
        if isinstance(tmp.index.get_level_values(0), pd.DatetimeIndex):
            return tmp, names[0]
        raise ValueError("MultiIndex ohne identifizierbare Datums-Ebene.")
    # Spaltenlayout
    for c in ("date","datetime","timestamp","time","Date","Timestamp"):
        if c in df.columns:
            out = df.copy()
            out[c] = pd.to_datetime(out[c], utc=True, errors="coerce")
            out = out.set_index(c).sort_index()
            return out, None
    raise ValueError("Keine Datums-Spalte gefunden.")

def slice_by_year(df: pd.DataFrame, date_level: Optional[str], year: int) -> pd.DataFrame:
    if isinstance(df.index, pd.MultiIndex):
        idx = df.index.get_level_values(date_level)  # type: ignore[arg-type]
        return df[idx.year == year]
    return df[df.index.year == year]

def check_datetime_index(df: pd.DataFrame, name: str) -> CheckResult:
    r = CheckResult(ok=True)
    tz = getattr(df.index, "tz", None)
    r.require(tz is not None, "", f"{name}: Index ist nicht tz-aware (UTC erwartet).")
    if tz is not None:
        r.require(str(tz) == "UTC", "", f"{name}: Index ist nicht UTC.")
    r.require(df.index.is_monotonic_increasing, "", f"{name}: Index nicht sortiert.")
    r.require(not df.index.has_duplicates, "", f"{name}: Duplikate im Index.")
    return r

# ---------- Kern-Checks ----------
def compare_year_splits_to_master(
    master: pd.DataFrame,
    years_dir: Path,
    *,
    label: str,
    strict_columns: bool = True,
    sample_rows: int = 5,
) -> CheckResult:
    res = CheckResult(ok=True)
    if not years_dir.is_dir():
        res.warns.append(f"{label}: Verzeichnis fehlt ({years_dir}).")
        return res

    yf = sorted(years_dir.glob("*.parquet"))
    res.require(len(yf) > 0, "", f"{label}: keine Year-Parquets gefunden.", hard=False)

    m_norm, m_lvl = normalize_datetime_index(master)
    m_norm = to_daily_utc_index(m_norm, m_lvl)

    for p in yf:
        year = int(p.stem)
        df = pd.read_parquet(p)
        try:
            df, lvl = normalize_datetime_index(df)
            df = to_daily_utc_index(df, lvl)
        except Exception as e:
            res.errors.append(f"{label}/{year}: Konnte Index nicht normalisieren: {e}")
            res.ok = False
            continue

        # Index-Checks
        r_idx = check_datetime_index(df, f"{label}/{year}")
        res.ok &= r_idx.ok
        res.errors += r_idx.errors; res.warns += r_idx.warns

        m_y = slice_by_year(m_norm, m_lvl, year)
        # Spalten
        cols_ok = (set(df.columns) == set(m_y.columns)) if strict_columns else set(df.columns).issubset(m_y.columns)
        res.require(cols_ok, "", f"{label}/{year}: Spalten stimmen nicht überein.")

        # Index-Gleichheit
        same_idx = df.index.equals(m_y.index)
        if not same_idx:
            only_file = df.index.difference(m_y.index)
            only_mast = m_y.index.difference(df.index)
            res.errors.append(f"{label}/{year}: Index unterscheidet sich "
                              f"(nur in Datei: {len(only_file)}, nur im Master: {len(only_mast)}).")
            res.ok = False

        # leichte Inhaltsprobe
        if sample_rows and len(df) and len(m_y):
            try:
                sample = df.index.to_series().sample(min(sample_rows, len(df)), random_state=7).sort_values()
                if not df.loc[sample].equals(m_y.loc[sample]):
                    res.warns.append(f"{label}/{year}: Stichproben-Inhalt weicht vom Master ab.")
            except Exception as e:
                res.warns.append(f"{label}/{year}: Stichprobe übersprungen ({e}).")

    return res
