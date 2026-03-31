from __future__ import annotations

import pandas as pd


WINTER_MONTHS = [11, 12, 1, 2, 3]


def filter_active_winter_rmm(rmm: pd.DataFrame, amplitude_threshold: float = 1.0) -> pd.DataFrame:
    out = rmm.copy()
    out = out[out["amplitude"] > amplitude_threshold].copy()
    out["month"] = out["date"].dt.month
    out = out[out["month"].isin(WINTER_MONTHS)].copy()
    out["year_month"] = out["date"].dt.to_period("M")
    return out


def monthly_mjo_summary(rmm: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        rmm.resample("MS", on="date")
        .agg({
            "RMM1": "mean",
            "RMM2": "mean",
            "phase": lambda x: x.mode().iloc[0] if len(x.mode()) else pd.NA,
            "amplitude": "mean",
        })
        .reset_index()
    )
    monthly["ym"] = monthly["date"].dt.to_period("M")
    return monthly


def compute_daily_anomalies(t_daily: pd.DataFrame) -> pd.DataFrame:
    out = t_daily.copy()
    out["month"] = out["valid_time"].dt.month
    out = out[out["month"].isin(WINTER_MONTHS)].copy()

    out["doy"] = out["valid_time"].dt.dayofyear
    clim = out.groupby("doy")["temp_C"].mean()

    out["temp_anom"] = out["doy"].map(clim)
    out["temp_anom"] = out["temp_C"] - out["temp_anom"]
    return out


def merge_temp_with_rmm(t_daily: pd.DataFrame, rmm: pd.DataFrame) -> pd.DataFrame:
    left = t_daily.copy()
    right = rmm.copy()
    left["date"] = pd.to_datetime(left["date"])
    right["date"] = pd.to_datetime(right["date"])
    return pd.merge(left, right, on="date", how="inner")


def add_autoregressive_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("date").reset_index(drop=True).copy()
    out["AR1"] = out["temp_anom"].shift(1)
    out["MA7"] = out["temp_anom"].rolling(7).mean().shift(1)
    return out


def build_lead_frame(df: pd.DataFrame, max_lead: int = 20) -> pd.DataFrame:
    base = df.sort_values("date").reset_index(drop=True).copy()
    frames = []

    for lead in range(max_lead + 1):
        tmp = base.copy()
        tmp["lead"] = lead
        tmp["init_date"] = tmp["date"]
        tmp["target_date"] = tmp["date"].shift(-lead)
        tmp["y"] = tmp["temp_anom"].shift(-lead)
        frames.append(tmp)

    out = pd.concat(frames, ignore_index=True)
    return out