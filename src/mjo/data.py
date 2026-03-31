from __future__ import annotations

from datetime import datetime
import pandas as pd


def load_bom_rmm(filepath: str) -> pd.DataFrame:
    rows = []

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line or not line[0].isdigit():
                continue

            parts = line.split()
            if len(parts) < 7:
                continue

            try:
                y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
                rmm1 = float(parts[3])
                rmm2 = float(parts[4])
                phase = int(float(parts[5]))
                amp = float(parts[6])

                if amp == 999 or abs(amp) >= 1e35 or abs(rmm1) >= 1e35 or abs(rmm2) >= 1e35:
                    continue

                rows.append([datetime(y, m, d), rmm1, rmm2, phase, amp])
            except Exception:
                continue

    df = pd.DataFrame(
        rows, columns=["date", "RMM1", "RMM2", "phase", "amplitude"]
    ).sort_values("date").reset_index(drop=True)

    return df


def load_city_temperature(filepath: str) -> pd.DataFrame:
    t = pd.read_csv(filepath)

    t["valid_time"] = pd.to_datetime(t["valid_time"])
    t["temp_C"] = t["t2m"] - 273.15

    daily = (
        t.set_index("valid_time")["temp_C"]
        .resample("D")
        .mean()
        .reset_index()
    )

    daily["date"] = daily["valid_time"]
    return daily