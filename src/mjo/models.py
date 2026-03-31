from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PREDICTOR_COLS = ["RMM1", "RMM2", "amplitude", "AR1", "MA7"]


def fit_speed_subset(df_sub: pd.DataFrame, lead: int):
    dfL = df_sub[df_sub["lead"] == lead].copy()

    if "y" not in dfL.columns:
        dfL["y"] = dfL["temp_anom"].shift(-lead)

    dfL = dfL.dropna(subset=["y", *PREDICTOR_COLS])

    X = dfL[PREDICTOR_COLS].values
    y = dfL["y"].values

    model = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 25))),
    ])

    model.fit(X, y)
    return model


def fit_all_leads(df_leads: pd.DataFrame, max_lead: int = 20) -> dict[int, Pipeline]:
    models = {}
    for lead in range(max_lead + 1):
        models[lead] = fit_speed_subset(df_leads, lead)
    return models


def evaluate_by_lead(df_leads: pd.DataFrame, max_lead: int = 20):
    results = []

    for lead in range(max_lead + 1):
        dfL = df_leads[df_leads["lead"] == lead].copy()

        if "y" not in dfL.columns:
            dfL["y"] = dfL["temp_anom"].shift(-lead)

        dfL = dfL.dropna(subset=["y", *PREDICTOR_COLS])

        if len(dfL) < 30:
            continue

        X = dfL[PREDICTOR_COLS].values
        y = dfL["y"].values

        model = Pipeline([
            ("scale", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 25))),
        ])
        model.fit(X, y)

        score = model.score(X, y)
        results.append((lead, score))

    return results


