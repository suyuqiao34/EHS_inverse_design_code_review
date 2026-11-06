"""Composition-weighted baselines on the Round-0 (M0) datasets, 100 random 80/20 splits."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
TASKS = {
    "Tm": ("eutectic_data_tm.csv", "Tm (°C)"),
    "dH": ("eutectic_data_dh.csv", "ΔHm (J/g)"),
}

for task, (fname, prop) in TASKS.items():
    df = pd.read_csv(ROOT / fname)
    target = [c for c in df.columns if ("Melting" in c or "Enthalpy" in c)]
    ycol = target[0]
    w = df[["ID1", "ID2", "ID3"]].apply(pd.to_numeric, errors="coerce").fillna(0).values
    props = df[[f"{k}-{prop}" for k in (1, 2, 3)]].apply(pd.to_numeric, errors="coerce").fillna(0).values
    x = (w * props).sum(axis=1)
    y = pd.to_numeric(df[ycol], errors="coerce").values
    ok = ~np.isnan(y)
    x, y = x[ok], y[ok]
    rows = []
    for seed in range(100):
        xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.2, random_state=seed)
        lr = LinearRegression().fit(xtr.reshape(-1, 1), ytr)
        p = lr.predict(xte.reshape(-1, 1))
        rows.append({
            "direct_R2": r2_score(yte, xte), "direct_RMSE": mean_squared_error(yte, xte) ** 0.5,
            "direct_MAE": mean_absolute_error(yte, xte),
            "lin_R2": r2_score(yte, p), "lin_RMSE": mean_squared_error(yte, p) ** 0.5,
            "lin_MAE": mean_absolute_error(yte, p),
        })
    r = pd.DataFrame(rows)
    print(task, ycol, "n =", len(y), "| full-data direct R2 %.3f" % r2_score(y, x))
    print(r.agg(["mean", "std"]).round(3).to_string())
