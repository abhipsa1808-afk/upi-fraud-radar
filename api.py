"""Real-time UPI fraud scoring API."""
import json, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from fastapi import FastAPI
from pydantic import BaseModel
from online_features import FeatureStore

meta = json.load(open("models/meta.json"))
FEATS, THR = meta["features"], meta["threshold"]
booster = lgb.Booster(model_file="models/lgbm.txt")
store = FeatureStore()
latencies = []
app = FastAPI(title="UPI Fraud Radar", version="1.0")


def describe(name, v):
    if name == "amount_ratio": return f"amount is {v:.1f}x this user's normal"
    if name.startswith("tx_count"): return f"{int(v)} payments in last {name.split('_')[-1]}"
    if name.startswith("amount_sum"): return f"Rs {v:,.0f} sent in last {name.split('_')[-1]}"
    if name == "is_night": return "payment at night (before 6 am)" if v else "daytime payment"
    if name == "device_tx_count": return f"phone used only {int(v)} times before"
    if name == "device_age_secs": return f"phone first seen {v/60:,.0f} min ago"
    if name == "secs_since_last": return f"{v:,.0f}s since previous payment"
    if name == "hour_dev": return f"{v:.0f}h away from usual payment time"
    if name.startswith("new_"): return name.replace("_", " ") if v else f"familiar {name[4:]}"
    return f"{name} = {v:,.2f}"


class Payment(BaseModel):
    user_id: str
    timestamp: str          # e.g. "2026-09-25 02:14:00"
    amount: float
    device_id: str
    city: str
    merchant_id: str


@app.on_event("startup")
def warm_up():
    """Replay the first 21 days so the store knows every user's normal behaviour."""
    df = pd.read_parquet("transactions.parquet").sort_values("timestamp")
    hist = df[df["timestamp"] < df["timestamp"].min() + pd.Timedelta(days=21)]
    t0 = time.time()
    for r in hist.itertuples(index=False):
        store.compute(r.user_id, r.timestamp.timestamp(), r.amount, r.device_id, r.city,
                      r.merchant_id, r.timestamp.hour)
    print(f"Warm-up: {len(hist):,} past payments loaded in {time.time() - t0:.1f}s")


@app.post("/score")
def score(p: Payment):
    t0 = time.perf_counter()
    ts = pd.Timestamp(p.timestamp)
    f = store.compute(p.user_id, ts.timestamp(), p.amount, p.device_id, p.city, p.merchant_id, ts.hour)
    x = np.array([[f[c] for c in FEATS]], dtype=float)
    prob = float(booster.predict(x)[0])
    contrib = booster.predict(x, pred_contrib=True)[0][:-1]      # how much each feature pushed the score
    top = sorted(zip(FEATS, contrib), key=lambda z: -z[1])[:3]
    reasons = [describe(n, f[n]) for n, c in top if c > 0]
    ms = (time.perf_counter() - t0) * 1000
    latencies.append(ms)
    return {"decision": "BLOCK" if prob >= THR else "ALLOW", "fraud_probability": round(prob, 4),
            "reasons": reasons, "latency_ms": round(ms, 2)}


@app.get("/stats")
def stats():
    lat = np.array(latencies) if latencies else np.array([0.0])
    return {"payments_scored": len(latencies), "p50_ms": round(float(np.percentile(lat, 50)), 2),
            "p99_ms": round(float(np.percentile(lat, 99)), 2)}
