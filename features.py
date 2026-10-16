"""Turn raw transactions into fraud features, using ONLY each user's past (no leakage)."""
import numpy as np
import pandas as pd

df = pd.read_parquet("transactions.parquet")
df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

# One sortable key per row: user number * big gap + seconds since start.
# This lets us find "the user's transactions in the last X seconds" for ALL rows at once.
code = pd.factorize(df["user_id"])[0].astype("int64")
secs = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds().values.astype("int64")
key = code * 10**8 + secs
idx = np.arange(len(df))
cs = np.concatenate([[0.0], np.cumsum(df["amount"].values)])

for name, w in [("10m", 600), ("1h", 3600), ("24h", 86400)]:
    start = np.searchsorted(key, key - w)          # first txn of this user inside the window
    df[f"tx_count_{name}"] = idx - start + 1
    df[f"amount_sum_{name}"] = np.round(cs[idx + 1] - cs[start], 2)


df["hour"] = df["timestamp"].dt.hour
df["is_night"] = (df["hour"] < 6).astype(int)
df["log_amount"] = np.log1p(df["amount"])
g = df.groupby("user_id")
prev_n = g.cumcount()                                    # how many past txns this user has
safe_n = prev_n.replace(0, np.nan)
prev_mean = (g["amount"].cumsum() - df["amount"]) / safe_n
df["amount_ratio"] = (df["amount"] / prev_mean).fillna(1.0)          # e.g. 12 = 12x their normal
df["hour_dev"] = (df["hour"] - (g["hour"].cumsum() - df["hour"]) / safe_n).abs().fillna(0)
df["secs_since_last"] = g["timestamp"].diff().dt.total_seconds().fillna(1e7)
df["user_history"] = prev_n
for col in ["device_id", "city", "merchant_id"]:
    first = df.groupby(["user_id", col]).cumcount() == 0
    df[f"new_{col.replace('_id', '')}"] = (first & (prev_n > 0)).astype(int)

dg = df.groupby(["user_id", "device_id"])
df["device_tx_count"] = dg.cumcount()
df["device_age_secs"] = (df["timestamp"] - dg["timestamp"].transform("min")).dt.total_seconds()

df = df.sort_values("timestamp").reset_index(drop=True)
df.to_parquet("features.parquet", index=False)
FEATS = [c for c in df.columns if c.startswith(("tx_count", "amount_", "log_", "is_night", "hour_dev",
                                                "secs_", "user_hist", "new_", "device_tx", "device_age"))]
print(f"Built {len(FEATS)} features for {len(df):,} transactions\n")
print(df.groupby("fraud_type")[["amount_ratio", "tx_count_10m", "new_device", "is_night"]].mean().round(2))
