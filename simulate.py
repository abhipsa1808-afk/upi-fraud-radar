"""Generate realistic synthetic UPI transactions with hidden fraud patterns."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N_USERS, N_MERCHANTS, DAYS = 5000, 500, 30
START = pd.Timestamp("2026-09-01")
CITIES = ["Jaipur", "Mumbai", "Delhi", "Bengaluru", "Kolkata", "Pune", "Hyderabad", "Chennai"]

users = pd.DataFrame({
    "user_id": [f"U{i:05d}" for i in range(N_USERS)],
    "city": rng.choice(CITIES, N_USERS),
    "avg_amount": np.round(rng.lognormal(6.0, 0.8, N_USERS), 2),   # typical spend ~₹400
    "tx_per_day": rng.gamma(2.0, 1.0, N_USERS),                      # how active they are
    "active_hour": np.where(rng.random(N_USERS) < 0.1, rng.integers(0, 5, N_USERS), rng.integers(8, 22, N_USERS)),                     # when they usually pay
})
users["device_id"] = [f"D{i:05d}" for i in range(N_USERS)]           # each user's own phone
merchants = [f"M{i:04d}" for i in range(N_MERCHANTS)]
MULES = [f"MULE{i:03d}" for i in range(15)]                          # fraudster accounts


# ---------- normal transactions ----------
counts = rng.poisson(users["tx_per_day"].values * DAYS)
uid = np.repeat(np.arange(N_USERS), counts)
n = len(uid)
day = rng.integers(0, DAYS, n)
hour = np.clip(np.round(rng.normal(users["active_hour"].values[uid], 3)), 0, 23)
sec = rng.integers(0, 3600, n)
fav = rng.integers(0, N_MERCHANTS, (N_USERS, 8))                     # 8 favourite shops each
use_fav = rng.random(n) < 0.85
merch_idx = np.where(use_fav, fav[uid, rng.integers(0, 8, n)], rng.integers(0, N_MERCHANTS, n))

normal = pd.DataFrame({
    "timestamp": START + pd.to_timedelta(day, "D") + pd.to_timedelta(hour, "h") + pd.to_timedelta(sec, "s"),
    "user_id": users["user_id"].values[uid],
    "merchant_id": np.array(merchants)[merch_idx],
    "amount": np.round(users["avg_amount"].values[uid] * rng.lognormal(0, 0.5, n), 2),
    "device_id": users["device_id"].values[uid],
    "city": users["city"].values[uid],
    "is_fraud": 0,
    "fraud_type": "none",
})


# ---------- realistic noise: genuine users who LOOK suspicious ----------
m = rng.random(n) < 0.004                               # big genuine buys (rent, phone, gold)
normal.loc[m, "amount"] = (normal.loc[m, "amount"] * rng.uniform(5, 20, m.sum())).round(2)
m = rng.random(n) < 0.02                                # travelling to another city
normal.loc[m, "city"] = rng.choice(CITIES, m.sum())
switchers = rng.random(N_USERS) < 0.05                  # 5% of users get a new phone mid-month
switch_day = rng.integers(1, DAYS, N_USERS)
m = switchers[uid] & (day >= switch_day[uid])
normal.loc[m, "device_id"] = "N" + normal.loc[m, "device_id"]


# ---------- fraud patterns ----------
rows = []


def burst(u, kind, k, minutes, amount_fn, new_device, to_mule, night):
    hour = int(rng.integers(1, 5)) if night else int(u.active_hour)
    t0 = START + pd.Timedelta(days=int(rng.integers(0, DAYS)), hours=hour,
                              minutes=int(rng.integers(0, 60)))
    dev = f"DX{rng.integers(0, 10**6):06d}" if new_device else u.device_id
    city = rng.choice([c for c in CITIES if c != u.city]) if new_device else u.city
    for _ in range(int(k)):
        merch = rng.choice(MULES) if to_mule else rng.choice(merchants)
        rows.append({
            "timestamp": t0 + pd.Timedelta(seconds=int(rng.integers(0, minutes * 60))),
            "user_id": u.user_id, "merchant_id": merch,
            "amount": round(float(amount_fn(u)), 2),
            "device_id": dev, "city": city, "is_fraud": 1, "fraud_type": kind,
        })


victims = list(users.sample(500, random_state=1).itertuples())
for u in victims[:250]:
    burst(u, "account_takeover", rng.integers(3, 7), 30,
          lambda u: u.avg_amount * rng.uniform(1.5, 8), True, True, rng.random() < 0.5)
for u in victims[250:400]:
    burst(u, "card_testing", rng.integers(3, 9), 45,
          lambda u: rng.uniform(1, 150), True, False, False)
for u in victims[400:]:
    burst(u, "mule_scam", 1, 1,
          lambda u: u.avg_amount * rng.uniform(2, 12), False, True, False)


# ---------- combine & save ----------
df = pd.concat([normal, pd.DataFrame(rows)], ignore_index=True)
df = df.sort_values("timestamp").reset_index(drop=True)
df.insert(0, "txn_id", [f"T{i:07d}" for i in range(len(df))])
df.to_parquet("transactions.parquet", index=False)

print(f"Transactions: {len(df):,}  |  users: {N_USERS:,}  |  days: {DAYS}")
print(f"Fraud rate: {df.is_fraud.mean():.2%}")
print(df.fraud_type.value_counts().to_string(), "\n")
print(df.sample(5, random_state=0)[["timestamp", "user_id", "merchant_id", "amount", "fraud_type"]].to_string(index=False))
