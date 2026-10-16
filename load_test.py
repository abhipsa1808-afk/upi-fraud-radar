"""Stream the last 9 days of payments through the live API, in time order."""
import time
import numpy as np
import pandas as pd
import requests

df = pd.read_parquet("transactions.parquet").sort_values("timestamp")
stream = df[df["timestamp"] >= df["timestamp"].min() + pd.Timedelta(days=21)]
s, blocked, t0 = requests.Session(), [], time.time()
for i, r in enumerate(stream.itertuples(index=False), 1):
    pay = {"user_id": r.user_id, "timestamp": str(r.timestamp), "amount": r.amount,
           "device_id": r.device_id, "city": r.city, "merchant_id": r.merchant_id}
    blocked.append(s.post("http://127.0.0.1:8000/score", json=pay).json()["decision"] == "BLOCK")
    if i % 10000 == 0:
        print(f"  {i:,} payments streamed...")
secs = time.time() - t0
blocked, y = np.array(blocked), stream["is_fraud"].values == 1
print(f"\nStreamed {len(stream):,} payments in {secs:.0f}s  ->  {len(stream)/secs:,.0f} payments/sec (1 client)")
print("Server latency:", s.get("http://127.0.0.1:8000/stats").json())
print(f"Fraud caught: {blocked[y].mean():.1%} of {y.sum()} | false alarms: {blocked[~y].mean():.2%} of genuine")
for t in ["account_takeover", "card_testing", "mule_scam"]:
    m = (stream["fraud_type"] == t).values
    print(f"  {t:17s} caught: {blocked[m].mean():.1%}")
