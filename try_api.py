import requests
import pandas as pd

df = pd.read_parquet("transactions.parquet")
u = df[(df.user_id == "U00683") & (df.timestamp < "2026-09-22")].iloc[-1]   # user's last known payment
avg = df[df.user_id == "U00683"].amount.median()

normal = {"user_id": u.user_id, "timestamp": "2026-09-23 14:05:00", "amount": round(avg, 2),
          "device_id": u.device_id, "city": u.city, "merchant_id": u.merchant_id}
attack = {"user_id": u.user_id, "timestamp": "2026-09-24 03:12:00", "amount": round(avg * 9, 2),
          "device_id": "STOLEN-PHONE-1", "city": "Kolkata" if u.city != "Kolkata" else "Delhi",
          "merchant_id": "MULE007"}

for name, pay in [("NORMAL payment", normal), ("STOLEN-PHONE payment", attack)]:
    r = requests.post("http://127.0.0.1:8000/score", json=pay).json()
    print(f"{name}: Rs {pay['amount']:,} at {pay['timestamp']}")
    print(f"  -> {r['decision']}  (fraud probability {r['fraud_probability']}, {r['latency_ms']} ms)")
    print(f"  -> reasons: {r['reasons']}\n")
print("Latency stats:", requests.get("http://127.0.0.1:8000/stats").json())
