"""Training-serving skew check: online features must equal offline (training) features."""
import time
import numpy as np
import pandas as pd
from online_features import FeatureStore

off = pd.read_parquet("features.parquet").set_index("txn_id")
raw = pd.read_parquet("transactions.parquet").sort_values("timestamp")
store, rows, t0 = FeatureStore(), [], time.time()
for r in raw.itertuples(index=False):
    f = store.compute(r.user_id, r.timestamp.timestamp(), r.amount, r.device_id, r.city,
                      r.merchant_id, r.timestamp.hour)
    f["txn_id"] = r.txn_id
    rows.append(f)
on = pd.DataFrame(rows).set_index("txn_id").loc[off.index]
secs = time.time() - t0
print(f"Replayed {len(on):,} payments in {secs:.1f}s ({len(on)/secs:,.0f} payments/sec)\n")
worst = 100.0
for c in on.columns:
    match = np.isclose(on[c].values.astype(float), off[c].values.astype(float), rtol=1e-6, atol=1e-6).mean() * 100
    worst = min(worst, match)
    print(f"{c:18s} {match:7.3f}% match")
print(f"\nWORST feature match: {worst:.3f}%  ->  {'PASS ✅' if worst > 99.9 else 'CHECK ❌'}")
