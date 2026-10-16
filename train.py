"""Train a baseline and a LightGBM fraud model with a time-based split."""
import json, os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

df = pd.read_parquet("features.parquet")
FEATS = [c for c in df.columns if c.startswith(("tx_count", "amount_", "log_", "is_night", "hour_dev",
                                                "secs_", "user_hist", "new_", "device_tx", "device_age"))]
if os.environ.get("NO_DEVICE"):
    FEATS = [f for f in FEATS if not f.startswith("device_")]
cut = df["timestamp"].min() + pd.Timedelta(days=21)
train, test = df[df["timestamp"] < cut], df[df["timestamp"] >= cut]
Xtr, ytr, Xte, yte = train[FEATS], train["is_fraud"], test[FEATS], test["is_fraud"]
print(f"Train: {len(train):,} txns ({ytr.sum()} fraud) | Test: {len(test):,} txns ({yte.sum()} fraud)")
print(f"Features ({len(FEATS)}): {FEATS}\n")


def report(name, p):
    """Fraud caught when only 1% of genuine payments get a false alarm."""
    thr = float(np.quantile(p[yte.values == 0], 0.99))
    flagged = p >= thr
    recall = flagged[yte.values == 1].mean()
    precision = yte.values[flagged].mean()
    print(f"{name:10s} | PR-AUC {average_precision_score(yte, p):.3f} | ROC-AUC {roc_auc_score(yte, p):.3f}"
          f" | caught @1% false alarms: {recall:.1%} | precision: {precision:.1%}")
    for t in ["account_takeover", "card_testing", "mule_scam"]:
        m = (test["fraud_type"] == t).values
        print(f"{'':10s}   {t:17s} caught: {flagged[m].mean():.1%}")
    return thr


os.makedirs("models", exist_ok=True)
base = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
base.fit(Xtr, ytr)
report("Baseline", base.predict_proba(Xte)[:, 1])
print()

model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42,
                           scale_pos_weight=(ytr == 0).sum() / ytr.sum(), verbose=-1)
model.fit(Xtr, ytr)
thr = report("LightGBM", model.predict_proba(Xte)[:, 1])
model.booster_.save_model("models/lgbm.txt")
json.dump({"features": FEATS, "threshold": thr}, open("models/meta.json", "w"), indent=2)
imp = pd.Series(model.feature_importances_, index=FEATS).sort_values(ascending=False)
print("\nTop features:\n" + imp.head(8).to_string())
