# 🛡️ UPI Fraud Radar

**A real-time fraud detection system for UPI payments: synthetic data → feature engineering → LightGBM → online feature store → sub-millisecond scoring API → live dashboard.**

🔗 **Live demo:** _coming soon_

---

## TL;DR

| | |
|---|---|
| Payments simulated | **301,811** (5,000 users, 30 days, ~0.7% fraud) |
| Fraud caught (live stream, last 9 days) | **98.6%** at a **1% false-alarm rate** |
| Scoring latency | **p50 0.48 ms · p99 0.53 ms** |
| Throughput | **~1,000 payments/sec** through the HTTP API (1 client, laptop) · **~170,000/sec** inside the feature store |
| Training–serving skew | **99.998%** of online features identical to training features ✅ |
| Explainability | every BLOCK comes with the top 3 reasons (LightGBM SHAP-style contributions) |

---

## Why this project

Payment companies (Google Pay, PhonePe, Paytm, Amazon Pay, Visa) have to decide **in milliseconds** whether a payment is fraud, using only what they know about the user **so far**. That makes fraud detection a *systems* problem, not just a modelling problem:

1. **Features must use only the past** (no data leakage)
2. **Fraud is rare** (<1%), so accuracy is meaningless
3. **The features computed live must match the features the model was trained on** (training–serving skew)
4. **Every decision needs a reason** a human analyst can check

This project builds all four.

---

## Architecture

```
simulate.py ──► transactions.parquet ──► features.py ──► features.parquet ──► train.py ──► models/lgbm.txt
                        │                (batch, pandas)                                         │
                        │                                                                        ▼
                        └── replay ──► online_features.py  ◄──── parity_test.py (must match) ──► api.py  (FastAPI /score)
                                       (per-user state, O(1))                                    │
                                                                                                 ▼
                                                                                          dashboard.py (Streamlit)
```

---

## 1. Realistic synthetic data (`simulate.py`)

Real fraud data is private, so I simulated **5,000 users × 30 days** with per-user habits (home city, phone, typical amount, active hours, favourite merchants) and injected three real UPI scam patterns:

| Fraud type | What it looks like |
|---|---|
| **Account takeover** | stolen phone/SIM → new device, another city, burst of larger payments to fraudster ("mule") accounts, often at night |
| **Card testing** | a fraudster checks stolen credentials with a quick series of small payments |
| **Mule scam** | the victim is tricked ("pay a fee to claim your prize") into one large payment **from their own phone**: the hardest case |

### The "100% accuracy" red flag 🚩

My first version of the data gave **ROC-AUC 1.000 and 100% recall for every fraud type**. Instead of celebrating, I treated it as a bug: fraudsters were too obvious, and *no genuine user ever looked suspicious*. I added realistic noise:

- 10% of users are night owls
- 5% of users switch to a new phone mid-month
- 2% of genuine payments happen while travelling
- 0.4% of genuine payments are big one-off purchases (rent, phone, gold)
- fraudsters use smaller amounts, only half of takeovers happen at night, card testing is slower

After this, scores dropped to realistic levels (and the baseline model's weaknesses became visible).

---

## 2. Features using only the past (`features.py`)

17 features per payment, all computed from the **user's own history up to that moment**:

| Group | Features |
|---|---|
| Velocity | payments and amount sent in the last **10 min / 1 h / 24 h** |
| Behaviour | amount vs the user's average (`amount_ratio`), distance from usual hour, seconds since last payment, history length |
| Novelty | new device / new city / new merchant |
| Device trust (v2) | `device_tx_count` (times this phone was used before), `device_age_secs` (when it was first seen) |

The sliding-window features are fully vectorised: rows are sorted by `(user, time)` and encoded into a single sortable key `user_index × 10⁸ + seconds`, so one `np.searchsorted` call finds every window start for all 300k rows at once.

**Train/test split is by time** (days 1–21 train, days 22–30 test), never random, so the model can't peek at the future.

---

## 3. Models (`train.py`)

Metric: **fraud caught when only 1% of genuine payments get a false alarm** (plus PR-AUC), not accuracy, since predicting "never fraud" would already be 99.3% accurate.

| Model | PR-AUC | Fraud caught @1% FPR | Account takeover | Card testing | **Mule scam** |
|---|---|---|---|---|---|
| Logistic regression (baseline) | 0.576 | 93.8% | 99.1% | 95.5% | **22.6%** |
| LightGBM v1 (no device-trust features) | 0.971 | 99.0% | 99.7% | 99.6% | 87.1% |
| **LightGBM v2** (with device-trust features) | **0.981** | 98.6% | 99.1% | 99.3% | **87.1%** |

**Findings**
- A linear model can't combine signals: it catches only **23%** of mule scams. LightGBM catches **87%** by learning interactions such as *"huge amount × familiar phone × brand-new merchant"*.
- **Device-trust features fix a real design flaw.** `new_device` is only 1 for the *first* payment on a new phone, so the 2nd–5th payments of a takeover burst looked "familiar". `device_tx_count` / `device_age_secs` stay suspicious for the whole burst; `device_age_secs` became the **#4 most important feature** and PR-AUC rose 0.971 → 0.981.
- **Mule scams remain the hardest** (87%): the victim uses their own phone, city and normal hours, so only the amount and the unknown receiver stand out.

**Honest trade-off:** at a 1% false-alarm rate, precision is ~41%, because 1% of ~90k genuine payments is more alerts than there are frauds. A real system would use this score to route payments: auto-block only very high scores, and send the middle band to step-up checks (OTP / UPI PIN re-entry) instead of a hard block.

---

## 4. Online feature store + parity test (`online_features.py`, `parity_test.py`)

In production you can't run pandas over all history for every payment. `FeatureStore` keeps a small state per user (24-hour deque of recent payments, running sums, seen devices/cities/merchants) and updates it in O(1) per payment.

To prove there's **no training–serving skew**, `parity_test.py` replays all 301,811 payments through the online store and compares every feature with the training features:

```
Replayed 301,811 payments in 1.8s (169,815 payments/sec)
WORST feature match: 99.998%  ->  PASS ✅
```

(The 0.002% are payments with identical timestamps, where sort order differs.)

---

## 5. Real-time API (`api.py`)

`POST /score` takes one payment, updates the user's state, scores it with LightGBM, and explains the decision using LightGBM's per-feature contributions (`pred_contrib=True`):

```json
{
  "decision": "BLOCK",
  "fraud_probability": 1.0,
  "reasons": ["phone first seen 0 min ago", "new device", "amount is 8.6x this user's normal"],
  "latency_ms": 0.63
}
```

On startup it warms up the store by replaying 21 days of history (211,217 payments in **0.9 s**).

### Load test (`load_test.py`)

Streamed the **entire last 9 days (90,594 payments)** through the live HTTP API in time order:

```
Streamed 90,594 payments in 89s -> 1,013 payments/sec (1 client)
Server latency: p50 0.48 ms | p99 0.53 ms
Fraud caught: 98.6% of 626 | false alarms: 1.00% of genuine
```

These live numbers **exactly match** the offline evaluation (98.6% / 99.1% / 99.3% / 87.1%), end-to-end confirmation that serving behaves like training.

---

## 6. Live dashboard (`dashboard.py`)

- **▶ Stream next batch**: scores real payments live and updates fraud caught, false alarms and p99 latency
- **🚨 Latest blocked payments**: each with its reasons, marked ✅ real fraud or ⚠️ false alarm
- **🧪 Try your own payment**: pick a user, change amount / hour / city, use a new phone or pay an unknown account, and see the decision and reasons instantly

The dashboard imports the **same scoring code** as the API, so the two can never disagree.

---

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas pyarrow scikit-learn lightgbm fastapi uvicorn streamlit requests
# macOS only: brew install libomp

python simulate.py        # 301k payments
python features.py        # 17 leakage-free features
python train.py           # baseline vs LightGBM
python parity_test.py     # online == offline?
uvicorn api:app --port 8000          # real-time API
python load_test.py                  # (in another terminal) stream 90k payments
streamlit run dashboard.py           # live dashboard
```

## Project structure

```
simulate.py          synthetic users, merchants, payments + 3 fraud patterns + realistic noise
features.py          batch feature engineering (past-only, vectorised windows)
train.py             time-based split, logistic regression vs LightGBM, fraud-specific metrics
online_features.py   per-user online feature store (O(1) updates)
parity_test.py       training-serving skew check
api.py               FastAPI scoring service with reasons + latency stats
load_test.py         streams 90k payments through the API
dashboard.py         Streamlit live monitoring + "try your own payment"
models/              trained LightGBM model + feature list + threshold
results_*.txt        raw outputs of every experiment
```

---

Built by **Abhipsa Rout**, B.Tech CSE, Manipal University
