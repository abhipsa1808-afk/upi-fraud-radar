"""Live fraud-monitoring dashboard for UPI Fraud Radar."""
import os, subprocess, sys
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="UPI Fraud Radar", page_icon="🛡️", layout="wide")


@st.cache_resource
def boot():
    if not os.path.exists("transactions.parquet"):          # e.g. on a fresh cloud server
        subprocess.run([sys.executable, "simulate.py"], check=True)
    import api                                               # same code as the live API = no skew
    api.warm_up()
    df = pd.read_parquet("transactions.parquet").sort_values("timestamp")
    stream = df[df["timestamp"] >= df["timestamp"].min() + pd.Timedelta(days=21)].reset_index(drop=True)
    return api, stream, {"pos": 0, "log": []}


with st.spinner("Booting fraud engine: loading 21 days of payment history..."):
    api, stream, S = boot()

st.title("🛡️ UPI Fraud Radar")
st.caption("Real-time fraud detection for UPI payments · LightGBM + online feature store · by Abhipsa Rout")

c1, c2 = st.columns([1, 3])
n = c1.selectbox("Payments per batch", [500, 2000, 5000], index=1)
if c1.button("▶ Stream next batch", type="primary", disabled=S["pos"] >= len(stream)):
    batch = stream.iloc[S["pos"]:S["pos"] + n]
    for r in batch.itertuples(index=False):
        out = api.score(api.Payment(user_id=r.user_id, timestamp=str(r.timestamp), amount=r.amount,
                                    device_id=r.device_id, city=r.city, merchant_id=r.merchant_id))
        S["log"].append({"time": r.timestamp, "user": r.user_id, "amount": r.amount, "city": r.city,
                         "decision": out["decision"], "fraud_prob": out["fraud_probability"],
                         "reasons": ", ".join(out["reasons"]), "truth": r.fraud_type})
    S["pos"] += len(batch)
c2.progress(S["pos"] / len(stream), text=f"{S['pos']:,} / {len(stream):,} payments streamed (last 9 days)")

log = pd.DataFrame(S["log"])
if len(log):
    fraud = log["truth"] != "none"
    blocked = log["decision"] == "BLOCK"
    lat = np.array(api.latencies) if api.latencies else np.array([0.0])
    m = st.columns(5)
    m[0].metric("Payments scored", f"{len(log):,}")
    m[1].metric("Blocked", f"{blocked.sum():,}")
    m[2].metric("Fraud caught", f"{blocked[fraud].mean():.1%}" if fraud.any() else "–")
    m[3].metric("False alarms", f"{blocked[~fraud].mean():.2%}")
    m[4].metric("p99 latency", f"{np.percentile(lat, 99):.2f} ms")

    st.subheader("🚨 Latest blocked payments")
    show = log[blocked].tail(15).iloc[::-1].copy()
    show["verdict"] = np.where(show["truth"] != "none", "✅ real fraud (" + show["truth"] + ")", "⚠️ false alarm")
    st.dataframe(show[["time", "user", "amount", "city", "fraud_prob", "reasons", "verdict"]],
                 hide_index=True, use_container_width=True)
else:
    st.info("Click **▶ Stream next batch** to start scoring live payments.")

with st.sidebar:
    st.header("🧪 Try your own payment")
    uid = st.selectbox("User", sorted(api.store.users)[:200])
    s = api.store.users[uid]
    normal_amt = s.amt_sum / s.n if s.n else 500.0
    home_dev = max(s.devices, key=lambda d: s.devices[d][0]) if s.devices else "D00000"
    home_city = sorted(s.cities)[0] if s.cities else "Jaipur"
    st.caption(f"This user usually spends ≈ Rs {normal_amt:,.0f} · {s.n} past payments · {home_city}")
    amount = st.number_input("Amount (Rs)", 1.0, 1_000_000.0, float(round(normal_amt, 2)))
    hour = st.slider("Hour of day", 0, 23, 14)
    cities = ["Jaipur", "Mumbai", "Delhi", "Bengaluru", "Kolkata", "Pune", "Hyderabad", "Chennai"]
    city = st.selectbox("City", cities, index=cities.index(home_city) if home_city in cities else 0)
    new_phone = st.checkbox("📱 Pay from a NEW phone")
    to_mule = st.checkbox("💸 Send to an unknown account")
    if st.button("Score this payment", type="primary"):
        when = pd.Timestamp(s.last_ts, unit="s").normalize() + pd.Timedelta(days=1, hours=hour)
        dev = f"NEW-PHONE-{np.random.randint(10**6)}" if new_phone else home_dev
        out = api.score(api.Payment(user_id=uid, timestamp=str(when), amount=amount, device_id=dev,
                                    city=city, merchant_id="MULE999" if to_mule else "M0001"))
        if out["decision"] == "BLOCK":
            st.error(f"🚫 BLOCKED · fraud probability {out['fraud_probability']:.0%}")
            for r in out["reasons"]:
                st.write(f"• {r}")
        else:
            st.success(f"✅ ALLOWED · fraud probability {out['fraud_probability']:.1%}")
        st.caption(f"Decided in {out['latency_ms']} ms")
