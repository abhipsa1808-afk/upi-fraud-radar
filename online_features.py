"""Online feature store: the SAME features as features.py, computed one transaction at a time."""
import math
from collections import defaultdict, deque

WINDOWS = {"10m": 600, "1h": 3600, "24h": 86400}


class UserState:
    """Everything we remember about one user."""
    __slots__ = ("recent", "n", "amt_sum", "hour_sum", "last_ts", "devices", "cities", "merchants")

    def __init__(self):
        self.recent = deque()      # (time, amount) of payments in the last 24h
        self.n = 0                 # total past payments
        self.amt_sum = 0.0
        self.hour_sum = 0.0
        self.last_ts = None
        self.devices = {}          # device -> [times used, first seen]
        self.cities = set()
        self.merchants = set()


class FeatureStore:
    def __init__(self):
        self.users = defaultdict(UserState)

    def compute(self, user_id, ts, amount, device_id, city, merchant_id, hour):
        """Features for ONE payment using only this user's past. ts = seconds."""
        s = self.users[user_id]
        while s.recent and s.recent[0][0] < ts - 86400:      # forget payments older than 24h
            s.recent.popleft()
        f = {}
        for name, w in WINDOWS.items():
            inside = [a for t, a in s.recent if t >= ts - w]
            f[f"tx_count_{name}"] = len(inside) + 1
            f[f"amount_sum_{name}"] = round(sum(inside) + amount, 2)
        f["is_night"] = int(hour < 6)
        f["log_amount"] = math.log1p(amount)
        f["amount_ratio"] = amount / (s.amt_sum / s.n) if s.n else 1.0
        f["hour_dev"] = abs(hour - s.hour_sum / s.n) if s.n else 0.0
        f["secs_since_last"] = ts - s.last_ts if s.last_ts is not None else 1e7
        f["user_history"] = s.n
        for key, val, seen in [("new_device", device_id, s.devices), ("new_city", city, s.cities),
                               ("new_merchant", merchant_id, s.merchants)]:
            f[key] = int(val not in seen and s.n > 0)
        cnt, first = s.devices.get(device_id, [0, ts])
        f["device_tx_count"] = cnt
        f["device_age_secs"] = ts - first

        # remember this payment for the next one
        s.recent.append((ts, amount))
        s.n += 1
        s.amt_sum += amount
        s.hour_sum += hour
        s.last_ts = ts
        s.devices.setdefault(device_id, [0, ts])[0] += 1
        s.cities.add(city)
        s.merchants.add(merchant_id)
        return f
