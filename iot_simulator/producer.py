from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import json
from math import sqrt
import os
import random
import time
from typing import Deque, Dict

from kafka import KafkaProducer


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


BOOTSTRAP = _env("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC = _env("KAFKA_TOPIC", "load_readings")
RATE_PER_SEC = float(_env("RATE_PER_SEC", "2"))
SEND_MEASURED_POWER = _env("SEND_MEASURED_POWER", "1") == "1"
SITE_ID = _env("SITE_ID", "site-001")

# Maintain a moving window of the last 3 measured kW values to compute lag/rolling features.
_window: Deque[float] = deque(maxlen=3)
_last_power_kw: float | None = None


def hour_profile(hour: int, weekend: bool) -> float:
    """
    Simple diurnal + weekend profile for CURRENT (A).
    Weekday business hours peak; nights/weekends lower.
    Returns a multiplier ~ [0.6, 1.2].
    """
    base = 0.8 if weekend else 1.0
    if 7 <= hour <= 10:
        return base * 1.05
    if 11 <= hour <= 15:
        return base * 1.15
    if 16 <= hour <= 19:
        return base * 1.20
    if 0 <= hour <= 5 or 22 <= hour <= 23:
        return base * 0.70
    return base * 0.9


def season_from_month(month: int) -> int:
    # 1: Winter, 2: Spring, 3: Summer, 4: Autumn (simple mapping)
    if month in (12, 1, 2):
        return 1
    if month in (3, 4, 5):
        return 2
    if month in (6, 7, 8):
        return 3
    return 4


def generate_record(now: datetime) -> Dict[str, float | int | str]:
    global _last_power_kw

    # Weekend flag: 0=weekday, 1=weekend (keeps your earlier convention)
    weekend_weekday = 1 if now.weekday() >= 5 else 0
    season = season_from_month(now.month)

    # Electrical quantities (load-side).
    # Voltage ~ 225-240 V with small noise
    voltage = random.uniform(225.0, 240.0)

    # Current baseline (A) with diurnal pattern + noise
    base_current = 40.0  # arbitrary asset scale; change via env later if needed
    current = max(
        0.0,
        random.gauss(base_current * hour_profile(now.hour, weekend_weekday == 1), 3.0),
    )

    # Power Factor near unity for typical commercial loads
    pf = min(0.99, max(0.88, random.gauss(0.95, 0.015)))

    # Simple ambient signals
    temp_f = random.gauss(82.0 if season == 3 else 72.0, 4.0)  # warmer in summer
    humidity = min(95.0, max(15.0, random.gauss(55.0, 10.0)))

    # Compute a synthetic measured real power (kW) from 3-phase approximation
    power_kw_measured = (sqrt(3) * voltage * current * pf) / 1000.0

    # Update window for lag/rolling features
    if _last_power_kw is not None:
        _window.append(_last_power_kw)
    _last_power_kw = power_kw_measured

    lag1 = _window[-1] if _window else power_kw_measured  # fallback to current if no history yet
    rolling_mean_3 = (
        sum(list(_window)[-3:] + [power_kw_measured]) / (len(_window) + 1) if _window else power_kw_measured
    )

    record: Dict[str, float | int | str] = {
        "VOLTAGE": round(voltage, 2),
        "CURRENT": round(current, 2),
        "PF": round(pf, 3),
        "Temp_F": round(temp_f, 1),
        "Humidity": round(humidity, 1),
        "WEEKEND_WEEKDAY": int(weekend_weekday),
        "SEASON": int(season),
        "lag1": round(lag1, 2),
        "rolling_mean_3": round(rolling_mean_3, 2),
        "ts": now.replace(tzinfo=timezone.utc).isoformat(),
        "site_id": SITE_ID,
    }
    if SEND_MEASURED_POWER:
        record["power_kw_measured"] = round(power_kw_measured, 2)

    return record


def main() -> None:
    print(
        f"[producer] bootstrap={BOOTSTRAP} topic={TOPIC} rate={RATE_PER_SEC}/s "
        f"send_measured_power={SEND_MEASURED_POWER}"
    )
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP,
        value_serializer=lambda d: json.dumps(d).encode("utf-8"),
        linger_ms=50,
        acks="all",
        retries=3,
    )
    try:
        interval = 1.0 / max(RATE_PER_SEC, 0.1)
        while True:
            now = datetime.utcnow()
            rec = generate_record(now)
            producer.send(TOPIC, rec)
            # Optional: log one line preview
            measured_kw = rec.get("power_kw_measured", "-")
            print(f"[producer] {rec['ts']} {rec['site_id']} -> kW~{measured_kw} sent")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[producer] stopped by user")
    finally:
        producer.flush(timeout=5)
        producer.close()


if __name__ == "__main__":
    main()
