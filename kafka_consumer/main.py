import csv
import json
import os
import threading

from fastapi import FastAPI
from kafka import KafkaConsumer

# ====== CSV Storage Setup ======


DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
CSV_FILE = os.path.join(DATA_DIR, "energy_data.csv")

# Create the /data directory if it doesn't exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Create CSV with header if it doesn't exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "power_kw",
                "temperature",
                "wind_speed",
                "solar_irradiance",
                "sensor_id",
            ]
        )


# Function to append a row to CSV
def append_to_csv(data):
    with open(CSV_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                data.get("timestamp", ""),
                data.get("power_kw", ""),
                data.get("temperature", ""),
                data.get("wind_speed", ""),
                data.get("solar_irradiance", ""),
                data.get("sensor_id", ""),
            ]
        )


# ====== FastAPI + Kafka Consumer App ======
app = FastAPI()
buffer = []


def consume():
    consumer = KafkaConsumer(
        "energy-data",
        bootstrap_servers="localhost:9092",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="latest",
        group_id="my-group",
    )
    for msg in consumer:
        buffer.append(msg.value)
        append_to_csv(msg.value)  # <-- Store in CSV
        print("Received and saved:", msg.value)


# Start Kafka consumer in a background thread
threading.Thread(target=consume, daemon=True).start()


@app.get("/latest")
def get_latest():
    return buffer[-10:] if buffer else []


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
