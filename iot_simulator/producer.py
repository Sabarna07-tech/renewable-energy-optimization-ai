import json
import time
import random
from datetime import datetime
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',  # Change if running Kafka elsewhere
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def generate_fake_sensor_data():
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "power_kw": round(random.uniform(20, 100), 2),
        "temperature": round(random.uniform(20, 45), 2),
        "wind_speed": round(random.uniform(0, 15), 2),
        "solar_irradiance": round(random.uniform(0, 1000), 2),
        "sensor_id": random.randint(1, 5)
    }

if __name__ == "__main__":
    while True:
        data = generate_fake_sensor_data()
        print("Sending data:", data)
        producer.send('energy-data', value=data)
        time.sleep(2)  # Send every 2 seconds
