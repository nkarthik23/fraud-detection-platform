import json
import random
import time
import uuid
from datetime import datetime
import os

from dotenv import load_dotenv
from kafka import KafkaProducer

# Load values from .env into environment variables
load_dotenv()

# Kafka config (defaults allow local dev even if .env is missing)
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "transactions_raw")

# Merchant/location pools for synthetic data generation
MERCHANTS = [
    "Grocery", "Gas", "Electronics", "Travel", "Restaurant",
    "Online Retail", "Luxury", "Crypto", "Pharmacy"
]
LOCATIONS = ["CA", "NY", "TX", "FL", "WA", "NV", "AZ", "IL", "MA"]

# Higher-risk merchant categories for simulation logic
RISKY_MERCHANTS = {"Luxury", "Crypto"}

# Kafka producer serializes Python dict -> JSON bytes
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)


def generate_transaction() -> dict:
    """
    Create one synthetic transaction event.
    Includes mostly normal traffic + occasional suspicious patterns.
    """
    merchant = random.choice(MERCHANTS)
    location = random.choice(LOCATIONS)
    amount = round(random.uniform(5, 500), 2)  # normal transaction amount range

    # Occasionally generate a high-value transaction (more suspicious)
    if random.random() < 0.15:
        amount = round(random.uniform(800, 5000), 2)

    # Simulated device fingerprint ID
    device_id = f"device_{random.randint(1, 200)}"

    # Simple rule-based fraud probability for synthetic label generation
    fraud_prob = 0.02  # base rate
    if amount > 1000:
        fraud_prob += 0.25
    if merchant in RISKY_MERCHANTS:
        fraud_prob += 0.20
    if location in {"NV", "AZ"}:
        fraud_prob += 0.10

    is_fraud = 1 if random.random() < fraud_prob else 0

    # Final event payload sent to Kafka
    return {
        "transaction_id": str(uuid.uuid4()),
        "user_id": random.randint(1000, 9999),
        "merchant": merchant,
        "amount": amount,
        "timestamp": datetime.utcnow().isoformat(),
        "device_id": device_id,
        "location": location,
        "is_fraud": is_fraud,
    }


def main():
    print(f"Producer started. Topic={KAFKA_TOPIC}, Broker={KAFKA_BOOTSTRAP_SERVERS}")

    # Infinite stream to simulate real-time payment flow
    while True:
        txn = generate_transaction()

        # Publish event to Kafka topic
        producer.send(KAFKA_TOPIC, txn)
        producer.flush()  # force send immediately for easier debugging/demo

        print(f"[PRODUCED] {txn}")

        # Delay between events to mimic transaction arrival cadence
        time.sleep(random.uniform(1, 2))


if __name__ == "__main__":
    main()