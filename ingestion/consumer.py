import json
import os

from dotenv import load_dotenv
from kafka import KafkaConsumer
import psycopg2

# Load environment variables from .env
load_dotenv()

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "transactions_raw")

# PostgreSQL configuration
POSTGRES_USER = os.getenv("POSTGRES_USER", "admin")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "admin")
POSTGRES_DB = os.getenv("POSTGRES_DB", "fraud_db")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5433"))

# Table creation SQL (idempotent)
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id TEXT PRIMARY KEY,
    user_id INT,
    amount NUMERIC,
    merchant TEXT,
    location TEXT,
    device_id TEXT,
    timestamp TIMESTAMP,
    is_fraud INT CHECK (is_fraud IN (0, 1))
);
"""

# Insert SQL with duplicate-safe behavior
# If same transaction_id arrives again, it is ignored.
INSERT_SQL = """
INSERT INTO transactions (
    transaction_id, user_id, amount, merchant, location, device_id, timestamp, is_fraud
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (transaction_id) DO NOTHING;
"""


def get_db_connection():
    """Create and return a PostgreSQL connection."""
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        dbname=POSTGRES_DB,
    )


def main():
    # Connect to Postgres and ensure target table exists
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(CREATE_TABLE_SQL)
    conn.commit()
    print("Connected to Postgres and ensured transactions table exists.")

    # Subscribe to Kafka topic and deserialize JSON messages
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        auto_offset_reset="earliest",  # first run reads from beginning
        group_id="fraud-consumer-group",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        enable_auto_commit=True,
    )
    print(f"Consumer listening. Topic={KAFKA_TOPIC}, Broker={KAFKA_BOOTSTRAP_SERVERS}")

    # Process stream continuously
    for message in consumer:
        txn = message.value

        # Insert transaction into Postgres
        cur.execute(
            INSERT_SQL,
            (
                txn["transaction_id"],
                txn["user_id"],
                txn["amount"],
                txn["merchant"],
                txn["location"],
                txn["device_id"],
                txn["timestamp"],
                txn["is_fraud"],
            ),
        )
        conn.commit()

        print(
            f"[INSERTED] transaction_id={txn['transaction_id']} "
            f"amount={txn['amount']} is_fraud={txn['is_fraud']}"
        )


if __name__ == "__main__":
    main()