create table if not exists transactions (
    transaction_id text primary key,
    user_id int,
    amount numeric,
    merchant text,
    location text,
    device_id text,
    timestamp timestamp,
    is_fraud int
)

