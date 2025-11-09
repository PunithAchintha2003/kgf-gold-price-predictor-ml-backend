#!/bin/bash
# Startup script with PostgreSQL enabled

export USE_POSTGRESQL=true
export POSTGRESQL_HOST=localhost
export POSTGRESQL_PORT=5432
export POSTGRESQL_DATABASE=gold_predictor
export POSTGRESQL_USER=$USER
export POSTGRESQL_PASSWORD=""

echo "🚀 Starting server with PostgreSQL..."
echo "USE_POSTGRESQL: $USE_POSTGRESQL"
echo "POSTGRESQL_DATABASE: $POSTGRESQL_DATABASE"
echo ""

python3 run_backend.py




