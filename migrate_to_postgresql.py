#!/usr/bin/env python3
"""
Migration script to migrate predictions from SQLite to PostgreSQL
"""
import sqlite3
import os
import sys
from pathlib import Path

# PostgreSQL support
try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:
    print("Error: psycopg2 not installed. Run: pip install psycopg2-binary")
    sys.exit(1)

# Configuration
BACKEND_DIR = Path(__file__).resolve().parent / "backend"
SQLITE_DB_PATH = str(BACKEND_DIR / "data/gold_predictions.db")

# PostgreSQL connection details (from environment or defaults)
POSTGRESQL_HOST = os.getenv("POSTGRESQL_HOST", "localhost")
POSTGRESQL_PORT = os.getenv("POSTGRESQL_PORT", "5432")
POSTGRESQL_DATABASE = os.getenv("POSTGRESQL_DATABASE", "gold_predictor")
POSTGRESQL_USER = os.getenv("POSTGRESQL_USER", "postgres")
POSTGRESQL_PASSWORD = os.getenv("POSTGRESQL_PASSWORD", "postgres")


def connect_postgresql():
    """Connect to PostgreSQL database"""
    try:
        conn = psycopg2.connect(
            host=POSTGRESQL_HOST,
            port=POSTGRESQL_PORT,
            database=POSTGRESQL_DATABASE,
            user=POSTGRESQL_USER,
            password=POSTGRESQL_PASSWORD
        )
        print(f"✅ Connected to PostgreSQL: {POSTGRESQL_DATABASE}")
        return conn
    except Exception as e:
        print(f"❌ Failed to connect to PostgreSQL: {e}")
        print("\nMake sure PostgreSQL is running and credentials are correct.")
        print("\nYou can set connection details via environment variables:")
        print("  export POSTGRESQL_HOST=localhost")
        print("  export POSTGRESQL_DATABASE=gold_predictor")
        print("  export POSTGRESQL_USER=postgres")
        print("  export POSTGRESQL_PASSWORD=postgres")
        sys.exit(1)


def migrate_data():
    """Migrate data from SQLite to PostgreSQL"""
    # Check if SQLite database exists
    if not os.path.exists(SQLITE_DB_PATH):
        print(f"❌ SQLite database not found: {SQLITE_DB_PATH}")
        sys.exit(1)

    # Connect to SQLite
    sqlite_conn = sqlite3.connect(SQLITE_DB_PATH)
    sqlite_cursor = sqlite_conn.cursor()

    # Connect to PostgreSQL
    pg_conn = connect_postgresql()
    pg_cursor = pg_conn.cursor()

    try:
        # Initialize PostgreSQL database
        print("\n📊 Initializing PostgreSQL tables...")
        pg_cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                prediction_date DATE NOT NULL,
                predicted_price REAL NOT NULL,
                actual_price REAL,
                accuracy_percentage REAL,
                prediction_method TEXT DEFAULT 'Lasso Regression',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        pg_cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_prediction_date ON predictions(prediction_date)
        ''')
        pg_cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_created_at ON predictions(created_at)
        ''')
        pg_conn.commit()
        print("✅ PostgreSQL tables created")

        # Get data from SQLite
        print("\n📥 Reading data from SQLite...")
        sqlite_cursor.execute('''
            SELECT prediction_date, predicted_price, actual_price, 
                   accuracy_percentage, created_at, updated_at
            FROM predictions
            ORDER BY id
        ''')
        predictions = sqlite_cursor.fetchall()
        print(f"✅ Found {len(predictions)} predictions to migrate")

        if len(predictions) == 0:
            print("⚠️  No predictions to migrate")
            return

        # Check if PostgreSQL table is empty
        pg_cursor.execute('SELECT COUNT(*) FROM predictions')
        existing_count = pg_cursor.fetchone()[0]

        if existing_count > 0:
            response = input(f"\n⚠️  PostgreSQL already has {existing_count} predictions. Overwrite? (y/N): ")
            if response.lower() != 'y':
                print("Migration cancelled")
                return
            print("🗑️  Clearing existing predictions...")
            pg_cursor.execute('DELETE FROM predictions')
            pg_conn.commit()

        # Migrate data
        print("\n📤 Migrating predictions to PostgreSQL...")
        for i, pred in enumerate(predictions, 1):
            try:
                # Handle prediction_method column (might not exist in SQLite)
                pg_cursor.execute('''
                    INSERT INTO predictions 
                    (prediction_date, predicted_price, actual_price, 
                     accuracy_percentage, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                ''', pred[:6])  # Take first 6 columns
            except Exception as e:
                print(f"⚠️  Error migrating prediction {i}: {e}")
                continue

            if i % 100 == 0:
                print(f"  Migrated {i}/{len(predictions)} predictions...")
                pg_conn.commit()

        pg_conn.commit()
        print(f"✅ Successfully migrated {len(predictions)} predictions")

        # Verify migration
        pg_cursor.execute('SELECT COUNT(*) FROM predictions')
        pg_count = pg_cursor.fetchone()[0]
        print(f"\n✅ Verification: PostgreSQL now has {pg_count} predictions")

    except Exception as e:
        print(f"❌ Migration error: {e}")
        pg_conn.rollback()
        raise
    finally:
        sqlite_conn.close()
        pg_conn.close()
        print("\n✅ Migration completed!")


if __name__ == "__main__":
    print("=" * 60)
    print("SQLite to PostgreSQL Migration Script")
    print("=" * 60)
    migrate_data()




