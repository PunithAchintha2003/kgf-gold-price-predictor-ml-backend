#!/usr/bin/env python3
"""
Backup Manager for Gold Price Predictor
Provides command-line interface for managing prediction backups
"""

import sqlite3
import os
import sys
from datetime import datetime

# Database paths
DB_PATH = "backend/data/gold_predictions.db"
BACKUP_DB_PATH = "backend/data/gold_predictions_backup.db"


def backup_predictions():
    """Backup all predictions to backup database"""
    try:
        # Connect to both databases
        main_conn = sqlite3.connect(DB_PATH)
        backup_conn = sqlite3.connect(BACKUP_DB_PATH)

        main_cursor = main_conn.cursor()
        backup_cursor = backup_conn.cursor()

        # Get all predictions from main database
        main_cursor.execute('''
            SELECT prediction_date, predicted_price, actual_price, accuracy_percentage, created_at, updated_at
            FROM predictions
            ORDER BY created_at
        ''')

        predictions = main_cursor.fetchall()

        # Clear existing backup data
        backup_cursor.execute('DELETE FROM predictions')

        # Insert all predictions into backup database
        for pred in predictions:
            backup_cursor.execute('''
                INSERT INTO predictions (prediction_date, predicted_price, actual_price, accuracy_percentage, created_at, updated_at, backup_created_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', pred)

        backup_conn.commit()

        main_conn.close()
        backup_conn.close()

        print(
            f"✅ Successfully backed up {len(predictions)} predictions to backup database")
        return True

    except Exception as e:
        print(f"❌ Error backing up predictions: {e}")
        return False


def restore_from_backup():
    """Restore predictions from backup database"""
    try:
        # Connect to both databases
        main_conn = sqlite3.connect(DB_PATH)
        backup_conn = sqlite3.connect(BACKUP_DB_PATH)

        main_cursor = main_conn.cursor()
        backup_cursor = backup_conn.cursor()

        # Get all predictions from backup database
        backup_cursor.execute('''
            SELECT prediction_date, predicted_price, actual_price, accuracy_percentage, created_at, updated_at
            FROM predictions
            ORDER BY created_at
        ''')

        predictions = backup_cursor.fetchall()

        # Clear existing main data
        main_cursor.execute('DELETE FROM predictions')

        # Insert all predictions into main database
        for pred in predictions:
            main_cursor.execute('''
                INSERT INTO predictions (prediction_date, predicted_price, actual_price, accuracy_percentage, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', pred)

        main_conn.commit()

        main_conn.close()
        backup_conn.close()

        print(
            f"✅ Successfully restored {len(predictions)} predictions from backup database")
        return True

    except Exception as e:
        print(f"❌ Error restoring from backup: {e}")
        return False


def backup_status():
    """Get backup database status"""
    try:
        backup_conn = sqlite3.connect(BACKUP_DB_PATH)
        backup_cursor = backup_conn.cursor()

        # Get backup statistics
        backup_cursor.execute('SELECT COUNT(*) FROM predictions')
        backup_count = backup_cursor.fetchone()[0]

        backup_cursor.execute('SELECT MAX(backup_created_at) FROM predictions')
        last_backup = backup_cursor.fetchone()[0]

        backup_conn.close()

        # Get main database count for comparison
        main_conn = sqlite3.connect(DB_PATH)
        main_cursor = main_conn.cursor()
        main_cursor.execute('SELECT COUNT(*) FROM predictions')
        main_count = main_cursor.fetchone()[0]
        main_conn.close()

        print("📊 Backup Status:")
        print(f"   Main database: {main_count} predictions")
        print(f"   Backup database: {backup_count} predictions")
        print(f"   Last backup: {last_backup}")
        print(
            f"   Backup synced: {'✅ Yes' if main_count == backup_count else '❌ No'}")

        return True
    except Exception as e:
        print(f"❌ Error checking backup status: {e}")
        return False


def init_backup_database():
    """Initialize backup SQLite database for storing predictions"""
    try:
        conn = sqlite3.connect(BACKUP_DB_PATH)
        cursor = conn.cursor()

        # Create predictions table with same structure plus backup timestamp
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_date TEXT NOT NULL,
                predicted_price REAL NOT NULL,
                actual_price REAL,
                accuracy_percentage REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                backup_created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()
        print("✅ Backup database initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Error initializing backup database: {e}")
        return False


def main():
    """Main function for command-line interface"""
    if len(sys.argv) < 2:
        print("Usage: python3 backup_manager.py <command>")
        print("Commands:")
        print("  init     - Initialize backup database")
        print("  backup   - Create backup of predictions")
        print("  restore  - Restore from backup")
        print("  status   - Show backup status")
        return

    command = sys.argv[1].lower()

    if command == "init":
        init_backup_database()
    elif command == "backup":
        backup_predictions()
    elif command == "restore":
        restore_from_backup()
    elif command == "status":
        backup_status()
    else:
        print(f"❌ Unknown command: {command}")
        print("Available commands: init, backup, restore, status")


if __name__ == "__main__":
    main()
