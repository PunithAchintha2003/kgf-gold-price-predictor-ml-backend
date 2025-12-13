#!/usr/bin/env python3
"""Check predictions in database and compare with backup"""
from backend.app.core.database import init_postgresql_pool, get_db_connection
from backend.app.repositories.prediction_repository import PredictionRepository
from datetime import datetime

# Initialize PostgreSQL
init_postgresql_pool()

# Expected predictions from backup (all unique dates)
expected_dates = [
    "2025-11-10", "2025-11-11", "2025-11-12", "2025-11-13", "2025-11-14",
    "2025-11-15", "2025-11-16", "2025-11-17", "2025-11-18", "2025-11-19",
    "2025-11-20", "2025-11-21", "2025-11-22", "2025-11-23", "2025-11-24",
    "2025-11-25", "2025-11-26", "2025-11-27", "2025-11-28", "2025-11-29",
    "2025-11-30", "2025-12-01", "2025-12-02", "2025-12-03", "2025-12-04",
    "2025-12-08", "2025-12-09", "2025-12-15"
]

# Check all predictions in database
print("=" * 60)
print("Checking Predictions in Database")
print("=" * 60)

with get_db_connection() as conn:
    cursor = conn.cursor()
    cursor.execute(
        'SELECT prediction_date, predicted_price, actual_price, prediction_method FROM predictions ORDER BY prediction_date')
    db_predictions = cursor.fetchall()

    db_dates = set()
    weekday_count = 0
    weekend_count = 0
    evaluated_count = 0
    pending_count = 0

    print(f"\nTotal records in database: {len(db_predictions)}")
    print("\nAll predictions in database:")
    for date, pred_price, actual_price, method in db_predictions:
        if isinstance(date, str):
            date_obj = datetime.strptime(date, '%Y-%m-%d')
        else:
            date_obj = date
        date_str = date_obj.strftime('%Y-%m-%d')
        db_dates.add(date_str)

        weekday = date_obj.weekday()
        is_weekend = weekday >= 5

        status = "✅ Evaluated" if actual_price else "⏳ Pending"
        weekend_str = " (WEEKEND)" if is_weekend else ""
        method_str = method or "Unknown"

        print(
            f"  {date_str}: ${pred_price:.2f} - {status}{weekend_str} - {method_str}")

        if not is_weekend:
            weekday_count += 1
            if actual_price:
                evaluated_count += 1
            else:
                pending_count += 1
        else:
            weekend_count += 1

print(f"\n📊 Database Summary:")
print(f"  Total records: {len(db_predictions)}")
print(f"  Unique dates: {len(db_dates)}")
print(f"  Weekday predictions: {weekday_count}")
print(f"  Weekend predictions: {weekend_count} (filtered out)")
print(f"  Evaluated: {evaluated_count}")
print(f"  Pending: {pending_count}")

# Compare with expected
print(f"\n📋 Comparison with Backup:")
print(f"  Expected dates: {len(expected_dates)}")
print(f"  Found in database: {len(db_dates)}")
missing = set(expected_dates) - db_dates
extra = db_dates - set(expected_dates)

if missing:
    print(f"  ❌ Missing dates ({len(missing)}): {sorted(missing)}")
if extra:
    print(f"  ⚠️  Extra dates ({len(extra)}): {sorted(extra)}")
if not missing and not extra:
    print(f"  ✅ All expected dates found!")

# Get stats from API
print("\n" + "=" * 60)
print("Stats from get_comprehensive_stats():")
print("=" * 60)
stats = PredictionRepository.get_comprehensive_stats()
print(f"Total predictions: {stats['total_predictions']}")
print(f"Evaluated: {stats['evaluated_predictions']}")
print(f"Pending: {stats['pending_predictions']}")
print(f"Average accuracy: {stats.get('average_accuracy', 'N/A')}")

# Check which dates are weekends
print("\n" + "=" * 60)
print("Weekend Analysis:")
print("=" * 60)
weekend_dates = []
weekday_dates = []
for date_str in expected_dates:
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    if date_obj.weekday() >= 5:
        weekend_dates.append(date_str)
    else:
        weekday_dates.append(date_str)

print(f"Weekend dates ({len(weekend_dates)}): {weekend_dates}")
print(f"Weekday dates ({len(weekday_dates)}): {len(weekday_dates)} total")
print(f"\nExpected weekday count: {len(weekday_dates)}")
print(f"Database weekday count: {weekday_count}")
print(f"Stats API weekday count: {stats['total_predictions']}")

if stats['total_predictions'] != len(weekday_dates):
    print(
        f"\n⚠️  DISCREPANCY: Stats shows {stats['total_predictions']} but should be {len(weekday_dates)}")
    print(
        f"   Missing: {len(weekday_dates) - stats['total_predictions']} predictions")
