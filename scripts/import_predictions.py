#!/usr/bin/env python3
"""Import predictions from backup SQL file"""
from backend.app.core.database import init_postgresql_pool, get_db_type
from backend.app.repositories.prediction_repository import PredictionRepository
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# Initialize PostgreSQL connection
print("Initializing database connection...")
init_postgresql_pool()
db_type = get_db_type()
print(f"Using database: {db_type.upper()}\n")


# All predictions from the backup file
# Format: (date, predicted_price, actual_price, method)
predictions_data = [
    ("2025-11-10", 3995.4087, 4061.3, "Lasso Regression"),
    ("2025-11-11", 4046.199, 4152.4, "Lasso Regression"),
    ("2025-11-12", 4089.694, 4116.2, "Lasso Regression"),
    ("2025-11-13", 4086.96, 4214.3, "Lasso Regression (Fallback)"),
    ("2025-11-14", 4173.9834, 4207.8, "Lasso Regression"),
    ("2025-11-15", 4166.22, 4087.6, "Lasso Regression (Fallback)"),
    ("2025-11-16", 4104.99, 4094.2, "Lasso Regression (Fallback)"),
    ("2025-11-17", 4113.6064, 4077.4, "Lasso Regression"),
    ("2025-11-18", 4073.1453, 4009.8, "Lasso Regression"),
    ("2025-11-19", 4055.9119, 4074.5, "Lasso Regression"),
    ("2025-11-20", 4059.19, 4069.2, "Lasso Regression"),
    ("2025-11-21", 4059.556, 4065, "Lasso Regression"),
    ("2025-11-22", 4071.06, 4076.7, "Lasso Regression (Fallback)"),
    ("2025-11-23", 4052.98, 4070.8, "Lasso Regression (Fallback)"),
    ("2025-11-24", 4065.0532, 4135, "Lasso Regression"),
    ("2025-11-25", 4083.255, 4130, "Lasso Regression"),
    ("2025-11-26", 4112.7, 4165, "Lasso Regression (Fallback)"),
    ("2025-11-27", 4135.1587, 4156, "Lasso Regression (Fallback)"),
    ("2025-11-28", 4138.9116, 4254.9, "Lasso Regression (Fallback)"),
    ("2025-11-29", 4135.616, 4254.9, "Lasso Regression (Fallback)"),
    ("2025-11-30", 4152.691, None, "Lasso Regression (Fallback)"),
    ("2025-12-01", 4152.691, 4239.3, "Lasso Regression (Fallback)"),
    ("2025-12-02", 4173.7827, 4186.6, "Lasso Regression (Fallback)"),
    ("2025-12-03", 4182.149, 4199.3, "Lasso Regression (Fallback)"),
    ("2025-12-04", 4162.219, 4211.8, "Lasso Regression (Fallback)"),
    ("2025-12-08", 4205.6396, 4236.9, "Lasso Regression (Fallback)"),
    ("2025-12-09", 4201.2734, None, "Lasso Regression (Fallback)"),
    ("2025-12-15", 4269.845, None, "Lasso Regression (Fallback)"),
]

if __name__ == "__main__":
    print("=" * 60)
    print("Importing Predictions from Backup File")
    print("=" * 60)
    print(f"Total predictions to import: {len(predictions_data)}\n")

    success = 0
    failed = 0

    for date, predicted_price, actual_price, method in predictions_data:
        try:
            result = PredictionRepository.save_prediction(
                prediction_date=date,
                predicted_price=predicted_price,
                actual_price=actual_price,
                prediction_method=method
            )
            if result:
                success += 1
                actual_str = f" (Actual: ${actual_price:.2f})" if actual_price else " (Pending)"
                print(f"✅ {date}: ${predicted_price:.2f}{actual_str} - {method}")
            else:
                failed += 1
                print(f"❌ Failed to import {date}")
        except Exception as e:
            failed += 1
            print(f"❌ Error importing {date}: {e}")

    print("\n" + "=" * 60)
    print(f"✅ Successfully imported: {success}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {len(predictions_data)}")
    print("=" * 60)



