#!/usr/bin/env python3
"""
Script to train the News-Enhanced Lasso model for gold price prediction.
Run this script to generate the enhanced_lasso_gold_model.pkl file.

Usage:
    python train_enhanced_model.py
"""
from models.news_prediction import main
import sys
import os
from pathlib import Path

# Add backend to Python path
project_root = Path(__file__).resolve().parent
backend_path = project_root / "backend"

if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now import the training module

if __name__ == "__main__":
    print("=" * 60)
    print("Training News-Enhanced Lasso Model for Gold Price Prediction")
    print("=" * 60)
    print("\nThis process will:")
    print("1. Fetch market data (gold prices, economic indicators)")
    print("2. Fetch and analyze news sentiment (may take a few minutes)")
    print("3. Create enhanced features combining market + news data")
    print("4. Train the enhanced Lasso regression model")
    print("5. Save the model to backend/models/enhanced_lasso_gold_model.pkl")
    print("\n" + "=" * 60)
    print("Starting training...\n")

    try:
        main()
        print("\n" + "=" * 60)
        print("✅ Training completed successfully!")
        print("The model has been saved to: backend/models/enhanced_lasso_gold_model.pkl")
        print("=" * 60)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
