#!/bin/bash
# Script to retrain the news-enhanced Lasso model with fresh data

echo "🔄 Retraining News-Enhanced Lasso Model"
echo "========================================"

# Check if we're in the backend directory
if [ ! -f "models/news_prediction.py" ]; then
    echo "❌ Error: Please run this script from the backend directory"
    echo "   cd backend && bash RETRAIN_MODEL.sh"
    exit 1
fi

# Check for API keys (optional but recommended)
if [ -z "$NEWS_API_KEY" ] && [ -z "$ALPHA_VANTAGE_KEY" ]; then
    echo "⚠️  Warning: No news API keys found in environment"
    echo "   Model will only use RSS feeds (limited data)"
    echo ""
    echo "   To improve accuracy, set API keys:"
    echo "   export NEWS_API_KEY='your_key'"
    echo "   export ALPHA_VANTAGE_KEY='your_key'"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✅ API keys found - will fetch from multiple sources"
fi

# Train the model
echo ""
echo "🏋️  Training model... (this may take a few minutes)"
python3 -m models.news_prediction

# Check if training was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Model training completed successfully!"
    echo ""
    echo "📊 Model saved to: models/enhanced_lasso_gold_model.pkl"
    echo ""
    echo "🚀 Next steps:"
    echo "   1. git add models/enhanced_lasso_gold_model.pkl"
    echo "   2. git commit -m 'Update news-enhanced model with fresh data'"
    echo "   3. git push origin $(git branch --show-current)"
    echo ""
else
    echo ""
    echo "❌ Model training failed!"
    echo "   Check the error messages above"
    exit 1
fi
