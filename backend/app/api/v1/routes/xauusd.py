"""XAU/USD routes"""
from fastapi import APIRouter, Depends
from typing import Optional

router = APIRouter()


# Dependency injection for services
def get_market_data_service():
    """Get market data service instance"""
    from ....services.market_data_service import MarketDataService
    from ....services.prediction_service import PredictionService
    # These will be injected from main.py
    # For now, return None - will be set up in main.py
    return None


def get_prediction_service():
    """Get prediction service instance"""
    from ....services.prediction_service import PredictionService
    return None


@router.get("")
async def get_daily_data(days: int = 90):
    """Get XAU/USD daily data"""
    # Import from parent main module
    import sys
    from pathlib import Path
    main_path = Path(__file__).parent.parent.parent.parent / "main.py"
    if main_path.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("main_module", main_path)
        main_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main_module)
        if hasattr(main_module, 'get_xauusd_daily_data'):
            return main_module.get_xauusd_daily_data(days=days)
    return {"status": "error", "message": "Service not available"}


@router.get("/realtime")
async def get_realtime_price():
    """Get real-time XAU/USD price"""
    import sys
    from pathlib import Path
    main_path = Path(__file__).parent.parent.parent.parent / "main.py"
    if main_path.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("main_module", main_path)
        main_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main_module)
        if hasattr(main_module, 'get_realtime_price_data') and hasattr(main_module, 'get_xauusd_daily_data'):
            realtime_data = main_module.get_realtime_price_data()
            if realtime_data:
                daily_data = main_module.get_xauusd_daily_data()
                if daily_data.get('status') == 'success':
                    daily_data.update({
                        'current_price': realtime_data['current_price'],
                        'price_change': realtime_data.get('price_change', 0),
                        'change_percentage': realtime_data.get('change_percentage', 0),
                        'last_updated': realtime_data.get('last_updated'),
                        'realtime_symbol': realtime_data.get('symbol')
                    })
                return daily_data
            else:
                return main_module.get_xauusd_daily_data()
    return {"status": "error", "message": "Service not available"}
