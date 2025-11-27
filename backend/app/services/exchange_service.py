"""Exchange rate service"""
import logging
from datetime import datetime
from typing import Dict

logger = logging.getLogger(__name__)


class ExchangeService:
    """Service for exchange rate operations"""
    
    @staticmethod
    def get_exchange_rate(from_currency: str, to_currency: str) -> Dict:
        """Get exchange rate between currencies"""
        try:
            from_currency = from_currency.upper()
            to_currency = to_currency.upper()
            
            if from_currency == "USD" and to_currency == "LKR":
                # For USD/LKR, use a fallback rate
                # In production, integrate with a proper forex API
                fallback_rate = 300.0
                
                return {
                    "from_currency": from_currency,
                    "to_currency": to_currency,
                    "exchange_rate": fallback_rate,
                    "timestamp": datetime.now().isoformat(),
                    "status": "success"
                }
            else:
                # For other currency pairs, could use yfinance or other APIs
                return {
                    "from_currency": from_currency,
                    "to_currency": to_currency,
                    "exchange_rate": 1.0,
                    "timestamp": datetime.now().isoformat(),
                    "status": "error",
                    "message": f"Exchange rate for {from_currency}/{to_currency} not supported"
                }
        except Exception as e:
            logger.error(f"Error getting exchange rate: {e}")
            return {
                "from_currency": from_currency,
                "to_currency": to_currency,
                "exchange_rate": 1.0,
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "message": str(e)
            }



