"""Fallback data sources when primary source is rate limited"""
import logging
from typing import Optional, Dict
from datetime import datetime

logger = logging.getLogger(__name__)


class FallbackDataProvider:
    """Provides fallback gold price data when primary sources are unavailable"""

    # Approximate gold price (updated manually or via fallback sources)
    # This serves as last resort when all API sources fail
    LAST_KNOWN_PRICES = {
        'XAUUSD': 2650.00,  # Approximate as of Dec 2024
        'last_update': '2024-12-01'
    }

    @staticmethod
    def get_fallback_realtime_data() -> Optional[Dict]:
        """
        Get fallback realtime data when all sources are rate limited
        Returns approximate data based on last known prices
        """
        logger.info(
            "⚠️  Using fallback data - all primary sources unavailable")

        return {
            'current_price': FallbackDataProvider.LAST_KNOWN_PRICES['XAUUSD'],
            'price_change': 0.0,
            'change_percentage': 0.0,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': 'FALLBACK',
            'timestamp': datetime.now().isoformat(),
            'is_fallback': True,
            'note': 'Approximate price - primary data sources unavailable'
        }

    @staticmethod
    def update_last_known_price(price: float):
        """Update the last known price for fallback purposes"""
        FallbackDataProvider.LAST_KNOWN_PRICES['XAUUSD'] = price
        FallbackDataProvider.LAST_KNOWN_PRICES['last_update'] = datetime.now().strftime(
            '%Y-%m-%d')
        logger.debug(f"Updated fallback price to ${price:.2f}")


# Global instance
fallback_provider = FallbackDataProvider()
