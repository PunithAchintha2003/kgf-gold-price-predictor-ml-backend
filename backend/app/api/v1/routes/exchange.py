"""Exchange rate routes"""
from fastapi import APIRouter
from ....services.exchange_service import ExchangeService

router = APIRouter()
exchange_service = ExchangeService()


@router.get("/{from_currency}/{to_currency}")
async def get_exchange_rate(from_currency: str, to_currency: str):
    """Get exchange rate between currencies"""
    return exchange_service.get_exchange_rate(from_currency, to_currency)

