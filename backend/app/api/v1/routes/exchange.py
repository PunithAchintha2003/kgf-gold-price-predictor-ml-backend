"""Exchange rate routes"""
from fastapi import APIRouter, Depends
from ....core.dependencies import get_exchange_service

router = APIRouter()


@router.get("/{from_currency}/{to_currency}")
async def get_exchange_rate(
    from_currency: str,
    to_currency: str,
    exchange_service=Depends(get_exchange_service)
):
    """Get exchange rate between currencies - supports both naming conventions"""
    return exchange_service.get_exchange_rate(from_currency, to_currency)
