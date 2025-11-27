"""Exchange rate schemas"""
from pydantic import BaseModel
from typing import Optional


class ExchangeRateResponse(BaseModel):
    """Exchange rate response"""
    from_currency: str
    to_currency: str
    exchange_rate: float
    timestamp: str
    status: str
    message: Optional[str] = None



