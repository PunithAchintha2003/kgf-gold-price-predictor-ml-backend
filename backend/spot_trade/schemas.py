"""Pydantic schemas for spot trading API"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import datetime


class SpotTradePriceResponse(BaseModel):
    """Response schema for current gold price"""
    symbol: str = "XAUUSD"
    current_price_lkr: float = Field(..., description="Current gold price in LKR per troy ounce")
    current_price_usd: float = Field(..., description="Current gold price in USD per troy ounce")
    spread_lkr: float = Field(default=500.0, description="Spread in LKR")
    buy_price_lkr: float = Field(..., description="Buy price (current + spread/2) in LKR")
    sell_price_lkr: float = Field(..., description="Sell price (current - spread/2) in LKR")
    exchange_rate: float = Field(..., description="USD to LKR exchange rate")
    timestamp: str = Field(..., description="Price timestamp")


class BuyOrderRequest(BaseModel):
    """Request schema for BUY order"""
    quantity: float = Field(..., gt=0, description="Quantity of gold to buy (in pawn)")
    
    @field_validator('quantity')
    @classmethod
    def validate_quantity(cls, v):
        if v <= 0:
            raise ValueError("Quantity must be greater than 0")
        if v > 4000:  # Reasonable limit (approx 1000 troy ounces = ~3888 pawn)
            raise ValueError("Quantity cannot exceed 4000 pawn")
        return v


class SellOrderRequest(BaseModel):
    """Request schema for SELL order"""
    quantity: float = Field(..., gt=0, description="Quantity of gold to sell (in pawn)")
    
    @field_validator('quantity')
    @classmethod
    def validate_quantity(cls, v):
        if v <= 0:
            raise ValueError("Quantity must be greater than 0")
        if v > 4000:  # Reasonable limit (approx 1000 troy ounces = ~3888 pawn)
            raise ValueError("Quantity cannot exceed 4000 pawn")
        return v


class TradeResponse(BaseModel):
    """Response schema for trade execution"""
    trade_id: int
    user_id: str
    order_type: str
    quantity: float
    price: float
    total_value: float
    status: str
    message: str
    created_at: str


class BalanceResponse(BaseModel):
    """Response schema for user balance"""
    user_id: str
    lkr_balance: float = Field(..., description="LKR balance")
    gold_balance: float = Field(..., description="Gold balance in troy ounces")
    total_value_lkr: float = Field(..., description="Total portfolio value in LKR (LKR + gold value)")


class TradeHistoryItem(BaseModel):
    """Schema for individual trade history item"""
    id: int
    user_id: str
    order_type: str
    quantity: float
    price: float
    total_value: float
    status: str
    created_at: str
    updated_at: str


class TradeHistoryResponse(BaseModel):
    """Response schema for trade history"""
    trades: list[TradeHistoryItem]
    total: int
    limit: int
    offset: int


class OpenOrderItem(BaseModel):
    """Schema for individual open order"""
    id: int
    user_id: str
    order_type: str
    quantity: float
    price: float
    total_value: float
    status: str
    created_at: str
    updated_at: str


class OpenOrdersResponse(BaseModel):
    """Response schema for open orders"""
    orders: list[OpenOrderItem]
    total: int

