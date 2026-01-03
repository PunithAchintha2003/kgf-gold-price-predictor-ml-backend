"""Spot trading API routes"""
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Optional
import sys
from pathlib import Path

# Add backend/app to path for imports
backend_app_path = Path(__file__).resolve().parent.parent / "app"
if str(backend_app_path) not in sys.path:
    sys.path.insert(0, str(backend_app_path))

from app.core.dependencies import SpotTradingServiceDep
from app.core.exceptions import ValidationError
from spot_trade.schemas import (
    SpotTradePriceResponse,
    BuyOrderRequest,
    SellOrderRequest,
    TradeResponse,
    BalanceResponse,
    TradeHistoryResponse,
    OpenOrdersResponse
)
from spot_trade.models import get_user_trades, get_open_orders

router = APIRouter()


# Helper function to get user ID from request
# In production, this should extract from JWT token or session
def get_user_id() -> str:
    """Get user ID from request (placeholder - should use actual auth)"""
    # TODO: Implement proper authentication
    # For now, using a default user ID for testing
    # In production, extract from JWT token or session
    return "default_user"


@router.get("/price", response_model=SpotTradePriceResponse)
async def get_current_price(
    spot_trading_service: SpotTradingServiceDep
):
    """Get current gold price in LKR with spread"""
    try:
        price_data = spot_trading_service.get_current_price()
        return SpotTradePriceResponse(**price_data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get current price: {str(e)}"
        )


@router.post("/buy", response_model=TradeResponse)
async def place_buy_order(
    request: BuyOrderRequest,
    spot_trading_service: SpotTradingServiceDep,
    user_id: str = Depends(get_user_id)
):
    """Place a BUY market order"""
    try:
        result = spot_trading_service.place_buy_order(
            user_id=user_id,
            quantity=request.quantity
        )
        return TradeResponse(**result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to place BUY order: {str(e)}"
        )


@router.post("/sell", response_model=TradeResponse)
async def place_sell_order(
    request: SellOrderRequest,
    spot_trading_service: SpotTradingServiceDep,
    user_id: str = Depends(get_user_id)
):
    """Place a SELL market order"""
    try:
        result = spot_trading_service.place_sell_order(
            user_id=user_id,
            quantity=request.quantity
        )
        return TradeResponse(**result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to place SELL order: {str(e)}"
        )


@router.get("/balance", response_model=BalanceResponse)
async def get_user_balance(
    spot_trading_service: SpotTradingServiceDep,
    user_id: str = Depends(get_user_id)
):
    """Get user balance"""
    try:
        balance = spot_trading_service.get_user_balance(user_id=user_id)
        return BalanceResponse(**balance)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get balance: {str(e)}"
        )


@router.get("/history", response_model=TradeHistoryResponse)
async def get_trade_history(
    limit: int = 100,
    offset: int = 0,
    user_id: str = Depends(get_user_id)
):
    """Get user trade history"""
    try:
        if limit < 1 or limit > 1000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Limit must be between 1 and 1000"
            )
        if offset < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Offset must be >= 0"
            )
        
        trades = get_user_trades(user_id=user_id, limit=limit, offset=offset)
        
        from .schemas import TradeHistoryItem
        trade_items = [TradeHistoryItem(**trade) for trade in trades]
        
        return TradeHistoryResponse(
            trades=trade_items,
            total=len(trade_items),
            limit=limit,
            offset=offset
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get trade history: {str(e)}"
        )


@router.get("/orders", response_model=OpenOrdersResponse)
async def get_open_orders_endpoint(
    user_id: str = Depends(get_user_id)
):
    """Get user's open orders"""
    try:
        orders = get_open_orders(user_id=user_id)
        
        from .schemas import OpenOrderItem
        order_items = [OpenOrderItem(**order) for order in orders]
        
        return OpenOrdersResponse(
            orders=order_items,
            total=len(order_items)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get open orders: {str(e)}"
        )

