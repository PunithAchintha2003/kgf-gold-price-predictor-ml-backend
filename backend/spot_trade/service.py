"""Spot trading service"""
import logging
from typing import Dict, Optional
from datetime import datetime
import sys
from pathlib import Path

# Add backend/app to path for imports
backend_app_path = Path(__file__).resolve().parent.parent / "app"
if str(backend_app_path) not in sys.path:
    sys.path.insert(0, str(backend_app_path))

from app.services.market_data_service import MarketDataService
from app.services.exchange_service import ExchangeService
from spot_trade.models import (
    get_or_create_user_balance,
    update_user_balance,
    create_trade,
    update_trade_status,
    OrderType,
    OrderStatus
)

logger = logging.getLogger(__name__)

# Spread in LKR
SPREAD_LKR = 500.0

# Conversion constants: 1 troy ounce = 31.1035 grams, 1 pawn = 8 grams
TROY_OUNCE_GRAMS = 31.1035
PAWN_GRAMS = 8.0
TROY_OUNCE_TO_PAWN = TROY_OUNCE_GRAMS / PAWN_GRAMS  # 3.8879375
PAWN_TO_TROY_OUNCE = PAWN_GRAMS / TROY_OUNCE_GRAMS  # 0.257205


class SpotTradingService:
    """Service for spot trading operations"""
    
    def __init__(self, market_data_service: MarketDataService, exchange_service: ExchangeService):
        self.market_data_service = market_data_service
        self.exchange_service = exchange_service
    
    def get_current_price(self) -> Dict:
        """Get current gold price in LKR per pawn with spread"""
        try:
            # Get current gold price in USD per troy ounce
            realtime_data = self.market_data_service.get_realtime_price()
            current_price_usd_per_troy_ounce = realtime_data.get('current_price', 0.0)
            
            if current_price_usd_per_troy_ounce <= 0:
                raise ValueError("Invalid gold price")
            
            # Get USD to LKR exchange rate
            exchange_data = self.exchange_service.get_exchange_rate('USD', 'LKR')
            exchange_rate = exchange_data.get('exchange_rate', 300.0)
            
            # Convert USD per troy ounce to LKR per troy ounce
            current_price_lkr_per_troy_ounce = current_price_usd_per_troy_ounce * exchange_rate
            
            # Convert to LKR per pawn (1 troy ounce = 31.1035 grams, 1 pawn = 8 grams)
            # So 1 troy ounce = 31.1035 / 8 = 3.8879375 pawn
            TROY_OUNCE_TO_PAWN = 31.1035 / 8.0  # 3.8879375
            current_price_lkr_per_pawn = current_price_lkr_per_troy_ounce / TROY_OUNCE_TO_PAWN
            
            # Calculate buy and sell prices with spread (in LKR per pawn)
            spread_half = SPREAD_LKR / 2.0
            buy_price_lkr = current_price_lkr_per_pawn + spread_half
            sell_price_lkr = current_price_lkr_per_pawn - spread_half
            
            return {
                "symbol": "XAUUSD",
                "current_price_lkr": round(current_price_lkr_per_pawn, 2),  # LKR per pawn
                "current_price_usd": round(current_price_usd_per_troy_ounce, 2),  # USD per troy ounce (for reference)
                "spread_lkr": SPREAD_LKR,
                "buy_price_lkr": round(buy_price_lkr, 2),  # LKR per pawn
                "sell_price_lkr": round(sell_price_lkr, 2),  # LKR per pawn
                "exchange_rate": round(exchange_rate, 2),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting current price: {e}", exc_info=True)
            raise
    
    def place_buy_order(self, user_id: str, quantity: float) -> Dict:
        """Place a BUY market order (quantity in pawn)"""
        try:
            # Get current price (already in LKR per pawn)
            price_data = self.get_current_price()
            buy_price_lkr = price_data['buy_price_lkr']
            
            # Calculate total value (quantity is in pawn, price is LKR per pawn)
            total_value_lkr = quantity * buy_price_lkr
            
            # Get or create user balance
            balance = get_or_create_user_balance(user_id)
            lkr_balance = balance['lkr_balance']
            
            # Validate sufficient balance
            if lkr_balance < total_value_lkr:
                raise ValueError(
                    f"Insufficient LKR balance. Required: {total_value_lkr:.2f} LKR, Available: {lkr_balance:.2f} LKR"
                )
            
            # Validate minimum order size (optional - can be configured)
            min_order_value = 1000.0  # Minimum 1000 LKR
            if total_value_lkr < min_order_value:
                raise ValueError(f"Minimum order value is {min_order_value} LKR")
            
            # Convert quantity from pawn to troy ounces for storage (backend stores in troy ounces)
            quantity_troy_ounces = quantity * PAWN_TO_TROY_OUNCE
            
            # Create trade record (store quantity in troy ounces for consistency)
            trade_id = create_trade(
                user_id=user_id,
                order_type=OrderType.BUY,
                quantity=quantity_troy_ounces,
                price=buy_price_lkr,  # Price is LKR per pawn
                total_value=total_value_lkr,
                status=OrderStatus.PENDING
            )
            
            if not trade_id:
                raise ValueError("Failed to create trade record")
            
            try:
                # Execute trade: Deduct LKR, Add Gold (gold_balance is stored in troy ounces)
                new_lkr_balance = lkr_balance - total_value_lkr
                new_gold_balance = balance['gold_balance'] + quantity_troy_ounces
                
                # Update balances atomically
                update_user_balance(
                    user_id=user_id,
                    lkr_balance=new_lkr_balance,
                    gold_balance=new_gold_balance
                )
                
                # Update trade status to completed
                update_trade_status(trade_id, OrderStatus.COMPLETED)
                
                logger.info(f"✅ BUY order executed: User {user_id}, Quantity: {quantity} pawn, Price: {buy_price_lkr} LKR/pawn")
                
                return {
                    "trade_id": trade_id,
                    "user_id": user_id,
                    "order_type": OrderType.BUY,
                    "quantity": quantity,  # Return in pawn for frontend
                    "price": buy_price_lkr,  # LKR per pawn
                    "total_value": total_value_lkr,
                    "status": OrderStatus.COMPLETED,
                    "message": f"Successfully bought {quantity:.4f} pawn of gold at {buy_price_lkr:.2f} LKR per pawn",
                    "created_at": datetime.now().isoformat()
                }
            except Exception as e:
                # Rollback: Mark trade as failed
                update_trade_status(trade_id, OrderStatus.FAILED)
                logger.error(f"Error executing BUY order: {e}", exc_info=True)
                raise ValueError(f"Failed to execute BUY order: {str(e)}")
        
        except ValueError as e:
            raise
        except Exception as e:
            logger.error(f"Error placing BUY order: {e}", exc_info=True)
            raise ValueError(f"Failed to place BUY order: {str(e)}")
    
    def place_sell_order(self, user_id: str, quantity: float) -> Dict:
        """Place a SELL market order (quantity in pawn)"""
        try:
            # Get current price (already in LKR per pawn)
            price_data = self.get_current_price()
            sell_price_lkr = price_data['sell_price_lkr']
            
            # Calculate total value (quantity is in pawn, price is LKR per pawn)
            total_value_lkr = quantity * sell_price_lkr
            
            # Get or create user balance (gold_balance is stored in troy ounces)
            balance = get_or_create_user_balance(user_id)
            gold_balance_troy_ounces = balance['gold_balance']
            
            # Convert quantity from pawn to troy ounces for validation
            quantity_troy_ounces = quantity * PAWN_TO_TROY_OUNCE
            
            # Validate sufficient gold balance
            if gold_balance_troy_ounces < quantity_troy_ounces:
                gold_balance_pawn = gold_balance_troy_ounces * TROY_OUNCE_TO_PAWN
                raise ValueError(
                    f"Insufficient gold balance. Required: {quantity:.4f} pawn, Available: {gold_balance_pawn:.4f} pawn"
                )
            
            # Validate minimum order size
            min_order_value = 1000.0  # Minimum 1000 LKR
            if total_value_lkr < min_order_value:
                raise ValueError(f"Minimum order value is {min_order_value} LKR")
            
            # Create trade record (store quantity in troy ounces for consistency)
            trade_id = create_trade(
                user_id=user_id,
                order_type=OrderType.SELL,
                quantity=quantity_troy_ounces,
                price=sell_price_lkr,  # Price is LKR per pawn
                total_value=total_value_lkr,
                status=OrderStatus.PENDING
            )
            
            if not trade_id:
                raise ValueError("Failed to create trade record")
            
            try:
                # Execute trade: Deduct Gold (in troy ounces), Add LKR
                new_gold_balance = gold_balance_troy_ounces - quantity_troy_ounces
                new_lkr_balance = balance['lkr_balance'] + total_value_lkr
                
                # Update balances atomically
                update_user_balance(
                    user_id=user_id,
                    lkr_balance=new_lkr_balance,
                    gold_balance=new_gold_balance
                )
                
                # Update trade status to completed
                update_trade_status(trade_id, OrderStatus.COMPLETED)
                
                logger.info(f"✅ SELL order executed: User {user_id}, Quantity: {quantity} pawn, Price: {sell_price_lkr} LKR/pawn")
                
                return {
                    "trade_id": trade_id,
                    "user_id": user_id,
                    "order_type": OrderType.SELL,
                    "quantity": quantity,  # Return in pawn for frontend
                    "price": sell_price_lkr,  # LKR per pawn
                    "total_value": total_value_lkr,
                    "status": OrderStatus.COMPLETED,
                    "message": f"Successfully sold {quantity:.4f} pawn of gold at {sell_price_lkr:.2f} LKR per pawn",
                    "created_at": datetime.now().isoformat()
                }
            except Exception as e:
                # Rollback: Mark trade as failed
                update_trade_status(trade_id, OrderStatus.FAILED)
                logger.error(f"Error executing SELL order: {e}", exc_info=True)
                raise ValueError(f"Failed to execute SELL order: {str(e)}")
        
        except ValueError as e:
            raise
        except Exception as e:
            logger.error(f"Error placing SELL order: {e}", exc_info=True)
            raise ValueError(f"Failed to place SELL order: {str(e)}")
    
    def get_user_balance(self, user_id: str) -> Dict:
        """Get user balance with total portfolio value (gold_balance returned in pawn)"""
        try:
            balance = get_or_create_user_balance(user_id)
            
            # Get current gold price (in LKR per pawn)
            price_data = self.get_current_price()
            current_price_lkr_per_pawn = price_data['current_price_lkr']
            
            # Convert gold balance from troy ounces to pawn for frontend display
            gold_balance_pawn = balance['gold_balance'] * TROY_OUNCE_TO_PAWN
            
            # Calculate total portfolio value (gold_balance in troy ounces, price in LKR per pawn)
            # So: gold_balance_troy_ounces * TROY_OUNCE_TO_PAWN * price_per_pawn
            gold_value_lkr = gold_balance_pawn * current_price_lkr_per_pawn
            total_value_lkr = balance['lkr_balance'] + gold_value_lkr
            
            return {
                "user_id": user_id,
                "lkr_balance": round(balance['lkr_balance'], 2),
                "gold_balance": round(gold_balance_pawn, 8),  # Return in pawn
                "total_value_lkr": round(total_value_lkr, 2)
            }
        except Exception as e:
            logger.error(f"Error getting user balance: {e}", exc_info=True)
            raise ValueError(f"Failed to get user balance: {str(e)}")

