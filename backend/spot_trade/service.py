"""Spot trading service"""
import logging
from typing import Dict, Optional, List
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
    OrderStatus,
    WalletTransactionType,
    WalletTransactionStatus,
    create_wallet_transaction,
    get_wallet_transaction_by_id,
    get_wallet_transaction_by_stripe_session,
    update_wallet_transaction_status,
    get_wallet_transactions,
)

logger = logging.getLogger(__name__)

# Spread in LKR
SPREAD_LKR = 1000.0

# Withdrawal fee in LKR
WITHDRAWAL_FEE = 100.0

# Transaction fee per gram in LKR
TRANSACTION_FEE_PER_GRAM = 1000.0

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
            
            # Calculate transaction fee based on quantity in grams
            quantity_grams = quantity * PAWN_GRAMS  # quantity is in pawn, convert to grams
            transaction_fee = quantity_grams * TRANSACTION_FEE_PER_GRAM
            
            # Get or create user balance
            balance = get_or_create_user_balance(user_id)
            lkr_balance = balance['lkr_balance']
            
            # User needs: trade value + transaction fee
            total_deduction = total_value_lkr + transaction_fee
            
            # Validate sufficient balance (add epsilon tolerance for floating-point precision)
            EPSILON = 0.01  # 1 cent tolerance for LKR comparisons
            if lkr_balance < total_deduction - EPSILON:
                raise ValueError(
                    f"Insufficient LKR balance. Required: {total_deduction:.2f} LKR "
                    f"(Trade: {total_value_lkr:.2f} + Fee: {transaction_fee:.2f}), "
                    f"Available: {lkr_balance:.2f} LKR"
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
                fee=transaction_fee,
                status=OrderStatus.PENDING
            )
            
            if not trade_id:
                raise ValueError("Failed to create trade record")
            
            try:
                # Execute trade: Deduct LKR (trade value + fee), Add Gold (gold_balance is stored in troy ounces)
                new_lkr_balance = lkr_balance - total_deduction
                new_gold_balance = balance['gold_balance'] + quantity_troy_ounces
                
                # Update balances atomically
                update_user_balance(
                    user_id=user_id,
                    lkr_balance=new_lkr_balance,
                    gold_balance=new_gold_balance
                )
                
                # Update trade status to completed
                update_trade_status(trade_id, OrderStatus.COMPLETED)
                
                logger.info(
                    f"✅ BUY order executed: User {user_id}, Quantity: {quantity} pawn (~{quantity * PAWN_GRAMS:.2f} g), "
                    f"Price: {buy_price_lkr} LKR/pawn, Fee: {transaction_fee:.2f} LKR"
                )
                
                return {
                    "trade_id": trade_id,
                    "user_id": user_id,
                    "order_type": OrderType.BUY,
                    "quantity": quantity,  # Return in pawn for frontend
                    "price": buy_price_lkr,  # LKR per pawn
                    "total_value": total_value_lkr,
                    "fee": transaction_fee,
                    "status": OrderStatus.COMPLETED,
                    "message": f"Successfully bought {quantity:.4f} pawn of gold at {buy_price_lkr:.2f} LKR per pawn (Fee: {transaction_fee:.2f} LKR)",
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
            
            # Calculate transaction fee based on quantity in grams
            quantity_grams = quantity * PAWN_GRAMS
            transaction_fee = quantity_grams * TRANSACTION_FEE_PER_GRAM
            
            # Get or create user balance (gold_balance is stored in troy ounces)
            balance = get_or_create_user_balance(user_id)
            gold_balance_troy_ounces = balance['gold_balance']
            
            # Convert quantity from pawn to troy ounces for validation
            quantity_troy_ounces = quantity * PAWN_TO_TROY_OUNCE
            
            # Validate sufficient gold balance (report in grams for clarity)
            # Add small epsilon tolerance for floating-point precision
            EPSILON = 1e-8  # Small tolerance for troy ounce comparisons
            if gold_balance_troy_ounces < quantity_troy_ounces - EPSILON:
                gold_balance_pawn = gold_balance_troy_ounces * TROY_OUNCE_TO_PAWN
                required_grams = quantity * PAWN_GRAMS
                available_grams = gold_balance_pawn * PAWN_GRAMS
                raise ValueError(
                    f"Insufficient gold balance. Required: {required_grams:.2f} grams, Available: {available_grams:.2f} grams"
                )
            
            # Check if final balance after sale and fee deduction will be non-negative
            # User receives sale proceeds and pays fee from that
            final_balance = balance['lkr_balance'] + total_value_lkr - transaction_fee
            # Add small epsilon tolerance for floating-point precision
            EPSILON_LKR = 0.01  # 1 cent tolerance for LKR
            if final_balance < -EPSILON_LKR:
                raise ValueError(
                    f"Insufficient funds. Sale proceeds ({total_value_lkr:.2f} LKR) cannot cover transaction fee ({transaction_fee:.2f} LKR) "
                    f"with current balance ({balance['lkr_balance']:.2f} LKR). "
                    f"Shortfall: {abs(final_balance):.2f} LKR"
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
                fee=transaction_fee,
                status=OrderStatus.PENDING
            )
            
            if not trade_id:
                raise ValueError("Failed to create trade record")
            
            try:
                # Execute trade: Deduct Gold (in troy ounces), Add LKR (sale proceeds), then Deduct Fee
                new_gold_balance = gold_balance_troy_ounces - quantity_troy_ounces
                new_lkr_balance = balance['lkr_balance'] + total_value_lkr - transaction_fee
                
                # Update balances atomically
                update_user_balance(
                    user_id=user_id,
                    lkr_balance=new_lkr_balance,
                    gold_balance=new_gold_balance
                )
                
                # Update trade status to completed
                update_trade_status(trade_id, OrderStatus.COMPLETED)
                
                logger.info(
                    f"✅ SELL order executed: User {user_id}, Quantity: {quantity} pawn (~{quantity * PAWN_GRAMS:.2f} g), "
                    f"Price: {sell_price_lkr} LKR/pawn, Fee: {transaction_fee:.2f} LKR"
                )
                
                return {
                    "trade_id": trade_id,
                    "user_id": user_id,
                    "order_type": OrderType.SELL,
                    "quantity": quantity,  # Return in pawn for frontend
                    "price": sell_price_lkr,  # LKR per pawn
                    "total_value": total_value_lkr,
                    "fee": transaction_fee,
                    "status": OrderStatus.COMPLETED,
                    "message": f"Successfully sold {quantity * PAWN_GRAMS:.2f} grams ({quantity:.4f} pawn) of gold at {sell_price_lkr:.2f} LKR per pawn (Fee: {transaction_fee:.2f} LKR)",
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

    def create_deposit_checkout(self, user_id: str, amount: float, stripe_session_id: str) -> Dict:
        """Create pending deposit transaction before Stripe checkout"""
        if amount < 5000:
            raise ValueError("Minimum deposit amount is LKR 5,000")
        get_or_create_user_balance(user_id)
        tx_id = create_wallet_transaction(
            user_id=user_id,
            transaction_type=WalletTransactionType.DEPOSIT,
            amount=amount,
            status=WalletTransactionStatus.PENDING,
            payment_method="STRIPE",
            stripe_session_id=stripe_session_id,
            notes="Waiting for Stripe payment confirmation",
        )
        if not tx_id:
            raise ValueError("Failed to create deposit transaction")
        return {"transaction_id": tx_id, "status": WalletTransactionStatus.PENDING}

    def complete_deposit_by_stripe_session(self, stripe_session_id: str) -> Dict:
        """Mark stripe deposit completed and credit wallet once"""
        tx = get_wallet_transaction_by_stripe_session(stripe_session_id)
        if not tx:
            raise ValueError("Deposit transaction not found")
        if tx["status"] == WalletTransactionStatus.COMPLETED:
            return tx
        if tx["transaction_type"] != WalletTransactionType.DEPOSIT:
            raise ValueError("Invalid transaction type for Stripe completion")

        balance = get_or_create_user_balance(tx["user_id"])
        new_lkr_balance = balance["lkr_balance"] + tx["amount"]
        update_user_balance(tx["user_id"], lkr_balance=new_lkr_balance)
        update_wallet_transaction_status(tx["id"], WalletTransactionStatus.COMPLETED)
        return get_wallet_transaction_by_id(tx["id"]) or tx

    def confirm_deposit_for_user(self, user_id: str, stripe_session_id: str, payment_status: str) -> Dict:
        """Confirm a Stripe deposit for the authenticated user."""
        tx = get_wallet_transaction_by_stripe_session(stripe_session_id)
        if not tx:
            raise ValueError("Deposit transaction not found")
        if tx["user_id"] != user_id:
            raise ValueError("Deposit transaction does not belong to the current user")
        if tx["status"] == WalletTransactionStatus.COMPLETED:
            return tx
        if payment_status.lower() != "paid":
            raise ValueError("Payment is not completed yet")
        return self.complete_deposit_by_stripe_session(stripe_session_id)

    def request_withdrawal(self, user_id: str, amount: float, bank_name: str, bank_account_number: str, bank_account_name: str) -> Dict:
        """Create pending withdrawal and reserve balance immediately (amount + fee)"""
        if amount <= 0:
            raise ValueError("Withdrawal amount must be greater than zero")
        
        balance = get_or_create_user_balance(user_id)
        
        # Calculate total deduction (withdrawal amount + fee)
        total_deduction = amount + WITHDRAWAL_FEE
        
        # Validate sufficient balance for amount + fee
        if total_deduction > balance["lkr_balance"]:
            raise ValueError(f"Insufficient balance. Need {total_deduction:.2f} LKR (withdrawal amount: {amount:.2f} + fee: {WITHDRAWAL_FEE:.2f})")
        
        # Create withdrawal transaction with fee
        tx_id = create_wallet_transaction(
            user_id=user_id,
            transaction_type=WalletTransactionType.WITHDRAWAL,
            amount=amount,
            fee=WITHDRAWAL_FEE,
            status=WalletTransactionStatus.PENDING,
            payment_method="BANK_TRANSFER",
            bank_name=bank_name,
            bank_account_number=bank_account_number,
            bank_account_name=bank_account_name,
            notes="Pending admin approval. Processing can take up to 3 days.",
        )
        if not tx_id:
            raise ValueError("Failed to create withdrawal request")
        
        # Deduct both amount and fee from user's LKR balance
        update_user_balance(user_id, lkr_balance=balance["lkr_balance"] - total_deduction)
        
        return get_wallet_transaction_by_id(tx_id) or {"id": tx_id, "status": WalletTransactionStatus.PENDING}

    def approve_withdrawal(self, transaction_id: int, admin_user_id: str, approve: bool, notes: Optional[str] = None) -> Dict:
        """Approve or reject pending withdrawal"""
        tx = get_wallet_transaction_by_id(transaction_id)
        if not tx:
            raise ValueError("Withdrawal transaction not found")
        if tx["transaction_type"] != WalletTransactionType.WITHDRAWAL:
            raise ValueError("Transaction is not a withdrawal")
        if tx["status"] != WalletTransactionStatus.PENDING:
            raise ValueError(f"Only pending withdrawals can be processed. Current status: {tx['status']}")

        if approve:
            update_wallet_transaction_status(transaction_id, WalletTransactionStatus.APPROVED, approved_by=admin_user_id, notes=notes)
        else:
            # Refund both amount and fee on rejection
            balance = get_or_create_user_balance(tx["user_id"])
            refund_total = tx["amount"] + tx["fee"]
            update_user_balance(tx["user_id"], lkr_balance=balance["lkr_balance"] + refund_total)
            update_wallet_transaction_status(transaction_id, WalletTransactionStatus.REJECTED, approved_by=admin_user_id, notes=notes)

        return get_wallet_transaction_by_id(transaction_id) or tx

    def get_wallet_transactions_for_user(self, user_id: str, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Get wallet transactions for user"""
        return get_wallet_transactions(user_id=user_id, limit=limit, offset=offset)

    def get_wallet_transactions_for_admin(
        self,
        status: Optional[str] = None,
        transaction_type: Optional[str] = None,
        limit: int = 200,
        offset: int = 0
    ) -> List[Dict]:
        """Get wallet transactions for admin portal"""
        return get_wallet_transactions(status=status, transaction_type=transaction_type, limit=limit, offset=offset)

