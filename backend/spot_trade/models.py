"""Database models for spot trading"""
import logging
from datetime import datetime
from typing import Optional
from enum import Enum
import sys
from pathlib import Path

# Add backend/app to path for imports
backend_app_path = Path(__file__).resolve().parent.parent / "app"
if str(backend_app_path) not in sys.path:
    sys.path.insert(0, str(backend_app_path))

from app.core.database import get_db_connection, get_db_type
from app.core.config import settings

logger = logging.getLogger(__name__)


class OrderType(str, Enum):
    """Order type enumeration"""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    """Order status enumeration"""
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


def init_spot_trade_tables():
    """Initialize spot trading database tables"""
    db_type = get_db_type()
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        if db_type == "postgresql":
            # PostgreSQL table creation
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_balances (
                    user_id VARCHAR(255) PRIMARY KEY,
                    lkr_balance DECIMAL(20, 2) NOT NULL DEFAULT 0.0,
                    gold_balance DECIMAL(20, 8) NOT NULL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS spot_trades (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(255) NOT NULL,
                    order_type VARCHAR(10) NOT NULL CHECK (order_type IN ('BUY', 'SELL')),
                    quantity DECIMAL(20, 8) NOT NULL CHECK (quantity > 0),
                    price DECIMAL(20, 2) NOT NULL CHECK (price > 0),
                    total_value DECIMAL(20, 2) NOT NULL CHECK (total_value > 0),
                    status VARCHAR(20) NOT NULL DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'COMPLETED', 'FAILED', 'CANCELLED')),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES user_balances(user_id) ON DELETE CASCADE
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_spot_trades_user_id ON spot_trades(user_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_spot_trades_created_at ON spot_trades(created_at DESC)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_spot_trades_status ON spot_trades(status)
            """)
        else:
            # SQLite table creation
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_balances (
                    user_id TEXT PRIMARY KEY,
                    lkr_balance REAL NOT NULL DEFAULT 0.0,
                    gold_balance REAL NOT NULL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS spot_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    order_type TEXT NOT NULL CHECK (order_type IN ('BUY', 'SELL')),
                    quantity REAL NOT NULL CHECK (quantity > 0),
                    price REAL NOT NULL CHECK (price > 0),
                    total_value REAL NOT NULL CHECK (total_value > 0),
                    status TEXT NOT NULL DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'COMPLETED', 'FAILED', 'CANCELLED')),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES user_balances(user_id) ON DELETE CASCADE
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_spot_trades_user_id ON spot_trades(user_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_spot_trades_created_at ON spot_trades(created_at DESC)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_spot_trades_status ON spot_trades(status)
            """)
        
        conn.commit()
        logger.debug("✅ Spot trading tables initialized")


def get_or_create_user_balance(user_id: str) -> dict:
    """Get or create user balance record"""
    db_type = get_db_type()
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Try to get existing balance
            if db_type == "postgresql":
                cursor.execute(
                    "SELECT user_id, lkr_balance, gold_balance, created_at, updated_at FROM user_balances WHERE user_id = %s",
                    (user_id,)
                )
            else:
                cursor.execute(
                    "SELECT user_id, lkr_balance, gold_balance, created_at, updated_at FROM user_balances WHERE user_id = ?",
                    (user_id,)
                )
            row = cursor.fetchone()
            
            if row:
                return {
                    "user_id": row[0],
                    "lkr_balance": float(row[1]),
                    "gold_balance": float(row[2]),
                    "created_at": row[3],
                    "updated_at": row[4]
                }
            
            # Create new balance record with default values
            now = datetime.now()
            
            if db_type == "postgresql":
                cursor.execute(
                    """INSERT INTO user_balances (user_id, lkr_balance, gold_balance, created_at, updated_at)
                       VALUES (%s, %s, %s, %s, %s)
                       ON CONFLICT (user_id) DO NOTHING
                       RETURNING user_id, lkr_balance, gold_balance, created_at, updated_at""",
                    (user_id, 0.0, 0.0, now, now)
                )
                row = cursor.fetchone()
                if row:
                    conn.commit()
                    return {
                        "user_id": row[0],
                        "lkr_balance": float(row[1]),
                        "gold_balance": float(row[2]),
                        "created_at": row[3],
                        "updated_at": row[4]
                    }
            else:
                cursor.execute(
                    """INSERT OR IGNORE INTO user_balances (user_id, lkr_balance, gold_balance, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (user_id, 0.0, 0.0, now, now)
                )
                conn.commit()
                
                # Fetch the created record
                cursor.execute(
                    "SELECT user_id, lkr_balance, gold_balance, created_at, updated_at FROM user_balances WHERE user_id = ?",
                    (user_id,)
                )
                row = cursor.fetchone()
                if row:
                    return {
                        "user_id": row[0],
                        "lkr_balance": float(row[1]),
                        "gold_balance": float(row[2]),
                        "created_at": row[3],
                        "updated_at": row[4]
                    }
            
            conn.commit()
            return {
                "user_id": user_id,
                "lkr_balance": 0.0,
                "gold_balance": 0.0,
                "created_at": now,
                "updated_at": now
            }
    except Exception as e:
        logger.error(f"Error in get_or_create_user_balance: {e}", exc_info=True)
        # Return default balance on error
        now = datetime.now()
        return {
            "user_id": user_id,
            "lkr_balance": 0.0,
            "gold_balance": 0.0,
            "created_at": now,
            "updated_at": now
        }


def update_user_balance(user_id: str, lkr_balance: Optional[float] = None, gold_balance: Optional[float] = None) -> bool:
    """Update user balance"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        db_type = get_db_type()
        now = datetime.now()
        
        updates = []
        params = []
        
        if lkr_balance is not None:
            updates.append("lkr_balance = ?" if db_type != "postgresql" else "lkr_balance = %s")
            params.append(lkr_balance)
        
        if gold_balance is not None:
            updates.append("gold_balance = ?" if db_type != "postgresql" else "gold_balance = %s")
            params.append(gold_balance)
        
        if not updates:
            return False
        
        updates.append("updated_at = ?" if db_type != "postgresql" else "updated_at = %s")
        params.append(now)
        params.append(user_id)
        
        if db_type == "postgresql":
            query = f"UPDATE user_balances SET {', '.join(updates)} WHERE user_id = %s"
        else:
            query = f"UPDATE user_balances SET {', '.join(updates)} WHERE user_id = ?"
        
        cursor.execute(query, params)
        conn.commit()
        return cursor.rowcount > 0


def create_trade(user_id: str, order_type: str, quantity: float, price: float, total_value: float, status: str = "PENDING") -> Optional[int]:
    """Create a new trade record"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        db_type = get_db_type()
        now = datetime.now()
        
        if db_type == "postgresql":
            cursor.execute(
                """INSERT INTO spot_trades (user_id, order_type, quantity, price, total_value, status, created_at, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                   RETURNING id""",
                (user_id, order_type, quantity, price, total_value, status, now, now)
            )
            row = cursor.fetchone()
            trade_id = row[0] if row else None
        else:
            cursor.execute(
                """INSERT INTO spot_trades (user_id, order_type, quantity, price, total_value, status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (user_id, order_type, quantity, price, total_value, status, now, now)
            )
            trade_id = cursor.lastrowid
        
        conn.commit()
        return trade_id


def update_trade_status(trade_id: int, status: str) -> bool:
    """Update trade status"""
    db_type = get_db_type()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        now = datetime.now()
        
        if db_type == "postgresql":
            cursor.execute(
                "UPDATE spot_trades SET status = %s, updated_at = %s WHERE id = %s",
                (status, now, trade_id)
            )
        else:
            cursor.execute(
                "UPDATE spot_trades SET status = ?, updated_at = ? WHERE id = ?",
                (status, now, trade_id)
            )
        
        conn.commit()
        return cursor.rowcount > 0


def get_user_trades(user_id: str, limit: int = 100, offset: int = 0) -> list:
    """Get user's trade history (quantities converted to pawn for frontend)"""
    # Conversion: 1 troy ounce = 31.1035 / 8 = 3.8879375 pawn
    TROY_OUNCE_TO_PAWN = 31.1035 / 8.0
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        db_type = get_db_type()
        
        if db_type == "postgresql":
            cursor.execute(
                """SELECT id, user_id, order_type, quantity, price, total_value, status, created_at, updated_at
                   FROM spot_trades
                   WHERE user_id = %s
                   ORDER BY created_at DESC
                   LIMIT %s OFFSET %s""",
                (user_id, limit, offset)
            )
        else:
            cursor.execute(
                """SELECT id, user_id, order_type, quantity, price, total_value, status, created_at, updated_at
                   FROM spot_trades
                   WHERE user_id = ?
                   ORDER BY created_at DESC
                   LIMIT ? OFFSET ?""",
                (user_id, limit, offset)
            )
        
        rows = cursor.fetchall()
        trades = []
        
        for row in rows:
            # Convert quantity from troy ounces to pawn
            quantity_troy_ounces = float(row[3])
            quantity_pawn = quantity_troy_ounces * TROY_OUNCE_TO_PAWN
            
            trades.append({
                "id": row[0],
                "user_id": row[1],
                "order_type": row[2],
                "quantity": quantity_pawn,  # Return in pawn
                "price": float(row[4]),  # Price is already in LKR per pawn
                "total_value": float(row[5]),
                "status": row[6],
                "created_at": row[7].isoformat() if hasattr(row[7], 'isoformat') else str(row[7]),
                "updated_at": row[8].isoformat() if hasattr(row[8], 'isoformat') else str(row[8])
            })
        
        return trades


def get_open_orders(user_id: str) -> list:
    """Get user's open (pending) orders (quantities converted to pawn for frontend)"""
    db_type = get_db_type()
    # Conversion: 1 troy ounce = 31.1035 / 8 = 3.8879375 pawn
    TROY_OUNCE_TO_PAWN = 31.1035 / 8.0
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        if db_type == "postgresql":
            cursor.execute(
                """SELECT id, user_id, order_type, quantity, price, total_value, status, created_at, updated_at
                   FROM spot_trades
                   WHERE user_id = %s AND status = 'PENDING'
                   ORDER BY created_at DESC""",
                (user_id,)
            )
        else:
            cursor.execute(
                """SELECT id, user_id, order_type, quantity, price, total_value, status, created_at, updated_at
                   FROM spot_trades
                   WHERE user_id = ? AND status = 'PENDING'
                   ORDER BY created_at DESC""",
                (user_id,)
            )
        
        rows = cursor.fetchall()
        orders = []
        
        for row in rows:
            # Convert quantity from troy ounces to pawn
            quantity_troy_ounces = float(row[3])
            quantity_pawn = quantity_troy_ounces * TROY_OUNCE_TO_PAWN
            
            orders.append({
                "id": row[0],
                "user_id": row[1],
                "order_type": row[2],
                "quantity": quantity_pawn,  # Return in pawn
                "price": float(row[4]),  # Price is already in LKR per pawn
                "total_value": float(row[5]),
                "status": row[6],
                "created_at": row[7].isoformat() if hasattr(row[7], 'isoformat') else str(row[7]),
                "updated_at": row[8].isoformat() if hasattr(row[8], 'isoformat') else str(row[8])
            })
        
        return orders

