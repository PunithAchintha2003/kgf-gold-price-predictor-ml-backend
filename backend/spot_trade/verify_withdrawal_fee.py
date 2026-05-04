"""Verification script for withdrawal fee implementation"""
import logging
import sys
from pathlib import Path

# Add backend directory to path for imports
backend_path = Path(__file__).resolve().parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from app.core.database import get_db_connection, get_db_type

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def verify_fee_column():
    """Verify that the fee column exists in wallet_transactions table"""
    db_type = get_db_type()
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            if db_type == "postgresql":
                cursor.execute("""
                    SELECT column_name, data_type, column_default
                    FROM information_schema.columns 
                    WHERE table_name='wallet_transactions' AND column_name='fee'
                """)
                result = cursor.fetchone()
                if result:
                    logger.info(f"✅ Fee column exists in PostgreSQL")
                    logger.info(f"   Column: {result[0]}, Type: {result[1]}, Default: {result[2]}")
                    return True
                else:
                    logger.error("❌ Fee column NOT found in PostgreSQL")
                    return False
            else:
                cursor.execute("PRAGMA table_info(wallet_transactions)")
                columns = {row[1]: row for row in cursor.fetchall()}
                
                if 'fee' in columns:
                    fee_col = columns['fee']
                    logger.info(f"✅ Fee column exists in SQLite")
                    logger.info(f"   Column: {fee_col[1]}, Type: {fee_col[2]}, Default: {fee_col[4]}")
                    return True
                else:
                    logger.error("❌ Fee column NOT found in SQLite")
                    return False
                    
    except Exception as e:
        logger.error(f"❌ Error checking fee column: {e}", exc_info=True)
        return False


def verify_withdrawal_transactions():
    """Check if any withdrawal transactions have fee values"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            db_type = get_db_type()
            
            placeholder = "%s" if db_type == "postgresql" else "?"
            
            cursor.execute(f"""
                SELECT COUNT(*) FROM wallet_transactions 
                WHERE transaction_type = {placeholder}
            """, ('WITHDRAWAL',))
            
            total_withdrawals = cursor.fetchone()[0]
            
            cursor.execute(f"""
                SELECT COUNT(*) FROM wallet_transactions 
                WHERE transaction_type = {placeholder} AND fee > 0
            """, ('WITHDRAWAL',))
            
            withdrawals_with_fee = cursor.fetchone()[0]
            
            logger.info(f"📊 Withdrawal Statistics:")
            logger.info(f"   Total withdrawals: {total_withdrawals}")
            logger.info(f"   Withdrawals with fee > 0: {withdrawals_with_fee}")
            logger.info(f"   Withdrawals with fee = 0: {total_withdrawals - withdrawals_with_fee}")
            
            if total_withdrawals > 0:
                cursor.execute(f"""
                    SELECT id, user_id, amount, fee, status
                    FROM wallet_transactions 
                    WHERE transaction_type = {placeholder}
                    ORDER BY created_at DESC
                    LIMIT 5
                """, ('WITHDRAWAL',))
                
                logger.info(f"\n📝 Recent withdrawals (last 5):")
                for row in cursor.fetchall():
                    logger.info(f"   ID: {row[0]}, User: {row[1]}, Amount: {row[2]}, Fee: {row[3]}, Status: {row[4]}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Error checking withdrawal transactions: {e}", exc_info=True)
        return False


def main():
    """Run all verification checks"""
    logger.info("=" * 60)
    logger.info("Withdrawal Fee Implementation Verification")
    logger.info("=" * 60)
    
    logger.info("\n1. Checking fee column in database...")
    fee_col_ok = verify_fee_column()
    
    logger.info("\n2. Checking withdrawal transactions...")
    withdrawals_ok = verify_withdrawal_transactions()
    
    logger.info("\n" + "=" * 60)
    if fee_col_ok and withdrawals_ok:
        logger.info("✅ All verification checks passed!")
        logger.info("\nNext steps:")
        logger.info("1. Restart the backend server")
        logger.info("2. Test withdrawal flow in frontend")
        logger.info("3. Check admin panel for fee display")
    else:
        logger.error("❌ Some verification checks failed")
        logger.error("\nPlease run the migration:")
        logger.error("  cd backend/spot_trade/migrations")
        logger.error("  python add_fee_column.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
