"""Migration to add fee column to wallet_transactions table"""
import logging
import sys
from pathlib import Path

# Add backend directory to path for imports
backend_path = Path(__file__).resolve().parent.parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from app.core.database import get_db_connection, get_db_type

logger = logging.getLogger(__name__)


def run_migration():
    """Add fee column to wallet_transactions table"""
    db_type = get_db_type()
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # First, check if the wallet_transactions table exists
            if db_type == "postgresql":
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'wallet_transactions'
                    )
                """)
                table_exists = cursor.fetchone()[0]
            else:
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='wallet_transactions'
                """)
                table_exists = cursor.fetchone() is not None
            
            if not table_exists:
                logger.info("ℹ️  wallet_transactions table doesn't exist yet")
                logger.info("   This is a fresh database. The table will be created with the fee column")
                logger.info("   when you start the backend server (it will call init_spot_trade_tables())")
                logger.info("✅ No migration needed - schema already includes fee column")
                return True
            
            # Table exists, check if fee column already exists
            if db_type == "postgresql":
                cursor.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name='wallet_transactions' AND column_name='fee'
                """)
                fee_exists = cursor.fetchone() is not None
            else:
                cursor.execute("PRAGMA table_info(wallet_transactions)")
                columns = [row[1] for row in cursor.fetchall()]
                fee_exists = 'fee' in columns
            
            if fee_exists:
                logger.info("✅ Fee column already exists in wallet_transactions table")
                return True
            
            # Add fee column to existing table
            logger.info("📝 Adding fee column to existing wallet_transactions table...")
            if db_type == "postgresql":
                cursor.execute("""
                    ALTER TABLE wallet_transactions 
                    ADD COLUMN fee DECIMAL(20, 2) DEFAULT 0.0
                """)
            else:
                cursor.execute("""
                    ALTER TABLE wallet_transactions 
                    ADD COLUMN fee REAL DEFAULT 0.0
                """)
            
            conn.commit()
            logger.info("✅ Successfully added fee column to wallet_transactions table")
            return True
            
    except Exception as e:
        logger.error(f"❌ Error adding fee column: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = run_migration()
    sys.exit(0 if success else 1)
