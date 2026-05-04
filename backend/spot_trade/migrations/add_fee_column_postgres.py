"""Direct PostgreSQL migration to add fee column to wallet_transactions table"""
import logging
import sys
import os
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv

# Load .env from project root
env_path = Path(__file__).resolve().parent.parent.parent.parent / ".env"
load_dotenv(env_path)

# Add backend directory to path for imports
backend_path = Path(__file__).resolve().parent.parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

import psycopg2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_pg_connection():
    """Get PostgreSQL connection from environment variables"""
    try:
        conn = psycopg2.connect(
            host=os.getenv('POSTGRESQL_HOST'),
            port=int(os.getenv('POSTGRESQL_PORT', 5432)),
            database=os.getenv('POSTGRESQL_DATABASE'),
            user=os.getenv('POSTGRESQL_USER'),
            password=os.getenv('POSTGRESQL_PASSWORD')
        )
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL: {e}")
        raise


def run_migration():
    """Add fee column to wallet_transactions table in PostgreSQL"""
    try:
        logger.info("Connecting to PostgreSQL...")
        conn = get_pg_connection()
        cursor = conn.cursor()
        
        # Check if wallet_transactions table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'wallet_transactions'
            )
        """)
        table_exists = cursor.fetchone()[0]
        
        if not table_exists:
            logger.info("ℹ️  wallet_transactions table doesn't exist yet")
            logger.info("   The table will be created with the fee column when you start the backend")
            logger.info("✅ No migration needed - schema already includes fee column")
            conn.close()
            return True
        
        logger.info("✅ wallet_transactions table found")
        
        # Check if fee column already exists
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='wallet_transactions' AND column_name='fee'
        """)
        fee_exists = cursor.fetchone() is not None
        
        if fee_exists:
            logger.info("✅ Fee column already exists in wallet_transactions table")
            conn.close()
            return True
        
        # Add fee column
        logger.info("📝 Adding fee column to wallet_transactions table...")
        cursor.execute("""
            ALTER TABLE wallet_transactions 
            ADD COLUMN fee DECIMAL(20, 2) DEFAULT 0.0
        """)
        
        conn.commit()
        logger.info("✅ Successfully added fee column to wallet_transactions table")
        
        # Verify the column was added
        cursor.execute("""
            SELECT column_name, data_type, column_default
            FROM information_schema.columns 
            WHERE table_name='wallet_transactions' AND column_name='fee'
        """)
        result = cursor.fetchone()
        if result:
            logger.info(f"   ✓ Column: {result[0]}, Type: {result[1]}, Default: {result[2]}")
        
        conn.close()
        return True
        
    except psycopg2.Error as e:
        logger.error(f"❌ PostgreSQL error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error adding fee column: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("PostgreSQL Migration: Add fee column to wallet_transactions")
    logger.info("=" * 60)
    logger.info(f"Database: {os.getenv('POSTGRESQL_DATABASE')}")
    logger.info(f"Host: {os.getenv('POSTGRESQL_HOST')}")
    logger.info("=" * 60)
    
    success = run_migration()
    
    if success:
        logger.info("\n" + "=" * 60)
        logger.info("✅ Migration completed successfully!")
        logger.info("=" * 60)
        logger.info("\nNext steps:")
        logger.info("1. Restart your backend server")
        logger.info("2. Test withdrawal functionality")
        logger.info("=" * 60)
    
    sys.exit(0 if success else 1)
