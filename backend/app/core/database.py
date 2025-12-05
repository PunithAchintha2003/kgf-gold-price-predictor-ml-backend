"""Database connection and initialization"""
import sqlite3
import os
from contextlib import contextmanager
from typing import Optional
import logging

# PostgreSQL support
try:
    import psycopg2
    from psycopg2 import pool
    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False
    psycopg2 = None

from .config import settings

logger = logging.getLogger(__name__)

# PostgreSQL connection pool
_postgresql_pool: Optional[pool.SimpleConnectionPool] = None


def init_postgresql_pool() -> bool:
    """Initialize PostgreSQL connection pool"""
    global _postgresql_pool
    if not POSTGRESQL_AVAILABLE:
        logger.warning(
            "PostgreSQL library not available, falling back to SQLite")
        return False

    try:
        _postgresql_pool = psycopg2.pool.SimpleConnectionPool(
            1, 20,  # min and max connections
            host=settings.postgresql_host,
            port=settings.postgresql_port,
            database=settings.postgresql_database,
            user=settings.postgresql_user,
            password=settings.postgresql_password
        )
        if _postgresql_pool:
            logger.info(
                f"PostgreSQL connection pool initialized: {settings.postgresql_database}")
            return True
    except Exception as e:
        logger.error(f"Failed to initialize PostgreSQL pool: {e}")
        logger.warning("Falling back to SQLite")
        return False
    return False


def get_db_type() -> str:
    """Get current database type"""
    use_postgresql = settings.use_postgresql and POSTGRESQL_AVAILABLE and _postgresql_pool is not None
    return "postgresql" if use_postgresql else "sqlite"


def get_date_function(days_offset: int = 0) -> str:
    """Get database-appropriate date function"""
    db_type = get_db_type()
    if db_type == "postgresql":
        if days_offset == 0:
            return "CURRENT_DATE"
        elif days_offset < 0:
            return f"CURRENT_DATE - INTERVAL '{abs(days_offset)} days'"
        else:
            return f"CURRENT_DATE + INTERVAL '{days_offset} days'"
    else:
        # SQLite
        if days_offset == 0:
            return "date('now')"
        else:
            return f"date('now', '{days_offset:+d} days')"


def _is_connection_alive(conn) -> bool:
    """Check if a PostgreSQL connection is still alive"""
    if not conn or not POSTGRESQL_AVAILABLE:
        return False
    try:
        # Try to execute a simple query to check if connection is alive
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        cursor.close()
        return True
    except Exception as e:
        # Check for PostgreSQL-specific errors
        if POSTGRESQL_AVAILABLE:
            if isinstance(e, (psycopg2.OperationalError, psycopg2.InterfaceError)):
                return False
        # Other exceptions might indicate connection issues
        if isinstance(e, AttributeError):
            return False
        # For other exceptions, assume connection might be alive but query failed
        # Log for debugging
        logger.debug(f"Connection health check exception: {e}")
        return False


@contextmanager
def get_db_connection(db_path: Optional[str] = None, max_retries: int = 3):
    """Context manager for database connections - Supports PostgreSQL and SQLite

    Args:
        db_path: Optional path for SQLite database
        max_retries: Maximum number of retries for PostgreSQL connections
    """
    conn = None
    use_postgresql = settings.use_postgresql and POSTGRESQL_AVAILABLE and _postgresql_pool is not None

    try:
        if use_postgresql:
            # Use PostgreSQL with retry logic
            for attempt in range(max_retries):
                try:
                    conn = _postgresql_pool.getconn()
                    if conn:
                        # Check if connection is alive before using it
                        if not _is_connection_alive(conn):
                            logger.warning(
                                f"Connection from pool is dead, attempt {attempt + 1}/{max_retries}")
                            try:
                                conn.close()
                            except:
                                pass
                            conn = None
                            if attempt < max_retries - 1:
                                continue
                            else:
                                raise Exception(
                                    "Failed to get a valid connection from pool after retries")
                        yield conn
                        break
                    else:
                        if attempt < max_retries - 1:
                            logger.warning(
                                f"Failed to get connection from pool, attempt {attempt + 1}/{max_retries}")
                            continue
                        else:
                            raise Exception(
                                "Failed to get connection from pool")
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Error getting connection (attempt {attempt + 1}/{max_retries}): {e}")
                        if conn:
                            try:
                                conn.close()
                            except:
                                pass
                            conn = None
                        continue
                    else:
                        raise
        else:
            # Fallback to SQLite
            if db_path is None:
                db_path = settings.db_path
            conn = sqlite3.connect(db_path, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL")  # Better performance
            conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes
            conn.execute("PRAGMA cache_size=10000")  # Larger cache
            yield conn
    except Exception as e:
        if conn:
            try:
                conn.rollback()
            except:
                pass
        # Log the error for debugging
        logger.debug(f"Database connection error: {e}", exc_info=True)
        raise e
    finally:
        if conn:
            if use_postgresql:
                try:
                    # Check if connection is still alive before returning to pool
                    if _is_connection_alive(conn):
                        # Return connection to pool (don't close it)
                        _postgresql_pool.putconn(conn)
                    else:
                        # Connection is dead, close it instead of returning to pool
                        logger.warning(
                            "Connection is dead, closing instead of returning to pool")
                        try:
                            conn.close()
                        except:
                            pass
                except Exception as e:
                    logger.error(f"Error returning connection to pool: {e}")
                    # If pool is full or connection is bad, close it
                    try:
                        conn.close()
                    except:
                        pass
            else:
                conn.close()


def init_database():
    """Initialize the main database with predictions table"""
    db_type = get_db_type()

    with get_db_connection() as conn:
        cursor = conn.cursor()

        if db_type == "postgresql":
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    prediction_date DATE NOT NULL,
                    predicted_price DECIMAL(10, 2) NOT NULL,
                    actual_price DECIMAL(10, 2),
                    accuracy_percentage DECIMAL(5, 2),
                    prediction_method VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Clean up duplicate records before adding unique constraint
            try:
                # Delete duplicates, keeping only the most recent record for each date
                cursor.execute('''
                    DELETE FROM predictions p1
                    WHERE p1.id NOT IN (
                        SELECT MAX(p2.id)
                        FROM predictions p2
                        GROUP BY p2.prediction_date
                    )
                ''')
                deleted_count = cursor.rowcount
                if deleted_count > 0:
                    logger.info(
                        f"Cleaned up {deleted_count} duplicate prediction records")
                conn.commit()
            except Exception as e:
                logger.warning(
                    f"Could not clean duplicates (table may be empty or already clean): {e}")
                conn.rollback()

            # Ensure UNIQUE constraint exists on prediction_date
            try:
                cursor.execute('''
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM pg_constraint 
                            WHERE conname = 'predictions_prediction_date_key'
                        ) THEN
                            ALTER TABLE predictions ADD CONSTRAINT predictions_prediction_date_key UNIQUE (prediction_date);
                        END IF;
                    END $$;
                ''')
                logger.info(
                    "Unique constraint on prediction_date verified/created")
            except Exception as e:
                # Check if constraint already exists (different error message)
                if "already exists" in str(e).lower() or "duplicate key" in str(e).lower():
                    logger.debug(f"Unique constraint already exists: {e}")
                else:
                    logger.warning(f"Could not add unique constraint: {e}")
        else:
            # SQLite
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_date DATE NOT NULL UNIQUE,
                    predicted_price REAL NOT NULL,
                    actual_price REAL,
                    accuracy_percentage REAL,
                    prediction_method TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

        # Create indexes for better query performance
        try:
            if db_type == "postgresql":
                # Create indexes for frequently queried columns
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_predictions_date 
                    ON predictions(prediction_date);
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_predictions_created_at 
                    ON predictions(created_at);
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_predictions_date_created 
                    ON predictions(prediction_date, created_at);
                ''')
                logger.debug("PostgreSQL indexes created/verified")
            else:
                # SQLite indexes
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_predictions_date 
                    ON predictions(prediction_date);
                ''')
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_predictions_created_at 
                    ON predictions(created_at);
                ''')
                logger.debug("SQLite indexes created/verified")
            conn.commit()
        except Exception as e:
            logger.warning(f"Could not create indexes: {e}")
            conn.rollback()

        conn.commit()
        logger.info(f"Database initialized successfully: {db_type.upper()}")


def init_backup_database():
    """Initialize the backup database"""
    db_type = get_db_type()

    with get_db_connection(db_path=settings.backup_db_path) as conn:
        cursor = conn.cursor()

        if db_type == "postgresql":
            # For PostgreSQL, backup would be in same database with different table
            # For now, we'll use SQLite for backup
            pass
        else:
            # SQLite backup
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions_backup (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_date DATE NOT NULL,
                    predicted_price REAL NOT NULL,
                    actual_price REAL,
                    accuracy_percentage REAL,
                    prediction_method TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

        conn.commit()
        logger.info("Backup database initialized")


# Export init_postgresql_pool for use in main.py
__all__ = [
    'get_db_connection',
    'get_db_type',
    'get_date_function',
    'init_database',
    'init_backup_database',
    'init_postgresql_pool'
]
