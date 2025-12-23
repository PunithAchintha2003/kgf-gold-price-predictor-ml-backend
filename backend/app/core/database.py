"""Database connection and initialization"""
import sqlite3
import os
from contextlib import contextmanager
from typing import Optional
import logging
import threading
import queue

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
        # Build connection parameters with SSL support for cloud databases (e.g., Render, Neon.tech)
        connection_params = {
            'host': settings.postgresql_host,
            'port': settings.postgresql_port,
            'database': settings.postgresql_database,
            'user': settings.postgresql_user,
            'password': settings.postgresql_password,
            # Add connection parameters to prevent stale connections
            'connect_timeout': 10,  # 10 second connection timeout
            'keepalives': 1,  # Enable TCP keepalives
            'keepalives_idle': 30,  # Start keepalives after 30 seconds of idle
            'keepalives_interval': 10,  # Send keepalive every 10 seconds
            'keepalives_count': 3,  # Close connection after 3 failed keepalives
        }

        # Add SSL mode for cloud databases (Render, Neon.tech, Heroku, AWS, etc.)
        # If hostname looks like a cloud database, require SSL
        if ('render.com' in settings.postgresql_host.lower() or
            'amazonaws.com' in settings.postgresql_host.lower() or
            'herokuapp.com' in settings.postgresql_host.lower() or
                'neon.tech' in settings.postgresql_host.lower()):
            connection_params['sslmode'] = 'require'

        # Use smaller pool size to avoid exhausting connections
        # Reduced from 50 to 20 to prevent connection exhaustion
        _postgresql_pool = psycopg2.pool.SimpleConnectionPool(
            2, 20,  # min and max connections
            **connection_params
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
        # Check connection status first (faster than executing a query)
        if hasattr(conn, 'closed') and conn.closed:
            return False
        # Try to execute a simple query to check if connection is alive
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        cursor.close()
        return True
    except Exception as e:
        # Check for PostgreSQL-specific errors
        if POSTGRESQL_AVAILABLE:
            if isinstance(e, (psycopg2.OperationalError, psycopg2.InterfaceError, 
                            psycopg2.DatabaseError, psycopg2.ProgrammingError)):
                return False
        # Other exceptions might indicate connection issues
        if isinstance(e, AttributeError):
            return False
        # For other exceptions, assume connection might be alive but query failed
        # Log for debugging
        logger.debug(f"Connection health check exception: {e}")
        return False


def _get_connection_with_timeout(pool_instance, timeout: float = 5.0):
    """Get a connection from the pool with a timeout using threading
    
    Args:
        pool_instance: The connection pool instance
        timeout: Maximum time to wait for a connection in seconds
        
    Returns:
        Connection object or None if timeout occurred
        
    Raises:
        Exception: If pool error occurs (not timeout)
    """
    if pool_instance is None:
        raise Exception("Connection pool is not initialized")
    
    result_queue = queue.Queue()
    exception_queue = queue.Queue()
    
    def get_conn():
        try:
            conn = pool_instance.getconn()
            result_queue.put(conn)
        except Exception as e:
            exception_queue.put(e)
    
    thread = threading.Thread(target=get_conn, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    
    if thread.is_alive():
        # Thread is still running, timeout occurred
        # Note: daemon thread will be cleaned up automatically
        logger.warning(f"getconn() timed out after {timeout}s")
        return None
    
    # Check for exceptions first (exceptions take priority)
    if not exception_queue.empty():
        raise exception_queue.get()
    
    # Check for result
    if not result_queue.empty():
        return result_queue.get()
    
    # No result and no exception - timeout occurred
    return None


@contextmanager
def get_db_connection(db_path: Optional[str] = None, max_retries: int = 3, timeout: float = 10.0):
    """Context manager for database connections - Supports PostgreSQL and SQLite

    Args:
        db_path: Optional path for SQLite database
        max_retries: Maximum number of retries for PostgreSQL connections
        timeout: Timeout in seconds for getting a connection from the pool
    """
    conn = None
    use_postgresql = settings.use_postgresql and POSTGRESQL_AVAILABLE and _postgresql_pool is not None

    try:
        if use_postgresql:
            # Use PostgreSQL with retry logic and timeout
            import time
            start_time = time.time()
            per_attempt_timeout = max(2.0, timeout / max_retries)  # Divide timeout across attempts
            
            for attempt in range(max_retries):
                try:
                    # Check if we've exceeded overall timeout
                    elapsed = time.time() - start_time
                    if elapsed > timeout:
                        raise Exception(f"Connection timeout after {timeout}s")
                    
                    # Calculate remaining timeout for this attempt
                    remaining_timeout = timeout - elapsed
                    attempt_timeout = min(per_attempt_timeout, remaining_timeout)
                    
                    # Try to get connection with timeout handling
                    try:
                        conn = _get_connection_with_timeout(_postgresql_pool, timeout=attempt_timeout)
                        if conn is None:
                            # Timeout occurred
                            if attempt < max_retries - 1:
                                logger.debug(
                                    f"Connection timeout, attempt {attempt + 1}/{max_retries}")
                                time.sleep(0.5)
                                continue
                            else:
                                raise Exception(f"Connection timeout after {timeout}s")
                    except Exception as pool_error:
                        # Pool exhausted or other pool error
                        error_msg = str(pool_error).lower()
                        if ("connection pool exhausted" in error_msg or 
                            "could not get connection" in error_msg or
                            "pool" in error_msg):
                            if attempt < max_retries - 1:
                                wait_time = min(0.5 * (attempt + 1), 2.0)  # Exponential backoff, max 2s
                                logger.debug(
                                    f"Connection pool exhausted, waiting {wait_time:.1f}s before retry {attempt + 1}/{max_retries}")
                                time.sleep(wait_time)
                                continue
                            else:
                                logger.error(f"Connection pool exhausted after {max_retries} attempts")
                                raise Exception("Connection pool exhausted - all connections are in use. Please try again later.")
                        else:
                            raise
                    
                    if conn:
                        # Check if connection is alive before using it
                        if not _is_connection_alive(conn):
                            logger.debug(
                                f"Connection from pool is dead, attempt {attempt + 1}/{max_retries} - getting fresh connection")
                            try:
                                # Try to close the dead connection
                                conn.close()
                            except:
                                pass
                            # Don't return dead connection to pool - it's already closed
                            conn = None
                            if attempt < max_retries - 1:
                                time.sleep(0.3)  # Brief pause before retry
                                continue
                            else:
                                raise Exception(
                                    "Failed to get a valid connection from pool after retries")
                        # Connection is valid, yield it
                        yield conn
                        break
                    else:
                        if attempt < max_retries - 1:
                            logger.warning(
                                f"Failed to get connection from pool, attempt {attempt + 1}/{max_retries}")
                            time.sleep(0.5)
                            continue
                        else:
                            raise Exception(
                                "Failed to get connection from pool")
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.debug(
                            f"Retrying connection (attempt {attempt + 1}/{max_retries}): {str(e)[:100]}")
                        if conn:
                            try:
                                conn.close()
                            except:
                                pass
                            conn = None
                        time.sleep(0.5)
                        continue
                    else:
                        logger.error(
                            f"Failed to get database connection after {max_retries} attempts: {e}")
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
                    # Always try to return connection to pool, but handle errors gracefully
                    try:
                        # Check if connection is still alive before returning to pool
                        if _is_connection_alive(conn):
                            # Return connection to pool (don't close it)
                            _postgresql_pool.putconn(conn, close=False)
                        else:
                            # Connection is dead, close it instead of returning to pool
                            logger.debug(
                                "Connection is dead, closing instead of returning to pool")
                            try:
                                conn.close()
                            except:
                                pass
                    except Exception as pool_error:
                        # Pool error (e.g., connection not from this pool, pool full, etc.)
                        error_msg = str(pool_error).lower()
                        if "pool" in error_msg or "connection" in error_msg:
                            logger.warning(f"Error returning connection to pool: {pool_error}")
                        else:
                            logger.debug(f"Error returning connection to pool: {pool_error}")
                        # If pool is full or connection is bad, close it
                        try:
                            conn.close()
                        except:
                            pass
                except Exception as e:
                    # Final fallback - ensure connection is closed
                    logger.error(f"Critical error managing connection: {e}")
                    try:
                        conn.close()
                    except:
                        pass
            else:
                try:
                    conn.close()
                except:
                    pass


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
