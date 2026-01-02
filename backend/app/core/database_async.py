"""Async database connection and initialization for optimal performance"""
import asyncio
import logging
from typing import Optional, AsyncGenerator, Any
from contextlib import asynccontextmanager

from .config import settings

logger = logging.getLogger(__name__)

# PostgreSQL async support
try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    asyncpg = None

# SQLite async support
try:
    import aiosqlite
    AIOSQLITE_AVAILABLE = True
except ImportError:
    AIOSQLITE_AVAILABLE = False
    aiosqlite = None

# PostgreSQL connection pool
# Using Any instead of asyncpg.Pool to avoid AttributeError when asyncpg is None
_postgresql_pool: Optional[Any] = None


async def init_postgresql_pool_async() -> bool:
    """Initialize async PostgreSQL connection pool"""
    global _postgresql_pool
    if not ASYNCPG_AVAILABLE:
        logger.warning("asyncpg not available, async PostgreSQL disabled")
        return False

    if not settings.use_postgresql:
        return False

    try:
        # Build connection string
        connection_params = {
            'host': settings.postgresql_host,
            'port': settings.postgresql_port,
            'database': settings.postgresql_database,
            'user': settings.postgresql_user,
            'password': settings.postgresql_password,
            'min_size': settings.postgresql_pool_min_size,
            'max_size': settings.postgresql_pool_max_size,
            'command_timeout': 60,
        }

        # Add SSL mode for cloud databases
        if any(domain in settings.postgresql_host.lower() 
               for domain in ['render.com', 'amazonaws.com', 'herokuapp.com', 'neon.tech']):
            connection_params['ssl'] = 'require'

        _postgresql_pool = await asyncpg.create_pool(**connection_params)
        
        if _postgresql_pool:
            logger.debug(f"Async PostgreSQL connection pool initialized: {settings.postgresql_database}")
            return True
    except Exception as e:
        logger.error(f"Failed to initialize async PostgreSQL pool: {e}")
        return False
    
    return False


async def close_postgresql_pool_async():
    """Close async PostgreSQL connection pool"""
    global _postgresql_pool
    if _postgresql_pool:
        await _postgresql_pool.close()
        _postgresql_pool = None
        logger.debug("Async PostgreSQL connection pool closed")


def get_db_type_async() -> str:
    """Get current database type for async operations"""
    use_postgresql = (
        settings.use_postgresql 
        and ASYNCPG_AVAILABLE 
        and _postgresql_pool is not None
    )
    return "postgresql" if use_postgresql else "sqlite"


def get_date_function_async(days_offset: int = 0) -> str:
    """Get database-appropriate date function for async queries"""
    db_type = get_db_type_async()
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


@asynccontextmanager
async def get_db_connection_async(
    db_path: Optional[str] = None,
    timeout: float = 10.0
) -> AsyncGenerator:
    """Async context manager for database connections
    
    Supports both PostgreSQL (asyncpg) and SQLite (aiosqlite)
    
    Args:
        db_path: Optional path for SQLite database
        timeout: Connection timeout in seconds
        
    Yields:
        Database connection (asyncpg.Connection or aiosqlite.Connection)
    """
    conn = None
    use_postgresql = (
        settings.use_postgresql 
        and ASYNCPG_AVAILABLE 
        and _postgresql_pool is not None
    )

    try:
        if use_postgresql:
            # Use async PostgreSQL
            conn = await asyncio.wait_for(
                _postgresql_pool.acquire(),
                timeout=timeout
            )
            yield conn
        else:
            # Use async SQLite
            if db_path is None:
                db_path = settings.db_path
            
            if not AIOSQLITE_AVAILABLE:
                raise RuntimeError("aiosqlite not available for async SQLite operations")
            
            conn = await aiosqlite.connect(db_path, timeout=timeout)
            # Optimize SQLite settings
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA synchronous=NORMAL")
            await conn.execute("PRAGMA cache_size=10000")
            await conn.execute("PRAGMA foreign_keys=ON")
            yield conn
    except asyncio.TimeoutError:
        logger.error(f"Database connection timeout after {timeout}s")
        raise
    except Exception as e:
        logger.error(f"Database connection error: {e}", exc_info=True)
        raise
    finally:
        if conn:
            if use_postgresql:
                # Return connection to pool
                await _postgresql_pool.release(conn)
            else:
                # Close SQLite connection
                await conn.close()


async def init_database_async():
    """Initialize the main database with predictions table (async)"""
    db_type = get_db_type_async()

    async with get_db_connection_async() as conn:
        if db_type == "postgresql":
            # PostgreSQL async
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    prediction_date DATE NOT NULL,
                    predicted_price DECIMAL(10, 2) NOT NULL,
                    actual_price DECIMAL(10, 2),
                    accuracy_percentage DECIMAL(5, 2),
                    prediction_method VARCHAR(100),
                    prediction_reasons TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT predictions_prediction_date_key UNIQUE (prediction_date)
                )
            ''')
            
            # Add prediction_reasons column if it doesn't exist
            await conn.execute('''
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns 
                        WHERE table_name = 'predictions' AND column_name = 'prediction_reasons'
                    ) THEN
                        ALTER TABLE predictions ADD COLUMN prediction_reasons TEXT;
                    END IF;
                END $$;
            ''')
            
            # Create indexes
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_predictions_date 
                ON predictions(prediction_date);
            ''')
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_predictions_created_at 
                ON predictions(created_at);
            ''')
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_predictions_date_created 
                ON predictions(prediction_date, created_at);
            ''')
        else:
            # SQLite async
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_date DATE NOT NULL UNIQUE,
                    predicted_price REAL NOT NULL,
                    actual_price REAL,
                    accuracy_percentage REAL,
                    prediction_method TEXT,
                    prediction_reasons TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Add prediction_reasons column if it doesn't exist
            try:
                cursor = await conn.execute("PRAGMA table_info(predictions)")
                rows = await cursor.fetchall()
                columns = [row[1] for row in rows]
                if 'prediction_reasons' not in columns:
                    await conn.execute('ALTER TABLE predictions ADD COLUMN prediction_reasons TEXT')
            except Exception:
                pass
            
            # Create indexes
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_predictions_date 
                ON predictions(prediction_date);
            ''')
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_predictions_created_at 
                ON predictions(created_at);
            ''')
            
            # SQLite requires explicit commit
            await conn.commit()
        
        # PostgreSQL auto-commits DDL operations, no need for explicit commit
        logger.debug(f"Async database initialized successfully: {db_type.upper()}")


__all__ = [
    'get_db_connection_async',
    'get_db_type_async',
    'get_date_function_async',
    'init_database_async',
    'init_postgresql_pool_async',
    'close_postgresql_pool_async',
]

