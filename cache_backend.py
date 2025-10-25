"""Redis-based caching backend for API responses."""
import json
import logging
from typing import Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available. Install with: pip install redis")


class CacheBackend:
    """Abstract cache backend interface."""

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        raise NotImplementedError

    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache with TTL in seconds."""
        raise NotImplementedError

    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        raise NotImplementedError

    def clear(self) -> bool:
        """Clear all cache entries."""
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        raise NotImplementedError


class InMemoryCache(CacheBackend):
    """In-memory cache implementation (default)."""

    def __init__(self):
        self.cache = {}
        self.timestamps = {}
        logger.info("Using in-memory cache backend")

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key not in self.cache:
            return None

        # Check if expired
        timestamp = self.timestamps.get(key)
        if timestamp and datetime.now() > timestamp:
            self.delete(key)
            return None

        return self.cache.get(key)

    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache with TTL."""
        try:
            self.cache[key] = value
            self.timestamps[key] = datetime.now() + timedelta(seconds=ttl)
            return True
        except Exception as e:
            logger.error(f"Failed to set cache key {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            self.cache.pop(key, None)
            self.timestamps.pop(key, None)
            return True
        except Exception as e:
            logger.error(f"Failed to delete cache key {key}: {e}")
            return False

    def clear(self) -> bool:
        """Clear all cache entries."""
        try:
            self.cache.clear()
            self.timestamps.clear()
            return True
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        return self.get(key) is not None


class RedisCache(CacheBackend):
    """Redis-based cache implementation."""

    def __init__(self, host: str = 'localhost', port: int = 6379,
                 db: int = 0, password: Optional[str] = None,
                 prefix: str = 'stockgenie:'):
        """Initialize Redis cache connection."""
        if not REDIS_AVAILABLE:
            raise ImportError("Redis package not installed. Install with: pip install redis")

        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            self.prefix = prefix
            # Test connection
            self.client.ping()
            logger.info(f"Connected to Redis at {host}:{port}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def _make_key(self, key: str) -> str:
        """Add prefix to key."""
        return f"{self.prefix}{key}"

    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        try:
            value = self.client.get(self._make_key(key))
            if value is None:
                return None
            return json.loads(value)
        except (redis.RedisError, json.JSONDecodeError) as e:
            logger.error(f"Failed to get cache key {key}: {e}")
            return None

    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in Redis cache with TTL."""
        try:
            serialized = json.dumps(value)
            return self.client.setex(
                self._make_key(key),
                ttl,
                serialized
            )
        except (redis.RedisError, TypeError, json.JSONEncodeError) as e:
            logger.error(f"Failed to set cache key {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete key from Redis cache."""
        try:
            self.client.delete(self._make_key(key))
            return True
        except redis.RedisError as e:
            logger.error(f"Failed to delete cache key {key}: {e}")
            return False

    def clear(self) -> bool:
        """Clear all cache entries with prefix."""
        try:
            keys = self.client.keys(f"{self.prefix}*")
            if keys:
                self.client.delete(*keys)
            logger.info(f"Cleared {len(keys)} cache entries")
            return True
        except redis.RedisError as e:
            logger.error(f"Failed to clear cache: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        try:
            return self.client.exists(self._make_key(key)) > 0
        except redis.RedisError as e:
            logger.error(f"Failed to check key existence {key}: {e}")
            return False

    def get_ttl(self, key: str) -> Optional[int]:
        """Get remaining TTL for key in seconds."""
        try:
            ttl = self.client.ttl(self._make_key(key))
            return ttl if ttl > 0 else None
        except redis.RedisError as e:
            logger.error(f"Failed to get TTL for key {key}: {e}")
            return None

    def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment counter in Redis."""
        try:
            return self.client.incrby(self._make_key(key), amount)
        except redis.RedisError as e:
            logger.error(f"Failed to increment key {key}: {e}")
            return None

    def decrement(self, key: str, amount: int = 1) -> Optional[int]:
        """Decrement counter in Redis."""
        try:
            return self.client.decrby(self._make_key(key), amount)
        except redis.RedisError as e:
            logger.error(f"Failed to decrement key {key}: {e}")
            return None


def get_cache_backend(use_redis: bool = False, **redis_kwargs) -> CacheBackend:
    """Factory function to get appropriate cache backend.

    Args:
        use_redis: Whether to use Redis cache (requires redis package)
        **redis_kwargs: Additional arguments for Redis connection

    Returns:
        CacheBackend instance (RedisCache or InMemoryCache)
    """
    if use_redis and REDIS_AVAILABLE:
        try:
            return RedisCache(**redis_kwargs)
        except Exception as e:
            logger.warning(f"Failed to initialize Redis cache: {e}. Falling back to in-memory cache.")
            return InMemoryCache()
    else:
        if use_redis and not REDIS_AVAILABLE:
            logger.warning("Redis requested but not available. Using in-memory cache.")
        return InMemoryCache()
