from functools import wraps
from datetime import datetime, timedelta
import asyncio
import json
from typing import Any, Callable, TypeVar, Optional, Dict, Tuple
import os
import logfire
from upstash_redis.asyncio import Redis

T = TypeVar('T')

# Initialize Redis client at the module level (you already have this)
redis_url = os.getenv("UPSTASH_REDIS_META_REST_URL")
redis_token = os.getenv("UPSTASH_REDIS_META_REST_TOKEN")
redis = Redis(url=redis_url, token=redis_token)

class MultiLevelCache:
    """
    A two-level cache with in-memory as L1 and Redis as L2.
    Implements SWR (stale-while-revalidate) pattern.
    """
    def __init__(self, maxsize=1000, ttl_seconds=3600, redis_ttl_seconds=86400):
        self.memory_cache: Dict[str, Tuple[Any, datetime]] = {}
        self.maxsize = maxsize
        self.memory_ttl = timedelta(seconds=ttl_seconds)
        self.redis_ttl = redis_ttl_seconds
        self.redis = redis
        self.refresh_tasks = {}  # Track background refresh tasks

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache, trying memory first, then Redis."""
        # Try memory cache first (L1)
        if key in self.memory_cache:
            value, timestamp = self.memory_cache[key]
            age = datetime.now() - timestamp

            # If data is fresh enough, return immediately
            if age < self.memory_ttl:
                return value

            # If stale but not expired, trigger background refresh and return stale data
            if age < self.memory_ttl * 2:
                # Only start a refresh if one isn't already running
                if key not in self.refresh_tasks or self.refresh_tasks[key].done():
                    # We'll implement refresh_key later
                    self.refresh_tasks[key] = asyncio.create_task(self._refresh_key(key))
                return value

        # If not in memory or too stale, try Redis (L2)
        try:
            redis_value = await self.redis.get(key)
            if redis_value:
                # Parse the JSON value
                parsed_value = json.loads(redis_value)
                # Update memory cache
                self.memory_cache[key] = (parsed_value, datetime.now())
                self._cleanup_if_needed()
                return parsed_value
        except Exception as e:
            logfire.error(f"Redis cache error: {str(e)}")

        # Not found in either cache
        return None

    async def set(self, key: str, value: Any) -> None:
        """Set value in both memory and Redis caches."""
        # Update memory cache
        self.memory_cache[key] = (value, datetime.now())
        self._cleanup_if_needed()

        # Update Redis cache
        try:
            # Convert to JSON for Redis storage
            json_value = json.dumps(value)
            await self.redis.set(key, json_value, ex=self.redis_ttl)
        except Exception as e:
            logfire.error(f"Redis cache set error: {str(e)}")

    async def invalidate(self, key: str) -> None:
        """Remove a key from both caches."""
        if key in self.memory_cache:
            del self.memory_cache[key]
        try:
            await self.redis.delete(key)
        except Exception as e:
            logfire.error(f"Redis cache delete error: {str(e)}")

    def _cleanup_if_needed(self) -> None:
        """Remove oldest items if cache exceeds maxsize."""
        if len(self.memory_cache) > self.maxsize:
            # Sort by timestamp and remove oldest entries
            sorted_items = sorted(
                self.memory_cache.items(),
                key=lambda x: x[1][1]  # Sort by timestamp
            )
            # Remove oldest items to get back to 90% capacity
            items_to_remove = len(self.memory_cache) - int(self.maxsize * 0.9)
            for i in range(items_to_remove):
                del self.memory_cache[sorted_items[i][0]]

    async def _refresh_key(self, key: str) -> None:
        """Background task to refresh a key using the original function."""
        # This will be implemented in the decorator
        pass

# Create a global cache instance
multi_cache = MultiLevelCache(maxsize=1000, ttl_seconds=3600, redis_ttl_seconds=86400)

async def update_redis_cache(cache_key, *args, **kwargs):
    redis_value = await multi_cache.redis.get(cache_key)
    if redis_value:
        parsed_value = json.loads(redis_value)
        multi_cache.memory_cache[cache_key] = (parsed_value, datetime.now())
        multi_cache._cleanup_if_needed()

def multi_level_cached(
    key_prefix: str = "",
    ttl_seconds: int = None,
    redis_ttl_seconds: int = None,
    key_builder: Callable = None,
    version: str = "1.0"  # Add version parameter
):
    """
    Decorator for caching async functions with multi-level caching.
    Implements SWR pattern with version control.

    Args:
        key_prefix: Prefix for cache keys
        ttl_seconds: Override default memory TTL for this function (optional)
        redis_ttl_seconds: Override default Redis TTL for this function (optional)
        key_builder: Optional function to build cache key from function args
        version: Version string for cache invalidation
    """
    def decorator(func):
        # Store function-specific TTL overrides
        func_memory_ttl = timedelta(seconds=ttl_seconds) if ttl_seconds is not None else multi_cache.memory_ttl
        func_redis_ttl = redis_ttl_seconds if redis_ttl_seconds is not None else multi_cache.redis_ttl

        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Build the cache key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                # Default key builder combines prefix, function name, and args
                arg_str = ":".join(str(arg) for arg in args)
                kwarg_str = ":".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = f"{key_prefix}:{func.__name__}:{arg_str}:{kwarg_str}"

            # Try to get from memory cache first (L1)
            if cache_key in multi_cache.memory_cache:
                value, timestamp = multi_cache.memory_cache[cache_key]

                # Check version before using cached data
                cached_version = value.get("version")
                if cached_version != version:
                    # logfire.warning(f"Version mismatch in memory cache for {cache_key}. Cached: {cached_version}, Current: {version}")
                    await multi_cache.invalidate(cache_key)
                else:
                    age = datetime.now() - timestamp
                    redis_last_updated = value.get("timestamp", 0)
                    last_updated_age = datetime.now() - datetime.fromtimestamp(redis_last_updated)

                    # Use function-specific TTL for freshness check
                    if age < func_memory_ttl:
                        pass
                    elif last_updated_age > func_redis_ttl:
                        if cache_key not in multi_cache.refresh_tasks or multi_cache.refresh_tasks[cache_key].done():
                            # logfire.warning(f"Invalidating redis cache: {cache_key}")
                            multi_cache.refresh_tasks[cache_key] = asyncio.create_task(
                                refresh_func(cache_key, *args, **kwargs)
                            )
                    elif age < func_redis_ttl: #60
                        redis_refresh_cache_key = f"{cache_key}:redis_refresh"

                        if redis_refresh_cache_key not in multi_cache.refresh_tasks or multi_cache.refresh_tasks[redis_refresh_cache_key].done():
                            # logfire.info(f"Invalidating memory cache: {cache_key}")
                            multi_cache.refresh_tasks[redis_refresh_cache_key] = asyncio.create_task(
                                update_redis_cache(cache_key, *args, **kwargs)
                            )

                    # logfire.info(f"Returning cached memory value: {cache_key}")
                    return value.get("data")

            # logfire.warning(f"Cache miss memory: {cache_key}")
            # If not in memory or too stale, try Redis (L2)
            try:
                # with logfire.span("Getting cache from redis"):
                redis_value = await multi_cache.redis.get(cache_key)
                if redis_value:
                    # Parse the JSON value
                    parsed_value = json.loads(redis_value)

                    # Check version in Redis cache
                    redis_version = parsed_value.get("version")
                    if redis_version != version:
                        # logfire.warning(f"Version mismatch in Redis cache for {cache_key}. Cached: {redis_version}, Current: {version}")
                        await multi_cache.invalidate(cache_key)
                    else:
                        # logfire.info(f"Cache hit redis: {cache_key}")
                        # Update memory cache
                        multi_cache.memory_cache[cache_key] = (parsed_value, datetime.now())
                        multi_cache._cleanup_if_needed()
                        return parsed_value.get("data")
            except Exception as e:
                logfire.error(f"Redis cache error: {str(e)}")

            # Cache miss - call the original function
            # logfire.warning(f"Cache miss redis: {cache_key}, refreshing")
            result = await func(*args, **kwargs)

            # When storing new results, include version
            cache_data = {
                "data": result,
                "timestamp": int(datetime.now().timestamp()),
                "version": version  # Add version to cache data
            }

            # Store in both caches
            multi_cache.memory_cache[cache_key] = (cache_data, datetime.now())
            multi_cache._cleanup_if_needed()

            try:
                json_value = json.dumps(cache_data)
                await multi_cache.redis.set(cache_key, json_value, ex=func_redis_ttl)
            except Exception as e:
                logfire.error(f"Redis cache set error: {str(e)}")

            return result

        # Function to refresh a cache entry
        async def refresh_func(key: str, *args, **kwargs):
            try:
                result = await func(*args, **kwargs)

                cache_data = {
                    "data": result,
                    "timestamp": int(datetime.now().timestamp()),
                    "version": version  # Add version to refresh data
                }

                # Update both caches
                multi_cache.memory_cache[key] = (cache_data, datetime.now())

                try:
                    json_value = json.dumps(cache_data)
                    await multi_cache.redis.set(key, json_value, ex=func_redis_ttl)
                except Exception as e:
                    logfire.error(f"Redis cache refresh error: {str(e)}")

                return result
            except Exception as e:
                logfire.error(f"Error refreshing cache key {key}: {str(e)}")

        return wrapper
    return decorator