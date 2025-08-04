"""
Redis Consumer Group utilities for reliable stream processing.
"""
import asyncio
import json
import logging
from typing import Optional, AsyncGenerator
from ..routes.log import normalize_log_data

logger = logging.getLogger(__name__)

class RedisStreamConsumerGroup:
    def __init__(self, redis_client, group_name: str = "log_consumers"):
        self.redis = redis_client
        self.group_name = group_name
        
    async def ensure_consumer_group(self, stream_name: str):
        """Create consumer group if it doesn't exist."""
        try:
            # Try to create the consumer group starting from the beginning
            await self.redis.execute([
                "XGROUP", "CREATE", stream_name, self.group_name, "0", "MKSTREAM"
            ])
            logger.info(f"Created consumer group '{self.group_name}' for stream '{stream_name}'")
        except Exception as e:
            # Group likely already exists, which is fine
            if "BUSYGROUP" not in str(e):
                logger.warning(f"Failed to create consumer group: {e}")
    
    async def stream_logs_with_consumer_group(
        self, 
        run_id: str, 
        consumer_name: str,
        log_level: Optional[str] = None,
        poll_interval: float = 0.1  # seconds between polls
    ) -> AsyncGenerator[str, None]:
        """
        Stream logs using Redis consumer groups for reliable delivery.
        Uses non-blocking operations compatible with Upstash Redis REST API.
        
        Args:
            run_id: The stream name (run ID)
            consumer_name: Unique consumer identifier
            log_level: Optional log level filter
            poll_interval: Time to wait between polls (seconds)
        """
        stream_name = run_id
        
        # Ensure consumer group exists
        await self.ensure_consumer_group(stream_name)
        
        try:
            while True:
                try:
                    # Use non-blocking XREADGROUP (no BLOCK parameter)
                    # This ensures each message is delivered to only one consumer
                    entries = await self.redis.execute([
                        "XREADGROUP", "GROUP", self.group_name, consumer_name,
                        "STREAMS", stream_name, ">"
                    ])
                    
                    if entries and len(entries) > 0:
                        # Process each stream entry
                        for stream, items in entries:
                            for message_id, fields in items:
                                try:
                                    # Parse the Redis stream entry
                                    if len(fields) >= 2:
                                        # Parse the serialized value
                                        serialized_value = fields[1]
                                        if isinstance(serialized_value, bytes):
                                            serialized_value = serialized_value.decode('utf-8')
                                        
                                        # Try to parse as JSON, fallback to string
                                        try:
                                            log_data = json.loads(serialized_value)
                                        except json.JSONDecodeError:
                                            log_data = serialized_value
                                        
                                        # Normalize the data to the expected schema
                                        normalized_logs = normalize_log_data(log_data, log_level)
                                        
                                        # Yield each normalized log entry
                                        for log_entry in normalized_logs:
                                            yield f"data: {json.dumps(log_entry)}\n\n"
                                        
                                        # Acknowledge the message as processed
                                        await self.redis.execute([
                                            "XACK", stream_name, self.group_name, message_id
                                        ])
                                        
                                except Exception as e:
                                    logger.error(f"Error processing Redis stream entry {message_id}: {e}")
                                    # Don't acknowledge failed messages so they can be retried
                                    continue
                    
                    # Small delay to prevent busy waiting in non-blocking mode
                    await asyncio.sleep(poll_interval)
                        
                except Exception as e:
                    logger.error(f"Error in Redis consumer group stream: {e}")
                    await asyncio.sleep(1)  # Wait before retrying
                    
        except asyncio.CancelledError:
            logger.info(f"Consumer {consumer_name} cancelled")
            raise
        except Exception as e:
            logger.error(f"Fatal error in Redis consumer group stream: {e}")
            raise
    
    async def get_pending_messages(self, stream_name: str, consumer_name: str):
        """Get messages that were delivered but not acknowledged."""
        try:
            result = await self.redis.execute([
                "XPENDING", stream_name, self.group_name, 
                "-", "+", "100", consumer_name
            ])
            return result
        except Exception as e:
            logger.error(f"Error getting pending messages: {e}")
            return []
    
    async def claim_pending_messages(self, stream_name: str, consumer_name: str, min_idle_time: int = 60000):
        """Claim messages that have been pending for too long."""
        try:
            # Get pending messages
            pending = await self.redis.execute([
                "XPENDING", stream_name, self.group_name, 
                "-", "+", "100"
            ])
            
            if pending and len(pending) > 0:
                # Claim messages that are idle for more than min_idle_time
                message_ids = [msg[0] for msg in pending if msg[1] > min_idle_time]
                if message_ids:
                    result = await self.redis.execute([
                        "XCLAIM", stream_name, self.group_name, consumer_name,
                        str(min_idle_time)
                    ] + message_ids)
                    return result
        except Exception as e:
            logger.error(f"Error claiming pending messages: {e}")
        return []