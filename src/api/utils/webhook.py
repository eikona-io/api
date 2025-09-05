from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Callable, Optional, Any
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)


class WebhookResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None


# Type alias for webhook handler functions
WebhookHandler = Callable[[Dict[str, Any], AsyncSession], WebhookResponse]


class WebhookRegistry:
    """Generic webhook handler registry system."""
    
    def __init__(self, name: str = "webhook"):
        self.name = name
        self.handlers: Dict[str, WebhookHandler] = {}
    
    def handler(self, event_type: str):
        """Decorator to register webhook handlers for specific event types."""
        def decorator(func: WebhookHandler) -> WebhookHandler:
            self.handlers[event_type] = func
            logger.info(f"Registered {self.name} handler for event type: {event_type}")
            return func
        return decorator
    
    async def handle_webhook(
        self, 
        event_type: str, 
        data: Dict[str, Any], 
        db: AsyncSession
    ) -> WebhookResponse:
        """Main webhook handling logic that routes to appropriate handlers."""
        logger.info(f"Received {self.name} event: {event_type}")
        
        # Get the appropriate handler
        handler = self.handlers.get(event_type)
        
        if not handler:
            logger.warning(f"No handler registered for {self.name} event type: {event_type}")
            return WebhookResponse(
                status="success",
                message=f"Event type '{event_type}' acknowledged but not processed",
                data={"event_type": event_type}
            )
        
        try:
            # Execute the handler
            response = await handler(data, db)
            return response
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            logger.error(f"Error processing {self.name} event {event_type}: {str(e)}")
            await db.rollback()
            raise HTTPException(
                status_code=500,
                detail=f"Error processing webhook: {str(e)}"
            )
