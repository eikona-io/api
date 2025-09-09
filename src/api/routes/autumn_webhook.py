from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)
from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Optional, Any
import logging
import os


from api.database import get_db
from api.utils import autumn
from api.utils.autumn import autumn_client, get_autumn_customer
from api.utils.webhook import WebhookRegistry, WebhookResponse
from clerk_backend_api import Clerk
import logfire

from api.models import (
    Machine,
    Workflow,
)
from sqlalchemy import (
    func,
)
from .utils import select


logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["Autumn Webhook"],
)

# Create Autumn webhook registry
autumn_registry = WebhookRegistry("Autumn webhook")


class AutumnWebhookPayload(BaseModel):
    """Simplified webhook payload - we'll fetch customer data from Autumn directly."""
    model_config = ConfigDict(extra="allow")

    data: Dict[str, Any]  # We'll extract customer.id from this
    type: str


def get_seats_from_autumn_data(autumn_customer: Dict[str, Any]) -> Optional[int]:
    """
    Get the seats count from Autumn customer data.
    Returns the seat count or None if no seats feature is found.
    """
    features = autumn_customer.get("features", {})
    seats_feature = features.get("seats")

    if not seats_feature:
        return None

    # If unlimited seats, return None (no limit to set)
    if seats_feature.get("unlimited", False):
        return None

    # Use balance or included_usage as the seat count
    seats_count = seats_feature.get("included_usage")

    return seats_count if seats_count and seats_count > 0 else None


async def update_clerk_org_seats(org_id: str, target_seats: int) -> Dict[str, Any]:
    """
    Update Clerk organization seats to the target seat count.
    Similar logic to update_seats in platform.py but adapted for webhook context.
    """

    try:
        async with Clerk(
            bearer_auth=os.getenv("CLERK_SECRET_KEY"),
        ) as clerk:
            org_data = await clerk.organizations.get_async(organization_id=org_id, include_members_count=True)

            if org_data is None:
                return {
                    "status": "error",
                    "message": f"Unable to find org ${org_id}",
                    "target_seats": target_seats
                }

            current_max_seats = org_data.max_allowed_memberships

            # Make sure to update this
            await autumn_client.set_feature_usage(
                customer_id=org_id,
                feature_id="seats",
                value=org_data.members_count
            ) 
            
            # Don't update if current max is 0 (unlimited) or already at target
            if current_max_seats == 0 or current_max_seats == target_seats:
                return {
                    "status": "success",
                    "message": "No update needed, seats already at target or unlimited",
                    "current_seats": "unlimited" if current_max_seats == 0 else current_max_seats,
                    "target_seats": target_seats
                }

            # Update seats to target count
            await clerk.organizations.update_async(organization_id=org_id, max_allowed_memberships=target_seats)

            result = {
                "status": "success",
                "message": f"Updated seats from {current_max_seats} to {target_seats}",
                "previous_seats": current_max_seats,
                "new_seats": target_seats
            }

            logfire.info(f"Seats updated via Autumn webhook for org_id {org_id}", extra=result)

            return result

    except Exception as e:
        logger.error(f"Error updating Clerk organization seats for {org_id}: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to update seats: {str(e)}",
            "target_seats": target_seats
        }


def is_org_id(customer_id: str) -> bool:
    """Check if the customer ID is a Clerk organization ID."""
    return customer_id.startswith("org_")


async def update_features_usage(customer_id: str, db: AsyncSession) -> Dict[str, Any]:
    # get machine count
    machine_count_query = (
        select(func.count())
        .select_from(Machine)
        .where(~Machine.deleted)
    )
    if customer_id.startswith("org_"):
        machine_count_query = machine_count_query.where(Machine.org_id == customer_id)
    else:
        machine_count_query = machine_count_query.where(
            Machine.user_id == customer_id,
            Machine.org_id.is_(None)
        )
    
    machine_count = await db.execute(machine_count_query)
    machine_count = machine_count.scalar()
    
    # Update seat usage for machine limit
    await autumn_client.set_feature_usage(
        customer_id=customer_id,
        feature_id="machine_limit",
        value=machine_count
    )
    
    # Count total workflows for this customer
    workflow_count_query = (
        select(func.count())
        .select_from(Workflow)
        .where(~Workflow.deleted)
    )
    if customer_id.startswith("org_"):
        workflow_count_query = workflow_count_query.where(Workflow.org_id == customer_id)
    else:
        workflow_count_query = workflow_count_query.where(
            Workflow.user_id == customer_id,
            Workflow.org_id.is_(None)
        )
    
    workflow_count = await db.execute(workflow_count_query)
    workflow_count = workflow_count.scalar()
    
    # Update seat usage for workflow limit
    await autumn_client.set_feature_usage(
        customer_id=customer_id,
        feature_id="workflow_limit",
        value=workflow_count
    )


@autumn_registry.handler("customer.products.updated")
async def handle_customer_products_updated(data: Dict[str, Any], db: AsyncSession) -> WebhookResponse:
    """Handle customer.products.updated webhook event."""

    # Extract customer ID from webhook data
    customer_data = data.get("customer", {})

    # Handle nested customer structure where ID is at data.customer.customer.id
    customer_inner = customer_data.get("customer", {})
    customer_id = customer_inner.get("id")

    # Fallback to direct customer.id structure if the nested structure doesn't exist
    if not customer_id:
        customer_id = customer_data.get("id")

    if not customer_id:
        logger.error("No customer ID found in webhook data", extra={"webhook_data": data})
        raise HTTPException(status_code=400, detail="Customer ID not found in webhook data")

    logger.info(f"Processing customer.products.updated for customer {customer_id}")
    
    await update_features_usage(customer_id, db)

    # Only process if this is an organization
    if not is_org_id(customer_id):
        return WebhookResponse(
            status="success",
            message="Customer is not an organization, skipping seat update",
            data={"customer_id": customer_id}
        )

    try:
        # Fetch fresh customer data from Autumn
        autumn_customer = await get_autumn_customer(customer_id, include_features=True)

        if not autumn_customer:
            return WebhookResponse(
                status="error",
                message="Failed to fetch customer data from Autumn",
                data={"customer_id": customer_id}
            )

        # Extract seats from features
        target_seats = get_seats_from_autumn_data(autumn_customer)

        if target_seats is None:
            return WebhookResponse(
                status="success",
                message="No seats feature found or unlimited seats, no update needed",
                data={"customer_id": customer_id}
            )

        # Update Clerk organization seats
        seat_update_result = await update_clerk_org_seats(customer_id, target_seats)

        return WebhookResponse(
            status="success",
            message="Customer products updated and seats processed",
            data={
                "customer_id": customer_id,
                "target_seats": target_seats,
                "seat_update": seat_update_result
            }
        )

    except Exception as e:
        logger.error(f"Error processing customer.products.updated for {customer_id}: {str(e)}")
        return WebhookResponse(
            status="error",
            message=f"Error processing webhook: {str(e)}",
            data={"customer_id": customer_id}
        )


# Main webhook endpoint
@router.post("/autumn/webhook", include_in_schema=False, response_model=WebhookResponse)
async def handle_autumn_webhook(
    webhook_payload: AutumnWebhookPayload,
    db: AsyncSession = Depends(get_db),
) -> WebhookResponse:
    """Main Autumn webhook endpoint that routes to appropriate handlers."""
    return await autumn_registry.handle_webhook(
        event_type=webhook_payload.type,
        data=webhook_payload.data,
        db=db
    )
