import os
from pydantic import BaseModel
from api.database import get_db
from api.models import Machine, SubscriptionStatus, Workflow, GPUEvent
from api.routes.utils import (
    fetch_user_icon,
    get_user_settings as get_user_settings_util,
    update_user_settings as update_user_settings_util,
    select,
)

from api.middleware.auth import (
  get_api_keys as get_api_keys_auth,
  delete_api_key as delete_api_key_auth
)
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import Date, and_, func, or_, cast, extract, text
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.responses import RedirectResponse
import aiohttp
from typing import Optional, Dict, Any
from datetime import datetime

router = APIRouter(
    tags=["Platform"],
)


@router.get("/platform/user-settings")
async def get_user_settings(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    return await get_user_settings_util(request, db)


@router.get("/user/{user_id}")
async def get_user_meta(
    user_id: str,
):
    return await fetch_user_icon(user_id)

class UserSettingsUpdateRequest(BaseModel):
    api_version: str
    custom_output_bucket: bool
    hugging_face_token: Optional[str] = None
    output_visibility: str
    s3_access_key_id: Optional[str] = None
    s3_secret_access_key: Optional[str] = None 
    s3_bucket_name: Optional[str] = None
    s3_region: Optional[str] = None
    spend_limit: Optional[float] = None

@router.patch("/platform/user-settings")
async def update_user_settings(
    request: Request,
    body: UserSettingsUpdateRequest,
    db: AsyncSession = Depends(get_db),
):
    return await update_user_settings_util(request, db, body)


class GetApiKeysRequest(BaseModel):
    limit: Optional[int] = None
    offset: Optional[int] = None
    search: Optional[str] = None

@router.get("/platform/api-keys")
async def get_api_keys(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    return await get_api_keys_auth(request, db)

@router.delete("/platform/api-keys/{key_id}")
async def delete_api_key(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    return await delete_api_key_auth(request, db)


import stripe
from fastapi import HTTPException

# You'll need to configure stripe with your API key
stripe.api_key = os.getenv("STRIPE_API_KEY")

# Define pricing plan mappings similar to TypeScript
PRICING_PLAN_MAPPING = {
    "ws_basic": os.getenv("STRIPE_PR_WS_BASIC"),
    "ws_pro": os.getenv("STRIPE_PR_WS_PRO"),
    "pro": os.getenv("STRIPE_PR_PRO"),
    "basic": os.getenv("STRIPE_PR_BASIC"),
    "creator": os.getenv("STRIPE_PR_ENTERPRISE"),
    "business": os.getenv("STRIPE_PR_BUSINESS"),
}

PRICING_PLAN_NAMES = {
    "ws_basic": "Workspace Basic",
    "ws_pro": "Workspace Pro",
    "pro": "API Pro (Early Adopter)",
    "creator": "API Creator (Early Adopter)",
    "business": "API Business",
    "basic": "API Basic",
}

# Update reverse mapping to use actual Stripe price IDs
PRICING_PLAN_REVERSE_MAPPING = {v: k for k, v in PRICING_PLAN_MAPPING.items() if v is not None}


async def get_current_plan(
    db: AsyncSession, user_id: str, org_id: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Get the current subscription plan for a user/org"""

    if not user_id:
        return None

    # Check if Stripe is disabled
    if os.getenv("DISABLE_STRIPE") == "true":
        return {
            "stripe_customer_id": "id",
            "user_id": user_id,
            "org_id": org_id,
            "plan": "enterprise",
            "status": "active",
            "subscription_id": "",
            "trial_start": 0,
            "cancel_at_period_end": False,
            "created_at": None,
            "updated_at": None,
            "subscription_item_api_id": "id",
            "subscription_item_plan_id": "id",
            "trial_end": 0,
            "last_invoice_timestamp": None,
        }

    # Query subscription status table
    # Note: This is a simplified version - you'll need to adapt the actual DB query
    # based on your SQLAlchemy models and schema
    query = (
        select(SubscriptionStatus)
        .where(
            and_(
                SubscriptionStatus.status != "deleted",
                or_(
                    and_(
                        SubscriptionStatus.org_id.is_(None),
                        SubscriptionStatus.user_id == user_id,
                    )
                    if not org_id
                    else SubscriptionStatus.org_id == org_id
                ),
            )
        )
        .order_by(SubscriptionStatus.created_at.desc())
    )

    result = await db.execute(query)
    subscription = result.scalar_one_or_none()

    if subscription:
        return {
            "stripe_customer_id": subscription.stripe_customer_id,
            "user_id": subscription.user_id,
            "org_id": subscription.org_id,
            "plan": subscription.plan,
            "status": subscription.status,
            "subscription_id": subscription.subscription_id,
            "trial_start": subscription.trial_start,
            "cancel_at_period_end": subscription.cancel_at_period_end,
            "created_at": subscription.created_at,
            "updated_at": subscription.updated_at,
            "subscription_item_api_id": subscription.subscription_item_api_id,
            "subscription_item_plan_id": subscription.subscription_item_plan_id,
            "trial_end": subscription.trial_end,
            "last_invoice_timestamp": subscription.last_invoice_timestamp,
        }

    return None


async def get_stripe_plan(
    db: AsyncSession, user_id: str, org_id: str
) -> Optional[Dict]:
    """Get the Stripe plan details for a user"""
    # First get the current plan (you'll need to implement this based on your DB schema)
    plan = await get_current_plan(db, user_id, org_id)

    if not plan or not plan.get("subscription_id"):
        return None

    try:
        stripe_plan = stripe.Subscription.retrieve(
            plan["subscription_id"], expand=["discounts"]
        )
        return stripe_plan
    except stripe.error.StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))


async def get_plans(db: AsyncSession, user_id: str, org_id: str) -> Dict[str, Any]:
    """Get plans information including current subscription details"""

    # Check if Stripe is disabled
    if os.getenv("DISABLE_STRIPE") == "true":
        return {
            "plans": ["creator"],
            "names": ["creator"],
            "prices": [os.getenv("STRIPE_PR_BUSINESS")],
            "amount": [100],
            "cancel_at_period_end": False,
            "canceled_at": None,
        }

    stripe_plan = await get_stripe_plan(db, user_id, org_id)

    if not stripe_plan:
        return None

    # Check for payment issues
    payment_issue = False
    payment_issue_reason = ""

    if stripe_plan.status in ["past_due", "unpaid"]:
        payment_issue = True
        payment_issue_reason = "Subscription payment is overdue"
    elif stripe_plan.status == "canceled" and stripe_plan.canceled_at:
        payment_issue = True
        payment_issue_reason = "Subscription has been canceled"
    elif stripe_plan.latest_invoice:
        try:
            invoice = stripe.Invoice.retrieve(stripe_plan.latest_invoice)
            if invoice.status == "uncollectible":
                payment_issue = True
                payment_issue_reason = "Your latest payment has failed. Update your payment method to continue this plan."
        except stripe.error.StripeError:
            pass
        
    print(stripe_plan)

    # Extract plans from stripe subscription
    plans = [
        PRICING_PLAN_REVERSE_MAPPING.get(item.price.id)
        for item in stripe_plan.get("items", {}).get("data", [])
        if PRICING_PLAN_REVERSE_MAPPING.get(item.price.id)
    ]

    # Calculate charges with discounts
    charges = []
    for item in stripe_plan.get("items", {}).get("data", []):
        base_amount = item.get("price", {}).get("unit_amount", 0)
        final_amount = base_amount

        # Apply subscription-level discounts
        if hasattr(stripe_plan, "discounts") and stripe_plan.discounts:
            for discount in stripe_plan.discounts:
                if isinstance(discount, str):
                    continue

                if discount.coupon.percent_off:
                    final_amount = final_amount * (
                        1 - discount.coupon.percent_off / 100
                    )
                elif discount.coupon.amount_off:
                    total_items = len(stripe_plan.items.data)
                    final_amount = max(
                        0, final_amount - (discount.coupon.amount_off / total_items)
                    )

        charges.append(round(final_amount))

    return {
        "plans": plans,
        "names": [item.get("plan", {}).get("nickname", "") for item in stripe_plan.get("items", {}).get("data", [])],
        "prices": [item.get("price", {}).get("id", "") for item in stripe_plan.get("items", {}).get("data", [])],
        "amount": [item.get("price", {}).get("unit_amount", 0) for item in stripe_plan.get("items", {}).get("data", [])],
        "charges": charges,
        "cancel_at_period_end": stripe_plan.cancel_at_period_end,
        "canceled_at": stripe_plan.canceled_at,
        "payment_issue": payment_issue,
        "payment_issue_reason": payment_issue_reason,
    }


DEFAULT_FEATURE_LIMITS = {
    "free": {"machine": 1, "workflow": 2, "private_model": False},
    "pro": {"machine": 5, "workflow": 10, "private_model": True},
    "creator": {"machine": 10, "workflow": 30, "private_model": True},
    "business": {"machine": 20, "workflow": 100, "private_model": True},
    "enterprise": {"machine": 100, "workflow": 300, "private_model": True},
}


async def get_machine_count(db: AsyncSession, request: Request) -> int:
    query = (
        select(func.count())
        .select_from(Machine)
        .where(~Machine.deleted)
        .apply_org_check_by_type(Machine, request)
    )
    result = await db.execute(query)
    return result.scalar() or 0


async def get_workflow_count(db: AsyncSession, request: Request) -> int:
    query = (
        select(func.count())
        .select_from(Workflow)
        .where(~Workflow.deleted)
        .apply_org_check_by_type(Workflow, request)
    )
    result = await db.execute(query)
    return result.scalar() or 0


@router.get("/platform/plan")
async def get_api_plan(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    # Get authenticated user info
    user_id = request.state.current_user.get("user_id")
    org_id = request.state.current_user.get("org_id")

    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")

    # Get plans, machines, workflows and user settings in parallel
    plans = await get_plans(db, user_id, org_id)
    machine_count = await get_machine_count(db, request)
    workflow_count = await get_workflow_count(db, request)
    user_settings = await get_user_settings_util(request, db)

    if plans is None:
        raise HTTPException(status_code=400, detail="Payment issue - No plans found")

    if plans.get("payment_issue"):
        raise HTTPException(
            status_code=400, detail="Payment issue - " + plans.get("payment_issue_reason")
        )

    # Check if user has subscription
    has_subscription = any(
        p in ["creator", "pro", "business"] for p in plans.get("plans", [])
    )
    plan = await get_current_plan(db, user_id, org_id) if has_subscription else None

    # Calculate limits based on plan
    if not plans.get("plans"):
        machine_max_count = DEFAULT_FEATURE_LIMITS["free"]["machine"]
        workflow_max_count = DEFAULT_FEATURE_LIMITS["free"]["workflow"]
    else:
        if "business" in plans["plans"]:
            machine_max_count = DEFAULT_FEATURE_LIMITS["business"]["machine"]
            workflow_max_count = DEFAULT_FEATURE_LIMITS["business"]["workflow"]
        elif "creator" in plans["plans"]:
            machine_max_count = DEFAULT_FEATURE_LIMITS["creator"]["machine"]
            workflow_max_count = DEFAULT_FEATURE_LIMITS["creator"]["workflow"]
        elif "pro" in plans["plans"]:
            machine_max_count = DEFAULT_FEATURE_LIMITS["pro"]["machine"]
            workflow_max_count = DEFAULT_FEATURE_LIMITS["pro"]["workflow"]
        else:
            machine_max_count = DEFAULT_FEATURE_LIMITS["free"]["machine"]
            workflow_max_count = DEFAULT_FEATURE_LIMITS["free"]["workflow"]

    # Use updated limits if available
    effective_machine_limit = max(user_settings.machine_limit or 0, machine_max_count)
    effective_workflow_limit = max(
        user_settings.workflow_limit or 0, workflow_max_count
    )

    machine_limited = machine_count >= effective_machine_limit
    workflow_limited = workflow_count >= effective_workflow_limit

    plan_key = plan.get("plan", "free") if plan else "free"
    target_plan = DEFAULT_FEATURE_LIMITS.get(plan_key, DEFAULT_FEATURE_LIMITS["free"])

    return {
        "sub": plan,
        "features": {
            "machineLimited": machine_limited,
            "machineLimit": effective_machine_limit,
            "currentMachineCount": machine_count,
            "workflowLimited": workflow_limited,
            "workflowLimit": effective_workflow_limit,
            "currentWorkflowCount": workflow_count,
            "priavteModels": target_plan["private_model"],
            "alwaysOnMachineLimit": user_settings.always_on_machine_limit,
        },
        "plans": plans,
    }


@router.get("/platform/upgrade-plan")
async def get_upgrade_plan(
    request: Request,
    plan: str,
    coupon: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """Get upgrade or new plan details with proration calculations"""
    user_id = request.state.current_user.get("user_id")
    org_id = request.state.current_user.get("org_id")
    
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")

    # Get current stripe plan
    stripe_plan = await get_stripe_plan(db, user_id, org_id)
    if not stripe_plan:
        return None

    # Find workspace and API plans
    ws_plan = next(
        (item for item in stripe_plan.get("items", {}).get("data", []) 
         if item.get("price", {}).get("id") in [PRICING_PLAN_MAPPING["ws_basic"], PRICING_PLAN_MAPPING["ws_pro"]]),
        None
    )
    
    api_plan = next(
        (item for item in stripe_plan.get("items", {}).get("data", []) 
         if item.get("price", {}).get("id") in [
             PRICING_PLAN_MAPPING["creator"],
             PRICING_PLAN_MAPPING["pro"],
             PRICING_PLAN_MAPPING["basic"],
             PRICING_PLAN_MAPPING["business"]
         ]),
        None
    )

    # Determine conflicting plan
    conflicting_plan = ws_plan if not plan.startswith("ws_") else api_plan

    # Get target price ID
    target_price_id = PRICING_PLAN_MAPPING.get(plan)
    if not target_price_id:
        raise HTTPException(status_code=400, detail="Invalid plan type")

    # Check if plan already exists
    has_target_price = any(item.get("price", {}).get("id") == target_price_id for item in stripe_plan.get("items", {}).get("data", []))
    if has_target_price:
        return None

    # Handle coupon if provided
    promotion_code_id = None
    if coupon:
        try:
            promotion_codes = stripe.PromotionCode.list(code=coupon, limit=1)
            if promotion_codes.data:
                promotion_code_id = promotion_codes.data[0].id
            else:
                raise HTTPException(status_code=400, detail="Invalid coupon")
        except stripe.error.StripeError as e:
            raise HTTPException(status_code=400, detail=str(e))

    try:
        # Calculate proration
        if conflicting_plan:
            return stripe.Invoice.upcoming(
                subscription=stripe_plan.id,
                subscription_details={
                    "proration_behavior": "always_invoice",
                    "items": [
                        {"id": conflicting_plan.id, "deleted": True},
                        {"price": target_price_id, "quantity": 1},
                    ],
                },
                discounts=[{"promotion_code": promotion_code_id}] if promotion_code_id else [],
            )
        else:
            return stripe.Invoice.upcoming(
                subscription=stripe_plan.id,
                subscription_details={
                    "proration_behavior": "always_invoice",
                    "items": [
                        {"price": target_price_id, "quantity": 1},
                    ],
                },
                discounts=[{"promotion_code": promotion_code_id}] if promotion_code_id else [],
            )

    except stripe.error.StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return None


async def get_clerk_user(user_id: str) -> dict:
    """
    Fetch user data from Clerk's Backend API
    
    Args:
        user_id: The Clerk user ID
        
    Returns:
        dict: User data from Clerk
        
    Raises:
        HTTPException: If the API call fails
    """
    clerk_api_key = os.getenv("CLERK_SECRET_KEY")
    headers = {
        "Authorization": f"Bearer {clerk_api_key}",
        "Content-Type": "application/json"
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"https://api.clerk.com/v1/users/{user_id}",
            headers=headers
        ) as response:
            if response.status == 200:
                return await response.json()
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to fetch user data from Clerk: {await response.text()}"
            )


@router.get("/platform/checkout")
async def stripe_checkout(
    request: Request,
    plan: str,
    redirect_url: str = None,
    trial: Optional[bool] = False,
    coupon: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """Handle Stripe checkout process"""
    user_id = request.state.current_user.get("user_id")
    org_id = request.state.current_user.get("org_id")

    if not user_id:
        return {"url": redirect_url or "/"}
    if not plan:
        return {"url": redirect_url or "/pricing"}

    # Get user data from Clerk
    user_data = await get_clerk_user(user_id)
    user_email = next(
        (email["email_address"] for email in user_data["email_addresses"] 
        if email["id"] == user_data["primary_email_address_id"]),
        None
    )
    
    # Get target price ID
    target_price_id = PRICING_PLAN_MAPPING.get(plan)
    if not target_price_id:
        raise HTTPException(status_code=400, detail="Invalid plan type")

    metadata = {
        "userId": user_id,
        "orgId": org_id,
        "plan": plan,
    }

    line_items = [{"price": target_price_id, "quantity": 1}]

    # Handle coupon if provided
    promotion_code_id = None
    if coupon:
        try:
            promotion_codes = stripe.PromotionCode.list(code=coupon, limit=1)
            if promotion_codes.data:
                promotion_code_id = promotion_codes.data[0].id
        except stripe.error.StripeError as e:
            print(f"Error fetching promotion code: {e}")

    # Get current plan
    current_plan = await get_current_plan(db, user_id, org_id)
    
    if current_plan and current_plan.get("subscription_id"):
        try:
            stripe_plan = stripe.Subscription.retrieve(current_plan["subscription_id"])

            if stripe_plan and stripe_plan.status != "canceled":
                # Check if user already has this plan
                has_existing_plan = any(
                    item.get("price", {}).get("id") == target_price_id for item in stripe_plan.get("items", {}).get("data", [])
                )

                if has_existing_plan:
                    # Return portal URL instead of redirecting
                    portal_session = stripe.billing_portal.Session.create(
                        customer=stripe_plan.customer,
                        return_url=redirect_url or "/pricing",
                    )
                    return {"url": portal_session.url}

                # Handle plan updates
                ws_plan = next(
                    (item for item in stripe_plan.get("items", {}).get("data", []) 
                     if item.get("price", {}).get("id") in [PRICING_PLAN_MAPPING["ws_basic"], PRICING_PLAN_MAPPING["ws_pro"]]),
                    None
                )
                api_plan = next(
                    (item for item in stripe_plan.get("items", {}).get("data", []) 
                     if item.get("price", {}).get("id") in [
                         PRICING_PLAN_MAPPING["creator"],
                         PRICING_PLAN_MAPPING["pro"],
                         PRICING_PLAN_MAPPING["basic"],
                         PRICING_PLAN_MAPPING["business"]
                     ]),
                    None
                )

                conflicting_plan = ws_plan if not plan.startswith("ws_") else api_plan

                if conflicting_plan:
                    # Update subscription with plan replacement
                    await stripe.Subscription.modify(
                        current_plan["subscription_id"],
                        proration_behavior="always_invoice",
                        metadata=metadata,
                        discounts=[{"promotion_code": promotion_code_id}] if promotion_code_id else [],
                        items=[
                            {"id": conflicting_plan.id, "deleted": True},
                            {"price": target_price_id, "quantity": 1},
                        ],
                    )
                else:
                    # Add new plan to subscription
                    await stripe.Subscription.modify(
                        current_plan["subscription_id"],
                        proration_behavior="always_invoice",
                        metadata=metadata,
                        discounts=[{"promotion_code": promotion_code_id}] if promotion_code_id else [],
                        items=[{"price": target_price_id, "quantity": 1}],
                    )

                return {"url": redirect_url or "/"}

        except stripe.error.StripeError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # Create new checkout session
    try:
        session_params = {
            "success_url": redirect_url or "/",
            "line_items": line_items,
            "metadata": metadata,
            "allow_promotion_codes": True,
            "client_reference_id": org_id or user_id,
            "customer_email": user_email,
            "mode": "subscription",
            "discounts": [{"promotion_code": promotion_code_id}] if promotion_code_id else [],
        }
        
        print(session_params)

        if trial:
            session_params["subscription_data"] = {
                "trial_settings": {
                    "end_behavior": {
                        "missing_payment_method": "cancel"
                    }
                },
                "trial_period_days": 7,
                "metadata": metadata,
            }
        else:
            session_params["subscription_data"] = {"metadata": metadata}

        session = stripe.checkout.Session.create(**session_params)
        
        if session.url:
            return {"url": session.url}

    except stripe.error.StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    raise HTTPException(status_code=400, detail="Failed to create checkout session")


# GPU pricing per second with 10% margin
GPU_PRICING = {
    "T4": round(0.000164 * 1.1, 6),
    "L4": round(0.000291 * 1.1, 6),
    "A10G": round(0.000306 * 1.1, 6),
    "A100": round(0.001036 * 1.1, 6),
    "A100-80GB": round(0.001553 * 1.1, 6),
    "H100": round(0.002125 * 1.1, 6),
    "CPU": round(0.000038 * 1.1, 6),  # Price per core
}

@router.get("/platform/gpu-pricing")
async def gpu_pricing():
    """Return the GPU pricing table"""
    return GPU_PRICING


@router.get("/platform/usage-details")
async def get_usage_details_by_day(
    request: Request,
    start_time: datetime,
    end_time: datetime,
    db: AsyncSession = Depends(get_db),
):
    """Get GPU usage details grouped by day"""
    user_id = request.state.current_user.get("user_id")
    org_id = request.state.current_user.get("org_id")

    if not user_id and not org_id:
        raise HTTPException(status_code=400, detail="User or org id is required")

    # Build the base query conditions
    conditions = [
        GPUEvent.end_time >= start_time,
        GPUEvent.end_time < end_time,
    ]

    # Add org/user conditions
    if org_id:
        conditions.append(GPUEvent.org_id == org_id)
    else:
        conditions.append(
            and_(
                or_(GPUEvent.org_id.is_(None), GPUEvent.org_id == ""),
                GPUEvent.user_id == user_id,
            )
        )

    # Create the query
    query = (
        select(
            cast(GPUEvent.start_time, Date).label("date"),
            GPUEvent.gpu,
            func.sum(
                extract('epoch', GPUEvent.end_time) - 
                extract('epoch', GPUEvent.start_time)
            ).label("usage_in_sec"),
            func.coalesce(func.sum(GPUEvent.cost), 0).label("cost"),
        )
        .where(and_(*conditions))
        .group_by(text("date"), GPUEvent.gpu)
        .order_by(text("date"))
    )

    result = await db.execute(query)
    usage_details = result.fetchall()

    # Transform the data into the desired format
    grouped_by_date = {}
    for row in usage_details:
        date_str = row.date.strftime("%Y-%m-%d")
        if date_str not in grouped_by_date:
            grouped_by_date[date_str] = {}
        
        if row.gpu:
            unit_amount = GPU_PRICING.get(row.gpu, 0)
            usage_seconds = float(row.usage_in_sec)  # Convert Decimal to float
            grouped_by_date[date_str][row.gpu] = unit_amount * usage_seconds

    # Convert to array format
    chart_data = [{"date": date, **gpu_costs} for date, gpu_costs in grouped_by_date.items()]

    return chart_data



