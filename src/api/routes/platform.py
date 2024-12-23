import os
from pydantic import BaseModel
from api.database import get_db
from api.models import Machine, SubscriptionStatus, Workflow
from api.routes.utils import (
    fetch_user_icon,
    get_user_settings as get_user_settings_util,
    update_user_settings as update_user_settings_util,
    select,
)
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import and_, func, or_
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, Dict, Any

router = APIRouter(
    tags=["Platform"],
)


@router.get("/platform/user-settings")
async def get_user_settings(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    return await get_user_settings_util(request, db)

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


@router.get("/user/{user_id}")
async def get_user_meta(
    user_id: str,
):
    return await fetch_user_icon(user_id)


import stripe
from fastapi import HTTPException

# You'll need to configure stripe with your API key
stripe.api_key = os.getenv("STRIPE_API_KEY")

# Define pricing plan mappings similar to TypeScript
PRICING_PLAN_MAPPING = {
    "creator": "price_id_for_creator",
    "pro": "price_id_for_pro",
    "business": "price_id_for_business",
    # Add other plans as needed
}

PRICING_PLAN_REVERSE_MAPPING = {v: k for k, v in PRICING_PLAN_MAPPING.items()}


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
