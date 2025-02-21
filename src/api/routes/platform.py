import os
import logfire
from pydantic import BaseModel
from api.database import get_db
from api.models import Machine, SubscriptionStatus, User, Workflow, GPUEvent
from api.routes.utils import (
    fetch_user_icon,
    get_user_settings as get_user_settings_util,
    update_user_settings as update_user_settings_util,
    select,
)

from api.middleware.auth import (
  get_api_keys as get_api_keys_auth,
  delete_api_key as delete_api_key_auth,
  create_api_key as create_api_key_auth
)
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import Date, and_, desc, func, or_, cast, extract, text, not_
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.responses import JSONResponse, RedirectResponse
import aiohttp
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import asyncio
import unicodedata
import re
import functools
from upstash_redis.asyncio import Redis
import json
import resend

redis_url = os.getenv("UPSTASH_REDIS_META_REST_URL")
redis_token = os.getenv("UPSTASH_REDIS_META_REST_TOKEN")
if redis_token is None:
    print("Redis token is None", redis_url)
    redisMeta = Redis(url="http://localhost:8079", token="example_token")
else:
    redisMeta = Redis(url=redis_url, token=redis_token)

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
    api_version: Optional[str] = None
    custom_output_bucket: Optional[bool] = None
    hugging_face_token: Optional[str] = None
    output_visibility: Optional[str] = None
    s3_access_key_id: Optional[str] = None
    s3_secret_access_key: Optional[str] = None
    s3_bucket_name: Optional[str] = None
    s3_region: Optional[str] = None
    spend_limit: Optional[float] = None

@router.put("/platform/user-settings")
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

@router.post("/platform/api-keys")
async def create_api_key(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    return await create_api_key_auth(request, db)

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

# Stripe price lookup keys for each plan
PLAN_LOOKUP_KEYS = {
    "creator_monthly": "creator_monthly",
    "creator_yearly": "creator_yearly",
    "creator_legacy_monthly": "creator_legacy_monthly",
    "business_monthly": "business_monthly",
    "business_yearly": "business_yearly",
    "deployment_monthly": "deployment_monthly",
    "deployment_yearly": "deployment_yearly",
}

@functools.lru_cache(maxsize=1)
def get_pricing_plan_mapping():
    """
    Fetches and caches Stripe pricing IDs using lookup keys.
    The cache is used to avoid repeated Stripe API calls.
    """
    try:
        prices = stripe.Price.list(active=True, expand=['data.product'])
        mapping = {}
        reverse_mapping = {}
        
        for price in prices.data:
            lookup_key = price.lookup_key
            if lookup_key in PLAN_LOOKUP_KEYS.values():
                mapping[lookup_key] = price.id
                reverse_mapping[price.id] = lookup_key
                
        return mapping, reverse_mapping
    except stripe.error.StripeError as e:
        logfire.error(f"Failed to fetch Stripe prices: {str(e)}")
        return {}, {}

def get_price_id(plan_key: str) -> str:
    """Get Stripe price ID for a given plan key"""
    mapping, _ = get_pricing_plan_mapping()
    return mapping.get(plan_key)

def get_plan_key(price_id: str) -> str:
    """Get plan key for a given Stripe price ID"""
    _, reverse_mapping = get_pricing_plan_mapping()
    return reverse_mapping.get(price_id)

# Update reverse mapping to use actual Stripe price IDs
@functools.lru_cache(maxsize=1)
def get_pricing_plan_reverse_mapping():
    """Get reverse mapping from price ID to plan key, lazily loaded and cached"""
    mapping, _ = get_pricing_plan_mapping()
    return {v: k for k, v in mapping.items() if v is not None}

PRICING_PLAN_NAMES = {
    "creator_monthly": "Creator Monthly",
    "creator_yearly": "Creator Yearly",
    "creator_legacy_monthly": "Creator (Legacy)",
    "business_monthly": "Business Monthly",
    "business_yearly": "Business Yearly",
    "deployment_monthly": "Deployment Monthly",
    "deployment_yearly": "Deployment Yearly",
}


async def update_subscription_redis_data(
    subscription_id: Optional[str] = None,
    user_id: Optional[str] = None,
    org_id: Optional[str] = None,
    last_invoice_timestamp: Optional[int] = None,
    db: Optional[AsyncSession] = None,
) -> dict:
    """
    Helper function to create and update subscription data in Redis with versioning.
    Always fetches fresh data from Stripe if subscription_id is provided.
    Preserves existing fields if they're not in the new data.
    """
    # Generate Redis key from org_id or user_id
    if not (org_id or user_id):
        raise ValueError("Either org_id or user_id must be provided")
    redis_key = f"plan:{org_id or user_id}"
    
    # Get existing data from Redis
    redis_data = {}
    try:
        raw_data = await redisMeta.get(redis_key)
        if raw_data:
            existing_data = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
            # Preserve fields we want to keep
            if "spent" in existing_data:
                redis_data["spent"] = existing_data["spent"]
    except Exception as e:
        logfire.error(f"Error fetching existing Redis data: {str(e)}")
        existing_data = None
            
    # If we have a subscription ID, fetch latest data from Stripe
    stripe_sub = None
    subscription_items = []
    if subscription_id:
        try:
            # Fetch the full subscription object with expanded items
            stripe_sub = stripe.Subscription.retrieve(
                subscription_id,
                expand=["items.data.price.product", "discounts"]
            )
            subscription_items = await get_subscription_items(subscription_id)
            logfire.info(f"Fetched latest subscription data for {subscription_id}")
        except stripe.error.StripeError as e:
            logfire.error(f"Error fetching subscription data: {str(e)}")
            
    # If we have Stripe subscription data, update core fields
    if stripe_sub:
        # Debug logging
        logfire.info(f"Subscription items: {subscription_items}")
        
        # Determine plan from subscription items
        plan = None
        if subscription_items and len(subscription_items) > 0:
            first_item = subscription_items[0]
            plan = get_plan_key(first_item.price.id)
            logfire.info(f"Got plan from subscription items: {plan}")
            
        # Fall back to existing data if no plan found
        if not plan and existing_data and existing_data.get("plan"):
            plan = existing_data["plan"]
            logfire.info(f"Using existing plan: {plan}")
            
        logfire.info(f"Final determined plan: {plan}")
            
        redis_data.update({
            "status": stripe_sub.get("status"),
            "stripe_customer_id": stripe_sub.get("customer"),
            "subscription_id": stripe_sub.get("id"),
            "trial_end": stripe_sub.get("trial_end"),
            "trial_start": stripe_sub.get("trial_start"),
            "cancel_at_period_end": stripe_sub.get("cancel_at_period_end"),
            "current_period_start": stripe_sub.get("current_period_start"),
            "current_period_end": stripe_sub.get("current_period_end"),
            "canceled_at": stripe_sub.get("canceled_at"),
            "payment_issue": stripe_sub.get("status") in ["past_due", "unpaid"],
            "payment_issue_reason": "Subscription payment is overdue" 
                if stripe_sub.get("status") in ["past_due", "unpaid"] else None,
            "plan": plan,
        })
    
        # Add subscription items
        redis_data["subscription_items"] = [
            {
                "id": item.id,
                "price_id": item.price.id,
                "plan_key": get_plan_key(item.price.id),
                "unit_amount": item.price.unit_amount,
                "nickname": item.price.nickname,
            }
            for item in subscription_items
            if get_plan_key(item.price.id) is not None
        ]
        
        # Calculate charges with discounts
        if stripe_sub.get("discounts"):
            charges = []
            for item in subscription_items:
                if not get_plan_key(item.price.id):
                    continue
                    
                base_amount = item.price.unit_amount
                if base_amount is None:
                    continue
                    
                final_amount = float(base_amount)
                
                # Apply subscription-level discounts
                for discount in stripe_sub.get("discounts", []):
                    if discount.get("coupon", {}).get("percent_off"):
                        final_amount = final_amount * (1 - discount["coupon"]["percent_off"] / 100)
                    elif discount.get("coupon", {}).get("amount_off"):
                        total_items = len(subscription_items)
                        final_amount = max(0, final_amount - (discount["coupon"]["amount_off"] / total_items))
                            
                charges.append(round(final_amount))
                
            redis_data["charges"] = charges
    
    # Add user/org IDs
    if user_id:
        redis_data["user_id"] = user_id
    if org_id:
        redis_data["org_id"] = org_id
        
    # Handle last_invoice_timestamp
    if last_invoice_timestamp is not None:
        redis_data["last_invoice_timestamp"] = last_invoice_timestamp
    elif "last_invoice_timestamp" not in redis_data and db is not None:
        # If last_invoice_timestamp is missing, try to fetch from subscription table
        try:
            query = (
                select(SubscriptionStatus.last_invoice_timestamp)
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
                .limit(1)
            )
            result = await db.execute(query)
            db_last_invoice = result.scalar_one_or_none()
            
            if db_last_invoice:
                redis_data["last_invoice_timestamp"] = int(db_last_invoice.timestamp())
                logfire.info(f"Retrieved last_invoice_timestamp from subscription table: {db_last_invoice}")
        except Exception as e:
            logfire.error(f"Error fetching last_invoice_timestamp from subscription table: {str(e)}")
        
    # Add data version
    redis_data["version"] = 2
    
    # Write to Redis
    await redisMeta.set(redis_key, redis_data)
    
    return redis_data

async def find_stripe_subscription(user_id: str, org_id: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Helper function to find Stripe subscription by user_id/org_id or email.
    Returns a tuple of (customer_id, subscription_id)
    Ensures proper organization context is maintained when searching by email.
    """
    try:
        # First try to search subscriptions by metadata with exact org context
        query = f"metadata['userId']:'{user_id}'"
        if org_id:
            query = f"metadata['orgId']:'{org_id}'"
            
        subscriptions = stripe.Subscription.search(
            query=query,
            limit=1,
            expand=["data.customer"]
        )
        if subscriptions.data:
            sub = subscriptions.data[0]
            return sub.customer.id, sub.id
            
        # If not found, try searching customers with exact org context
        # customer_query = f"metadata['userId']:'{user_id}'"
        # if org_id:
        #     customer_query = f"metadata['orgId']:'{org_id}'"
            
        # customers = stripe.Customer.search(
        #     query=customer_query,
        #     limit=1,
        #     expand=["data.subscriptions"]
        # )
        # if customers.data:
        #     customer = customers.data[0]
        #     if customer.subscriptions and customer.subscriptions.data:
        #         # Verify subscription has correct org context
        #         for subscription in customer.subscriptions.data:
        #             if org_id and subscription.metadata.get('orgId') == org_id:
        #                 return customer.id, subscription.id
        #             elif not org_id and not subscription.metadata.get('orgId'):
        #                 return customer.id, subscription.id
        #     return customer.id, None    
            
        # Email search as last resort, but only if we have user_id and no org_id
        # This prevents org context mixup since org subscriptions should be found by metadata
        if user_id and user_id.startswith("user_") and not org_id:
            try:
                user_data = await get_clerk_user(user_id)
                if user_data:
                    email = next(
                        (
                            email["email_address"]
                            for email in user_data["email_addresses"]
                            if email["id"] == user_data["primary_email_address_id"]
                        ),
                        None,
                    )
                    if email:
                        # Search Stripe by email but verify user context
                        customers = stripe.Customer.search(
                            query=f"email:'{email}'",
                            limit=10,  # Increased limit to search through more customers
                            expand=["data.subscriptions"]
                        )
                        for customer in customers.data:
                            # Check customer metadata first
                            if customer.metadata.get('userId') == user_id:
                                if customer.subscriptions and customer.subscriptions.data:
                                    # Find subscription without org context
                                    for subscription in customer.subscriptions.data:
                                        if not subscription.metadata.get('orgId'):
                                            return customer.id, subscription.id
                                return customer.id, None
                                
                        # If no exact match found, use first customer but update their metadata
                        if customers.data:
                            customer = customers.data[0]
                            # Update customer metadata to prevent future mixups
                            stripe.Customer.modify(
                                customer.id,
                                metadata={'userId': user_id}
                            )
                            if customer.subscriptions and customer.subscriptions.data:
                                for subscription in customer.subscriptions.data:
                                    if not subscription.metadata.get('orgId'):
                                        # Update subscription metadata
                                        stripe.Subscription.modify(
                                            subscription.id,
                                            metadata={'userId': user_id}
                                        )
                                        return customer.id, subscription.id
                            return customer.id, None
            except Exception as e:
                logfire.error(f"Error fetching user data from Clerk: {str(e)}")
                
    except stripe.error.StripeError as e:
        logfire.error(f"Error searching Stripe: {str(e)}")
    
    return None, None

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

    # First try to get from Redis
    redis_key = f"plan:{org_id or user_id}"
    plan_data = None
    
    try:
        raw_data = await redisMeta.get(redis_key)
        if raw_data:
            plan_data = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
    except json.JSONDecodeError:
        logfire.error(f"Failed to parse Redis data for key {redis_key}")
        plan_data = None

    # If Redis data is missing or invalid, try to recover from DB and Stripe
    if not plan_data or not plan_data.get("subscription_id"):
        # First try to get customer ID from subscription status table
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
            .limit(1)
        )
        result = await db.execute(query)
        subscription = result.scalar_one_or_none()

        # Get customer ID and subscription ID either from DB or by searching
        customer_id = None
        subscription_id = None
        
        if subscription and subscription.stripe_customer_id:
            # This is old data
            customer_id = subscription.stripe_customer_id
            subscription_id = subscription.subscription_id
        else:
            # Try to find customer and subscription by user_id/org_id or email
            customer_id, found_sub_id = await find_stripe_subscription(user_id, org_id)
            if customer_id:
                logfire.info(f"Found Stripe customer {customer_id} by searching")
                if found_sub_id:
                    subscription_id = found_sub_id
                    logfire.info(f"Found Stripe subscription {subscription_id} by searching")

        if customer_id:
            try:
                # If we don't have a subscription ID yet, search for active/trialing ones
                if not subscription_id:
                    # Search for active subscriptions for this customer
                    subscriptions = stripe.Subscription.list(
                        customer=customer_id,
                        status="active",
                        expand=["data.items.data.price.product", "data.discounts"]
                    )
                    
                    if subscriptions.data:
                        # Use the most recent active subscription
                        active_sub = subscriptions.data[0]
                        subscription_id = active_sub.id
                    else:
                        # Also check for trialing subscriptions
                        trial_subs = stripe.Subscription.list(
                            customer=customer_id,
                            status="trialing",
                            expand=["data.items.data.price.product", "data.discounts"]
                        )
                        if trial_subs.data:
                            subscription_id = trial_subs.data[0].id

                if subscription_id:
                    # Update Redis with the found subscription
                    plan_data = await update_subscription_redis_data(
                        subscription_id=subscription_id,
                        user_id=user_id,
                        org_id=org_id,
                        last_invoice_timestamp=int(subscription.last_invoice_timestamp.timestamp()) if subscription and subscription.last_invoice_timestamp else None,
                        db=db
                    )
                    logfire.info(f"Recovered subscription data for customer {customer_id}")
            except stripe.error.StripeError as e:
                logfire.error(f"Error fetching Stripe subscriptions: {str(e)}")
                return None

    if not plan_data:
        return None

    # Get the first subscription item as the main plan
    subscription_items = plan_data.get("subscription_items", [])
    main_plan = next(iter(subscription_items), {}) if subscription_items else {}
    
    return {
        "stripe_customer_id": plan_data.get("stripe_customer_id"),
        "user_id": plan_data.get("user_id"),
        "org_id": plan_data.get("org_id"),
        "plan": main_plan.get("plan_key", plan_data.get("plan", "free")),  # Fallback to old plan field
        "status": plan_data.get("status"),
        "subscription_id": plan_data.get("subscription_id"),
        "trial_start": plan_data.get("trial_start"),
        "cancel_at_period_end": plan_data.get("cancel_at_period_end", False),
        "created_at": plan_data.get("current_period_start"),
        "updated_at": None,
        "subscription_item_api_id": main_plan.get("id"),
        "subscription_item_plan_id": main_plan.get("price_id"),
        "trial_end": plan_data.get("trial_end"),
        "last_invoice_timestamp": plan_data.get("last_invoice_timestamp"),
    }


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

    # Get current plan data using get_current_plan
    plan_data = await get_current_plan(db, user_id, org_id)
    if not plan_data:
        return {
            "plans": [],
            "names": [],
            "prices": [],
            "amount": [],
            "charges": [],
            "cancel_at_period_end": False,
            "canceled_at": None,
            "payment_issue": False,
            "payment_issue_reason": "",
        }

    # Get Redis data for additional fields not in plan_data
    redis_key = f"plan:{org_id or user_id}"
    try:
        raw_data = await redisMeta.get(redis_key)
        if raw_data:
            redis_data = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
            subscription_items = redis_data.get("subscription_items", [])
            
            return {
                "plans": [item["plan_key"] for item in subscription_items],
                "names": [item["nickname"] for item in subscription_items],
                "prices": [item["price_id"] for item in subscription_items],
                "amount": [item["unit_amount"] for item in subscription_items],
                "charges": redis_data.get("charges", []),
                "cancel_at_period_end": plan_data.get("cancel_at_period_end", False),
                "canceled_at": plan_data.get("canceled_at"),
                "payment_issue": plan_data.get("payment_issue", False),
                "payment_issue_reason": plan_data.get("payment_issue_reason", ""),
            }
    except Exception as e:
        logfire.error(f"Error getting Redis data in get_plans: {str(e)}")
    
    # Fallback to empty response if Redis data is not available
    return {
        "plans": [],
        "names": [],
        "prices": [],
        "amount": [],
        "charges": [],
        "cancel_at_period_end": False,
        "canceled_at": None,
        "payment_issue": False,
        "payment_issue_reason": "",
    }


DEFAULT_FEATURE_LIMITS = {
    "free": {"machine": 1, "workflow": 2, "private_model": False},
    "pro": {"machine": 5, "workflow": 10, "private_model": True},
    "creator": {"machine": 10, "workflow": 30, "private_model": True},
    "business": {"machine": 20, "workflow": 100, "private_model": True},
    "enterprise": {"machine": 100, "workflow": 300, "private_model": True},
}

# Map plan keys to feature sets
PLAN_FEATURE_MAPPING = {
    # Legacy plans
    "creator_legacy_monthly": "creator",  # old pro plan
    
    # Current plans
    "creator_monthly": "creator",
    "creator_yearly": "creator",
    "business_monthly": "business",
    "business_yearly": "business",
    "deployment_monthly": "business",  # deployment plans have same limits as business
    "deployment_yearly": "business",
}

def get_feature_set_for_plan(plan_key: str) -> str:
    """Get the feature set name for a given plan key"""
    return PLAN_FEATURE_MAPPING.get(plan_key, "free")

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

    # Handle case where plans is None
    if plans is None:
        plans = {"plans": []}  # Provide default empty plans

    # Check if user has subscription by checking if any plan maps to a paid feature set
    has_subscription = any(
        get_feature_set_for_plan(p) in ["creator", "business"] 
        for p in plans.get("plans", [])
    )
    plan = await get_current_plan(db, user_id, org_id) if has_subscription else None

    # Calculate limits based on plan
    # Get the highest tier feature set from all active plans
    feature_sets = [get_feature_set_for_plan(p) for p in plans.get("plans", [])]
    if "business" in feature_sets:
        machine_max_count = DEFAULT_FEATURE_LIMITS["business"]["machine"]
        workflow_max_count = DEFAULT_FEATURE_LIMITS["business"]["workflow"]
    elif "creator" in feature_sets:
        machine_max_count = DEFAULT_FEATURE_LIMITS["creator"]["machine"]
        workflow_max_count = DEFAULT_FEATURE_LIMITS["creator"]["workflow"]
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

    # Get feature set for current plan
    current_plan_key = plan.get("plan", "free") if plan else "free"
    feature_set = get_feature_set_for_plan(current_plan_key)
    target_plan = DEFAULT_FEATURE_LIMITS.get(feature_set, DEFAULT_FEATURE_LIMITS["free"])

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
    
    if plan in ["creator", "business"]:
        plan = f"{plan}_monthly"

    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")

    # Get current stripe plan
    stripe_plan = await get_stripe_plan(db, user_id, org_id)
    if not stripe_plan:
        return None

    # Find workspace and API plans
    # ws_plan = next(
    #     (
    #         item
    #         for item in stripe_plan.get("items", {}).get("data", [])
    #         if item.get("price", {}).get("id") in [
    #             get_price_id("creator_monthly"),
    #             get_price_id("creator_yearly"),
    #             get_price_id("business_monthly"),
    #             get_price_id("business_yearly"),
    #             get_price_id("deployment_monthly"),
    #             get_price_id("deployment_yearly"),
    #             get_price_id("creator_legacy_monthly"),
    #         ]
    #     ),
    #     None,
    # )

    api_plan = next(
        (
            item
            for item in stripe_plan.get("items", {}).get("data", [])
            if item.get("price", {}).get("id") in [
                get_price_id("creator_monthly"),
                get_price_id("creator_yearly"),
                get_price_id("business_monthly"),
                get_price_id("business_yearly"),
                get_price_id("deployment_monthly"),
                get_price_id("deployment_yearly"),
                get_price_id("creator_legacy_monthly"),
            ]
        ),
        None,
    )

    # Determine conflicting plan
    conflicting_plan = api_plan

    # Get target price ID
    target_price_id = get_price_id(plan)
    if not target_price_id:
        raise HTTPException(status_code=400, detail="Invalid plan type")

    # Check if plan already exists
    has_target_price = any(
        item.get("price", {}).get("id") == target_price_id
        for item in stripe_plan.get("items", {}).get("data", [])
    )
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
                discounts=[{"promotion_code": promotion_code_id}]
                if promotion_code_id
                else [],
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
                discounts=[{"promotion_code": promotion_code_id}]
                if promotion_code_id
                else [],
            )

    except stripe.error.StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return None

clerk_api_key = os.getenv("CLERK_SECRET_KEY")

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
    headers = {
        "Authorization": f"Bearer {clerk_api_key}",
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"https://api.clerk.com/v1/users/{user_id}", headers=headers
        ) as response:
            if response.status == 200:
                return await response.json()
            raise HTTPException(
                status_code=400,
                detail=f"Failed to fetch user data from Clerk: {await response.text()}",
            )

async def get_clerk_org(org_id: str) -> dict:
    headers = {
        "Authorization": f"Bearer {clerk_api_key}",
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"https://api.clerk.com/v1/organizations/{org_id}", headers=headers
        ) as response:
            if response.status == 200:
                return await response.json()
            raise HTTPException(
                status_code=400,
                detail=f"Failed to fetch org data from Clerk: {await response.text()}",
            )

async def slugify(workflow_name: str, current_user_id: str) -> str:
    slug_part = re.sub(
        r'[-\s]+',                                # Replace spaces or multiple dashes with single dash
        '-',
        re.sub(
            r'[^\w\s-]',                          # Remove special chars
            '',
            unicodedata.normalize('NFKD', workflow_name)    # Normalize Unicode
            .encode('ascii', 'ignore')             # Remove non-ascii
            .decode('ascii')                       # Convert back to string
        ).strip().lower()
    ).strip('-')                                  # Remove leading/trailing dashes

    user_part = None
    prefix, _, _ = current_user_id.partition("_")

    if prefix == "org":
        org_data = await get_clerk_org(current_user_id)
        user_part = org_data["slug"].lower()
    elif prefix == "user":
        user_data = await get_clerk_user(current_user_id)
        user_part = user_data["username"].lower()

    return f"{user_part}_{slug_part}"


# Define allowed Stripe webhook events
ALLOWED_STRIPE_EVENTS = [
    "customer.subscription.created",
    "checkout.session.completed",
    "customer.subscription.paused",
    "customer.subscription.resumed",
    "customer.subscription.deleted",
    "customer.subscription.updated",
    "customer.subscription.trial_will_end",
    "invoice.created",
    "invoice.finalized",
    "invoice.paid",
    "payment_intent.succeeded",
    "customer.updated",
    "invoice.payment_succeeded",
    "invoice.updated",
    "charge.succeeded"
]

# PostHog event mapping
POSTHOG_EVENT_MAPPING = {
    "customer.subscription.created": "create_subscription",
    "customer.subscription.paused": "pause_subscription",
    "customer.subscription.resumed": "resume_subscription",
    "customer.subscription.deleted": "delete_subscription",
    "customer.subscription.updated": "update_subscription"
}

async def process_all_active_subscriptions(
    db: AsyncSession,
    dry_run: bool = False,
    send_email: bool = False
) -> List[Dict]:
    """
    Process all active subscriptions and charge for GPU usage.
    This function is meant to be called by a cron job via the admin endpoint.
    
    Args:
        db: Database session
        dry_run: If True, only simulate the process without creating invoices
        send_email: If True, send email notifications about usage
        
    Returns:
        List of processed subscription details
    """
    processed_subscriptions = []
    try:
        # Get all active subscriptions from Stripe
        subscriptions = stripe.Subscription.list(
            status="active",
            expand=["data.customer"]
        )
        
        logfire.info(f"Processing {len(subscriptions.data)} active subscriptions")
        
        for subscription in subscriptions.data:
            try:
                # Get metadata from subscription
                metadata = subscription.metadata
                user_id = metadata.get("userId")
                org_id = metadata.get("orgId")
                
                if not user_id and not org_id:
                    logfire.warning(f"Subscription {subscription.id} has no user_id or org_id in metadata")
                    continue
                
                # Get current plan to ensure Redis data exists and is up to date
                current_plan = await get_current_plan(db, user_id, org_id)
                if not current_plan:
                    logfire.warning(f"No plan data found for subscription {subscription.id}")
                    continue
                
                # Calculate usage charges
                final_cost, last_invoice_timestamp = await calculate_usage_charges(
                    user_id=user_id,
                    org_id=org_id,
                    end_time=datetime.now(),
                    db=db
                )
                
                # Extract only essential customer info
                customer_info = {
                    "id": subscription.customer.id,
                    "email": subscription.customer.email
                } if subscription.customer else {"id": None, "email": None}
                
                subscription_info = {
                    "subscription_id": subscription.id,
                    "customer": customer_info,
                    "user_id": user_id,
                    "org_id": org_id,
                    "final_cost": final_cost,
                    "last_invoice_timestamp": last_invoice_timestamp
                }
                
                if final_cost > 0:
                    # Convert to cents for Stripe
                    amount = int(final_cost * 100)
                    subscription_info["amount_cents"] = amount
                    
                    if not dry_run:
                        # Create invoice item
                        stripe.InvoiceItem.create(
                            customer=subscription.customer.id,
                            amount=amount,
                            currency="usd",
                            description="GPU Compute Usage",
                        )
                        logfire.info(f"Added GPU Compute Usage ({amount} cents) for subscription {subscription.id}")
                    else:
                        logfire.info(f"[DRY RUN] Would add GPU Compute Usage ({amount} cents) for subscription {subscription.id}")
                    
                    if send_email:
                        try:
                            # Get user email from Clerk
                            user_data = None
                            if user_id and user_id.startswith("user_"):
                                user_data = await get_clerk_user(user_id)
                            elif org_id and org_id.startswith("org_"):
                                user_data = await get_clerk_org(org_id)
                                
                            if user_data:
                                email = None
                                if "email_addresses" in user_data:
                                    email = next(
                                        (email["email_address"] for email in user_data["email_addresses"]
                                         if email["id"] == user_data["primary_email_address_id"]),
                                        None
                                    )
                                elif "email" in user_data:
                                    email = user_data["email"]
                                    
                                if email:
                                    message = f"""
                                    <h2>GPU Usage Invoice</h2>
                                    <p>Here's your GPU usage summary:</p>
                                    <ul>
                                        <li>Total Usage Cost: ${final_cost:.2f}</li>
                                        <li>Period End: {datetime.fromtimestamp(last_invoice_timestamp).strftime('%Y-%m-%d %H:%M:%S')}</li>
                                    </ul>
                                    <p>{'This is a dry run notification.' if dry_run else 'An invoice will be generated for this usage.'}</p>
                                    """
                                    params: resend.Emails.SendParams = {
                                        "from": "Comfy Deploy <billing@comfydeploy.com>",
                                        "to": email,
                                        "subject": "Comfy Deploy - GPU Usage Summary" if final_cost == 0 else "Comfy Deploy - GPU Usage Invoice",
                                        "html": message
                                    }
                                    email_result = resend.Emails.send(params)
                                    subscription_info["email_sent"] = True
                                    subscription_info["email_id"] = email_result["id"]
                                    logfire.info(f"Sent usage email to {email}")
                        except Exception as e:
                            logfire.error(f"Error sending email for subscription {subscription.id}: {str(e)}")
                            subscription_info["email_error"] = str(e)
                
                if not dry_run:
                    # Update Redis with new last invoice timestamp
                    await update_subscription_redis_data(
                        subscription_id=subscription.id,
                        user_id=user_id,
                        org_id=org_id,
                        last_invoice_timestamp=last_invoice_timestamp,
                        db=db
                    )
                
                processed_subscriptions.append(subscription_info)
                
            except Exception as e:
                logfire.error(f"Error processing subscription {subscription.id}: {str(e)}")
                continue
                
    except stripe.error.StripeError as e:
        logfire.error(f"Error fetching subscriptions from Stripe: {str(e)}")
        raise
    except Exception as e:
        logfire.error(f"Unexpected error in process_all_active_subscriptions: {str(e)}")
        raise
        
    return processed_subscriptions

# Add Resend client
resend.api_key = os.getenv("RESEND_API_KEY")

@router.post("/platform/stripe/webhook")
async def stripe_webhook(request: Request, db: AsyncSession = Depends(get_db)):
    """Handle Stripe webhook events"""
    try:
        # Get the raw request body
        body = await request.body()
        body_str = body.decode('utf-8')
        
        # Get Stripe signature from headers
        stripe_signature = request.headers.get('stripe-signature')
        if not stripe_signature:
            raise HTTPException(status_code=400, detail="No Stripe signature found")
            
        # Verify webhook signature
        webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
        if not webhook_secret:
            raise HTTPException(status_code=500, detail="Webhook secret not configured")
            
        try:
            event = stripe.Webhook.construct_event(
                body_str,
                stripe_signature,
                webhook_secret
            )
        except stripe.error.SignatureVerificationError:
            raise HTTPException(status_code=400, detail="Invalid signature")
            
        print(f"Webhook event type: {event.get('type')}")
        # print(f"Webhook metadata: {event.get('data', {}).get('object', {}).get('metadata', {})}")
        
        # Use the generic event handler
        await handle_stripe_event(event, db)
        return {"result": event, "ok": True}
        
    except Exception as e:
        print(f"Error processing webhook: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing webhook: {str(e)}"
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
    
    if plan in ["creator", "business"]:
        plan = f"{plan}_monthly"
        
    if not user_id:
        return {"url": redirect_url or "/"}
    if not plan:
        return {"url": redirect_url or "/pricing"}

    # Get user data from Clerk
    user_data = await get_clerk_user(user_id)
    user_email = next(
        (
            email["email_address"]
            for email in user_data["email_addresses"]
            if email["id"] == user_data["primary_email_address_id"]
        ),
        None,
    )

    # Get target price ID
    target_price_id = get_price_id(plan)
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
                    item.get("price", {}).get("id") == target_price_id
                    for item in stripe_plan.get("items", {}).get("data", [])
                )

                if has_existing_plan:
                    # Return portal URL instead of redirecting
                    portal_session = stripe.billing_portal.Session.create(
                        customer=stripe_plan.customer,
                        return_url=redirect_url or "/pricing",
                    )
                    return {"url": portal_session.url}

                # Handle plan updates
                # ws_plan = next(
                #     (
                #         item
                #         for item in stripe_plan.get("items", {}).get("data", [])
                #         if item.get("price", {}).get("id")
                #         in [
                #             get_price_id("creator_monthly"),
                #             get_price_id("creator_yearly"),
                #             get_price_id("business_monthly"),
                #             get_price_id("business_yearly"),
                #             get_price_id("deployment_monthly"),
                #             get_price_id("deployment_yearly"),
                #             get_price_id("creator_legacy_monthly"),
                #         ]
                #     ),
                #     None,
                # )
                api_plan = next(
                    (
                        item
                        for item in stripe_plan.get("items", {}).get("data", [])
                        if item.get("price", {}).get("id")
                        in [
                            get_price_id("creator_monthly"),
                            get_price_id("creator_yearly"),
                            get_price_id("business_monthly"),
                            get_price_id("business_yearly"),
                            get_price_id("deployment_monthly"),
                            get_price_id("deployment_yearly"),
                            get_price_id("creator_legacy_monthly"),
                        ]
                    ),
                    None,
                )

                conflicting_plan = api_plan

                if conflicting_plan:
                    # Update subscription with plan replacement
                    stripe.Subscription.modify(
                        current_plan["subscription_id"],
                        proration_behavior="always_invoice",
                        metadata=metadata,
                        discounts=[{"promotion_code": promotion_code_id}]
                        if promotion_code_id
                        else [],
                        items=[
                            {"id": conflicting_plan.id, "deleted": True},
                            {"price": target_price_id, "quantity": 1},
                        ],
                    )
                else:
                    # Add new plan to subscription
                    stripe.Subscription.modify(
                        current_plan["subscription_id"],
                        proration_behavior="always_invoice",
                        metadata=metadata,
                        discounts=[{"promotion_code": promotion_code_id}]
                        if promotion_code_id
                        else [],
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
            "discounts": [{"promotion_code": promotion_code_id}]
            if promotion_code_id
            else [],
        }

        print(session_params)

        if trial:
            session_params["subscription_data"] = {
                "trial_settings": {
                    "end_behavior": {"missing_payment_method": "cancel"}
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


def r(price: float) -> float:
    return round(price * 1.1, 6)


# GPU pricing per second with 10% margin
PRICING_LOOKUP_TABLE = {
    "T4": r(0.000164),
    "L4": r(0.000291),
    "A10G": r(0.000306),
    "L40S": r(0.000542),
    "A100": r(0.001036),
    "A100-80GB": r(0.001553),
    "H100": r(0.002125),
    "CPU": r(0.000038),
}

FREE_TIER_USAGE = 500  # in cents, $5

@router.get("/platform/gpu-pricing")
async def gpu_pricing():
    """Return the GPU pricing table"""
    return PRICING_LOOKUP_TABLE

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
                extract("epoch", GPUEvent.end_time)
                - extract("epoch", GPUEvent.start_time)
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
            unit_amount = PRICING_LOOKUP_TABLE.get(row.gpu, 0)
            usage_seconds = float(row.usage_in_sec)  # Convert Decimal to float
            grouped_by_date[date_str][row.gpu] = unit_amount * usage_seconds

    # Convert to array format
    chart_data = [
        {"date": date, **gpu_costs} for date, gpu_costs in grouped_by_date.items()
    ]

    return chart_data


async def get_usage_details(
    db: AsyncSession,
    start_time: datetime,
    end_time: datetime,
    org_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> List[Dict]:
    """Get usage details for a given time period"""
    if not user_id and not org_id:
        raise ValueError("User or org id is required")

    # Build the query conditions
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
        
    # Filter out public share workflows
    conditions.append(
        or_(
            GPUEvent.environment != "public-share",
            GPUEvent.environment == None
        )
    )

    # Create the query
    query = (
        select(
            GPUEvent.machine_id,
            Machine.name.label("machine_name"),
            GPUEvent.gpu,
            GPUEvent.ws_gpu,
            func.sum(
                extract("epoch", GPUEvent.end_time)
                - extract("epoch", GPUEvent.start_time)
            ).label("usage_in_sec"),
            GPUEvent.cost_item_title,
            func.sum(func.coalesce(GPUEvent.cost, 0)).label("cost"),
        )
        .select_from(GPUEvent)
        .outerjoin(Machine, GPUEvent.machine_id == Machine.id)
        .where(and_(*conditions))
        .group_by(
            GPUEvent.machine_id,
            Machine.name,
            GPUEvent.gpu,
            GPUEvent.ws_gpu,
            GPUEvent.cost_item_title,
        )
        .order_by(desc("usage_in_sec"))
    )

    result = await db.execute(query)
    usage_details = result.fetchall()

    # Convert to list of dicts
    return [
        {
            "machine_id": row.machine_id,
            "machine_name": row.machine_name,
            "gpu": row.gpu,
            "ws_gpu": row.ws_gpu,
            "usage_in_sec": float(row.usage_in_sec) if row.usage_in_sec else 0,
            "cost_item_title": row.cost_item_title,
            "cost": float(row.cost) if row.cost else (
                # Calculate cost based on GPU type if row.cost is not available
                (float(row.usage_in_sec) / 3600) if row.ws_gpu else  # Workspace GPU cost
                (PRICING_LOOKUP_TABLE.get(row.gpu, 0) * float(row.usage_in_sec) if row.gpu else 0)  # Regular GPU cost
            ),
        }
        for row in usage_details
    ]


def get_gpu_event_cost(event: Dict) -> float:
    """Calculate cost for a GPU event"""
    if event.get("cost_item_title") and event.get("cost") is not None:
        return event["cost"]

    if event.get("ws_gpu"):
        return event["usage_in_sec"] / 3600

    gpu = event.get("gpu")
    if not gpu or gpu not in PRICING_LOOKUP_TABLE:
        return 0

    return PRICING_LOOKUP_TABLE[gpu] * event["usage_in_sec"]


@router.get("/platform/usage")
async def get_usage(
    request: Request,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    db: AsyncSession = Depends(get_db),
):
    """Get usage details and total cost for a given time period"""
    user_id = request.state.current_user.get("user_id")
    org_id = request.state.current_user.get("org_id")

    if not user_id and not org_id:
        raise HTTPException(status_code=400, detail="User or org id is required")

    # Get current subscription
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
        .limit(1)
    )
    result = await db.execute(query)
    subscription = result.scalar_one_or_none()

    # If no subscription, get user creation date
    user_created_at = None
    if not subscription or not subscription.subscription_id:
        query = select(User.created_at).where(User.id == user_id)
        result = await db.execute(query)
        user_created_at = result.scalar_one_or_none()

    # Determine start time based on billing period
    effective_start_time = (
        start_time or subscription.last_invoice_timestamp
        if subscription
        else None or user_created_at
        if user_created_at
        else None or datetime.now()
    )

    effective_end_time = end_time or datetime.now()

    # Get usage details
    usage_details = await get_usage_details(
        db=db,
        start_time=effective_start_time,
        end_time=effective_end_time,
        org_id=org_id,
        user_id=user_id,
    )
    
    user_settings = await get_user_settings_util(request, db)

    # Calculate total cost
    total_cost = sum(get_gpu_event_cost(event) for event in usage_details)

    # Apply free tier credit ($5 = 500 cents)
    final_cost = max(total_cost - FREE_TIER_USAGE / 100, 0)

    return {
        "usage": usage_details,
        "total_cost": total_cost,
        "final_cost": final_cost,
        "free_tier_credit": FREE_TIER_USAGE,  # Convert to dollars
        "credit": user_settings.credit,
        "period": {
            "start": effective_start_time.isoformat(),
            "end": effective_end_time.isoformat(),
        },
    }


@router.get("/platform/invoices")
async def get_monthly_invoices(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Get monthly invoices for the current user/org"""
    user_id = request.state.current_user.get("user_id")
    org_id = request.state.current_user.get("org_id")

    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")

    # Get current plan to get subscription ID
    current_plan = await get_current_plan(db, user_id, org_id)
    if not current_plan or not current_plan.get("subscription_id"):
        return []

    try:
        # Fetch invoices from Stripe
        invoices = stripe.Invoice.list(
            subscription=current_plan["subscription_id"],
            limit=12,
            expand=["data.lines"],
        )

        # Transform invoice data
        return [
            {
                "id": invoice.id,
                "period_start": datetime.fromtimestamp(invoice.period_start).strftime("%Y-%m-%d"),
                "period_end": datetime.fromtimestamp(invoice.period_end).strftime("%Y-%m-%d"),
                "amount_due": invoice.amount_due / 100,  # Convert cents to dollars
                "status": invoice.status,
                "invoice_pdf": invoice.invoice_pdf,
                "line_items": [
                    {
                        "description": item.description,
                        "amount": item.amount / 100,
                        "quantity": item.quantity,
                    }
                    for item in invoice.lines.data
                ],
                "subtotal": invoice.subtotal / 100,
                "total": invoice.total / 100,
            }
            for invoice in invoices.data
        ]

    except stripe.error.StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/platform/stripe/dashboard")
async def get_dashboard_url(
    request: Request,
    redirect_url: str = None,
    db: AsyncSession = Depends(get_db),
):
    user_id = request.state.current_user.get("user_id")
    org_id = request.state.current_user.get("org_id")
    
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    # Get user's subscription
    sub = await get_current_plan(
        db=db,
        org_id=org_id,
        user_id=user_id
    )
    
    if not sub:
        raise HTTPException(status_code=404, detail="No subscription found")

    # Create Stripe billing portal session
    session = stripe.billing_portal.Session.create(
        customer=sub["stripe_customer_id"],
        return_url=redirect_url
    )

    return JSONResponse({
        "url": session.url
    })


async def get_subscription_items(subscription_id: str) -> List[Dict]:
    """Helper function to get subscription items from Stripe"""
    try:
        items = stripe.SubscriptionItem.list(
            subscription=subscription_id,
            limit=5
        )
        return items.data  # Return just the data array
    except stripe.error.StripeError as e:
        print(f"Error fetching subscription items: {e}")
        return []

async def handle_stripe_event(event: dict, db: AsyncSession):
    """Generic event handler for Stripe webhook events"""
    event_type = event.get("type")
    if event_type not in ALLOWED_STRIPE_EVENTS:
        print(f"Unhandled event type: {event_type}")
        return
        
    event_object = event.get("data", {}).get("object", {})
    
    # Extract common fields
    customer_id = event_object.get("customer")
    if isinstance(customer_id, dict):
        customer_id = customer_id.get("id")
        
    # For invoice events, get the subscription from the invoice
    subscription_id = None
    if event_type.startswith("invoice."):
        subscription_id = event_object.get("subscription")
        if subscription_id:
            try:
                # Fetch the full subscription object
                subscription = stripe.Subscription.retrieve(subscription_id)
                event_object = subscription
            except stripe.error.StripeError as e:
                logfire.error(f"Error fetching subscription from invoice: {str(e)}")
                return
    else:
        subscription_id = event_object.get("id")
    
    metadata = event_object.get("metadata", {})
    
    # If metadata is empty in the event object, try to get it from the subscription
    if not metadata and customer_id:
        try:
            # Get customer to find associated subscription
            customer = stripe.Customer.retrieve(customer_id, expand=['subscriptions'])
            if customer.get('subscriptions') and customer.subscriptions.data:
                latest_sub = customer.subscriptions.data[0]
                metadata = latest_sub.metadata
                if not subscription_id:
                    subscription_id = latest_sub.id
                    event_object = latest_sub
        except stripe.error.StripeError as e:
            logfire.error(f"Error fetching customer data: {str(e)}")
    
    user_id = metadata.get("userId")
    org_id = metadata.get("orgId")
    
    # Skip if we can't identify the user/org
    if not user_id and not org_id:
        logfire.error(f"Could not identify user/org from event: {event_type}")
        return
    
    # Handle invoice.created event for GPU usage billing
    last_invoice_timestamp = None
    # if event_type == "invoice.created":
    #     invoice = event.get("data", {}).get("object", {})
        
    #     # Only process subscription cycle invoices
    #     if invoice.get("billing_reason") == "subscription_cycle":
    #         try:
    #             last_invoice_timestamp = await charge_usage_details(
    #                 invoice_id=invoice.get("id"),
    #                 customer_id=customer_id
    #             )
    #             logfire.info(f"Successfully charged GPU usage for invoice {invoice.get('id')}")
    #         except Exception as e:
    #             logfire.error(f"Failed to charge GPU usage: {str(e)}")
    
    try:
        await update_subscription_redis_data(
            subscription_id=subscription_id,
            user_id=user_id,
            org_id=org_id,
            last_invoice_timestamp=last_invoice_timestamp,
            db=db
        )
        logfire.info(f"Updated Redis data for plan:{org_id or user_id} after {event_type}")
    except Exception as e:
        logfire.error(f"Error updating Redis data: {str(e)}")

async def calculate_usage_charges(
    user_id: Optional[str] = None,
    org_id: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    db: Optional[AsyncSession] = None,
) -> Tuple[float, int]:
    """
    Calculate GPU compute usage charges for a given time period.
    Returns a tuple of (final_cost, last_invoice_timestamp).
    This function doesn't depend on Stripe and can be used for testing.
    
    Args:
        user_id: Optional user ID
        org_id: Optional organization ID
        start_time: Start time for usage calculation
        end_time: End time for usage calculation
        db: Optional database session
        
    Returns:
        Tuple[float, int]: (final_cost, last_invoice_timestamp)
        final_cost is in dollars (not cents)
        last_invoice_timestamp is Unix timestamp
    """
    if not user_id and not org_id:
        raise ValueError("Either user_id or org_id must be provided")
        
    if not db:
        async with AsyncSession(get_db()) as db:
            return await calculate_usage_charges(user_id, org_id, start_time, end_time, db)
            
    # Get current plan to ensure Redis data exists and is up to date
    current_plan = await get_current_plan(db, user_id, org_id)
    if not current_plan:
        logfire.warning(f"No plan data found for user {user_id} or org {org_id}")
        return 0, int(datetime.now().timestamp())
            
    # If start_time not provided, use last_invoice_timestamp from current plan
    if not start_time and current_plan.get("last_invoice_timestamp"):
        start_time = datetime.fromtimestamp(current_plan["last_invoice_timestamp"])
            
    # If still no start_time, use subscription start or user creation date
    if not start_time:
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
            .limit(1)
        )
        result = await db.execute(query)
        subscription = result.scalar_one_or_none()
        
        if subscription and subscription.last_invoice_timestamp:
            start_time = subscription.last_invoice_timestamp
        else:
            # Try user creation date
            query = select(User.created_at).where(User.id == user_id)
            result = await db.execute(query)
            user_created_at = result.scalar_one_or_none()
            start_time = user_created_at or datetime.now()
            
    end_time = end_time or datetime.now()
    
    # Get usage details
    usage_details = await get_usage_details(
        db=db,
        start_time=start_time,
        end_time=end_time,
        org_id=org_id,
        user_id=user_id
    )
    
    # Calculate total cost
    total_cost = sum(get_gpu_event_cost(event) for event in usage_details)
    
    # Apply free tier credit ($5 = 500 cents)
    final_cost = max(total_cost - FREE_TIER_USAGE / 100, 0)
    
    return final_cost, int(end_time.timestamp())