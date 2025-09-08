import logging
from fnmatch import fnmatch
from typing import Dict, Any, Optional

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from api.utils.autumn import autumn_client
from api.utils.feature_gate import AUTUMN_GUARD_RULES
from api.routes.platform import get_customer_plan_cached


logger = logging.getLogger(__name__)


class AutumnAccessMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        try:
            if await self._should_block(request):
                # If blocked, _should_block already returns a JSONResponse; but
                # to keep the interface clear, we return from here.
                # However, the method returns bool, so we construct the response here.
                return JSONResponse(status_code=403, content={"detail": "Access denied for requested operation"})

            response = await call_next(request)
            return response
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    async def _should_block(self, request: Request) -> bool:
        path = request.url.path
        method = request.method.upper()

        # Must have authenticated user loaded by AuthMiddleware
        current_user: Optional[Dict[str, Any]] = getattr(request.state, "current_user", None)
        if not current_user:
            return False

        # Identify customer id (org preferred)
        customer_id = current_user.get("org_id") or current_user.get("user_id")
        if not customer_id:
            return False

        plan = current_user.get("plan")
        # logger.info(f"AutumnAccessMiddleware: {plan}")

        # Try each rule until one applies
        for rule in AUTUMN_GUARD_RULES:
            if method not in {m.upper() for m in rule.get("methods", [])}:
                continue

            pattern = rule.get("pattern")
            if not pattern or not fnmatch(path, pattern):
                continue

            # Skip excluded paths
            for ex in rule.get("exclude", []) or []:
                if fnmatch(path, ex):
                    return False

            # Only apply for free plan if configured
            only_free = rule.get("only_free", True)
            if only_free:
                effective_plan = plan
                if not effective_plan:
                    try:
                        plan_data = await get_customer_plan_cached(customer_id)
                        if plan_data and plan_data.get("plans"):
                            effective_plan = plan_data["plans"][0]
                        else:
                            effective_plan = "free"
                    except Exception as e:
                        logger.error(f"Failed to resolve plan for {customer_id}: {e}")
                        effective_plan = None

                if effective_plan and effective_plan != "free":
                    # Paid plans: skip this rule
                    continue

            checks = rule.get("checks", [])
            mode = rule.get("mode", "all")

            results = []
            for check in checks:
                feature_id = check.get("feature_id")
                required_balance = int(check.get("required_balance", 1))

                if not feature_id:
                    # Ill-formed rule; skip gracefully
                    logger.warning("Autumn guard rule missing feature_id for pattern %s", pattern)
                    continue

                try:
                    res = await autumn_client.check(
                        customer_id=customer_id,
                        feature_id=feature_id,
                        required_balance=required_balance,
                        # with_preview=True,
                    )
                except Exception as e:
                    # If Autumn is misconfigured or unreachable, fail-open (allow).
                    logger.error(f"Autumn check failed for {feature_id}: {e}")
                    res = None

                # If result is None (e.g., misconfiguration or network), allow request
                allowed = True if res is None else bool(res.get("allowed") is True)
                results.append((allowed, res))

            if not results:
                # No usable checks means no block
                continue

            if mode == "any":
                decision = any(allowed for allowed, _ in results)
            else:
                decision = all(allowed for allowed, _ in results)

            if not decision:
                # Pick a failing check response to get the feature type
                failing = next((res for allowed, res in results if not allowed and res), None)
                
                # Return simple message based on feature type
                if failing and failing.get("feature_id") == "gpu-credit":
                    message = "Insufficient balance for requested operation"
                else:
                    message = "Access denied for requested operation"

                raise HTTPException(status_code=403, detail=message)

        return False
