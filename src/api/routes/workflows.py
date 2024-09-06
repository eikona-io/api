import json
import logging
from fastapi import APIRouter, Depends, HTTPException, Request
from typing import List, Optional
from .types import WorkflowListResponse, WorkflowModel
from api.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from .utils import select, post_process_outputs
from api.models import Deployment, User, Workflow, WorkflowRun, WorkflowVersion
from .utils import get_user_settings
from sqlalchemy import func, select as sa_select, distinct, and_, or_
from sqlalchemy.orm import joinedload, load_only, contains_eager
from fastapi.responses import JSONResponse
from pprint import pprint
from sqlalchemy import desc
from sqlalchemy import text
from datetime import datetime
from uuid import UUID

logger = logging.getLogger(__name__)

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, UUID)):
            return str(obj)
        return super().default(obj)

def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime, UUID)):
        return str(obj)
    raise TypeError(f"Type {type(obj)} not serializable")

router = APIRouter(tags=["workflows"])


@router.get("/workflows", response_model=List[WorkflowModel])
async def get_workflows(
    request: Request,
    search: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    raw_query = text("""
    WITH RECURSIVE search_param(term) AS (
        SELECT lower(:search)
    ),
    filtered_workflows AS (
        SELECT id
        FROM comfyui_deploy.workflows
        WHERE ((CAST(:org_id AS TEXT) IS NOT NULL AND org_id = CAST(:org_id AS TEXT))
            OR (CAST(:org_id AS TEXT) IS NULL AND org_id IS NULL AND user_id = CAST(:user_id AS TEXT)))
            AND (CAST(:search AS TEXT) IS NULL OR lower(name) LIKE '%' || (SELECT term FROM search_param) || '%')
    ),
    latest_versions AS (
        SELECT 
            workflow_id,
            MAX(version) AS max_version
        FROM 
            comfyui_deploy.workflow_versions
        WHERE workflow_id IN (SELECT id FROM filtered_workflows)
        GROUP BY 
            workflow_id
    ),
    recent_runs AS (
        SELECT DISTINCT ON (wr.workflow_id)
            wr.workflow_id,
            wr.created_at AS latest_run_at,
            wr.status,
            wro.data AS latest_output
        FROM 
            comfyui_deploy.workflow_runs wr
        LEFT JOIN LATERAL (
            SELECT data
            FROM comfyui_deploy.workflow_run_outputs
            WHERE run_id = wr.id
            ORDER BY created_at DESC
            LIMIT 1
        ) wro ON true
        WHERE wr.workflow_id IN (SELECT id FROM filtered_workflows)
        ORDER BY 
            wr.workflow_id, wr.created_at DESC
    )
    SELECT
        wf.id AS id,
        wf.name AS name, 
        wf.created_at AS created_at, 
        wf.updated_at AS updated_at, 
        vr.version AS latest_version, 
        users.name AS user_name,
        users.id AS user_id,
        rr.latest_run_at,
        rr.status,
        rr.latest_output
    FROM 
        comfyui_deploy.workflows AS wf
    INNER JOIN 
        latest_versions ON wf.id = latest_versions.workflow_id
    INNER JOIN 
        comfyui_deploy.workflow_versions AS vr 
            ON wf.id = vr.workflow_id AND vr.version = latest_versions.max_version
    INNER JOIN 
        comfyui_deploy.users AS users ON users.id = vr.user_id
    LEFT JOIN 
        recent_runs AS rr ON wf.id = rr.workflow_id
    WHERE 
        wf.id IN (SELECT id FROM filtered_workflows)
    ORDER BY 
        wf.pinned DESC,
        wf.updated_at DESC
    LIMIT :limit
    OFFSET :offset
    """)

    current_user = request.state.current_user
    user_id = current_user["user_id"]
    org_id = current_user["org_id"]

    # Execute the query
    result = await db.execute(
        raw_query,
        {
            "search": search,
            "limit": limit,
            "offset": offset,
            "org_id": org_id,
            "user_id": user_id,
        },
    )

    workflows = [
        dict(row._mapping)
        for row in result.fetchall()
    ]

    # Use the custom encoder to serialize the data
    return JSONResponse(
        status_code=200, 
        content=json.loads(json.dumps(workflows, cls=CustomJSONEncoder))
    )
