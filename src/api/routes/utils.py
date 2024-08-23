from typing import Any, Self, TypeVar, Tuple
from fastapi import Request
from sqlalchemy import GenerativeSelect, Select
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql.selectable import _ColumnsClauseArgument

Base = declarative_base()

T = TypeVar("T")

def get_org_or_user_condition(target: Base, request: Request):
    current_user = request.state.current_user
    user_id = current_user['user_id']
    org_id = current_user['org_id']
    
    return (
        (target.org_id == org_id)
        if org_id
        else ((target.user_id == user_id) & (target.org_id.is_(None)))
    )

class OrgAwareSelect(Select[Tuple[T]]):
    inherit_cache=True
    
    def apply_org_check(self, request: Request) -> Self:
        return self.where(get_org_or_user_condition(self.column_descriptions[0]['entity'], request))

def select(__ent0: _ColumnsClauseArgument[T], /, *entities: Any) -> OrgAwareSelect[T]:
    return OrgAwareSelect(__ent0, *entities)