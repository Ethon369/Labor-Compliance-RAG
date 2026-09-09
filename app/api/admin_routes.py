"""管理员接口：用户列表、升降级、重置密码、禁用/解禁。

全部经 `require_admin` 依赖把关——前端隐藏入口只是体验，后端鉴权才是边界。
两条自保规则：管理员不能改自己的角色、不能禁用自己（否则可能把自己锁在门外）。
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.auth import list_users, reset_password, set_disabled, set_role
from app.api.deps import CurrentUser, require_admin
from app.api.models import ChangeRoleRequest, ResetPasswordRequest, SetDisabledRequest

router = APIRouter(prefix="/admin", tags=["admin"])

_MAX_PAGE_SIZE = 100


@router.get("/users")
async def admin_list_users(request: Request, page: int = 1, page_size: int = 20,
                           _admin: CurrentUser = Depends(require_admin)):
    """分页列出全部用户（含角色与禁用状态）。"""
    page = max(1, page)
    page_size = min(max(1, page_size), _MAX_PAGE_SIZE)
    settings = request.app.state.settings
    records, total = list_users(settings.database_url, page_size, (page - 1) * page_size)
    return {"users": records, "total": total, "page": page, "page_size": page_size}


@router.patch("/users/{uid}/role")
async def admin_change_role(uid: int, req: ChangeRoleRequest, request: Request,
                            admin: CurrentUser = Depends(require_admin)):
    """升降级用户角色。"""
    if uid == admin.id:
        raise HTTPException(status_code=400, detail="不能修改自己的角色")
    settings = request.app.state.settings
    if not set_role(settings.database_url, uid, req.role):
        raise HTTPException(status_code=404, detail="用户不存在")
    return {"ok": True, "id": uid, "role": req.role}


@router.post("/users/{uid}/reset-password")
async def admin_reset_user_password(uid: int, req: ResetPasswordRequest, request: Request,
                                    _admin: CurrentUser = Depends(require_admin)):
    """重置他人密码（不需要旧密码）。"""
    settings = request.app.state.settings
    if not reset_password(settings.database_url, uid, req.new_password):
        raise HTTPException(status_code=404, detail="用户不存在")
    return {"ok": True}


@router.post("/users/{uid}/disabled")
async def admin_set_user_disabled(uid: int, req: SetDisabledRequest, request: Request,
                                  admin: CurrentUser = Depends(require_admin)):
    """禁用/解禁账号。禁用后该用户无法登录（已有 token 也会在下次请求时被拒）。"""
    if uid == admin.id:
        raise HTTPException(status_code=400, detail="不能禁用自己的账号")
    settings = request.app.state.settings
    if not set_disabled(settings.database_url, uid, req.disabled):
        raise HTTPException(status_code=404, detail="用户不存在")
    return {"ok": True, "id": uid, "disabled": req.disabled}
