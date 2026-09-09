"""FastAPI 依赖：当前用户、管理员校验、知识库读/写权限。

为什么新路由用 Depends 而不是像既有 auth.py 那样每个端点手动调 `_bearer_user_id`：
新路由（/kb、/admin）数量多、权限规则统一，依赖注入让"鉴权"成为签名的一部分——
漏写就通不过类型检查，比"记得在每个函数里调一次"更不容易出错。既有聊天路由保持
手动风格不动（避免为重构而重构）。

越权语义（决策 14）：
- 看不到的资源 → 404（不暴露"它存在"）
- 看得到但不能写 → 403（资源可见，拒绝的是权限）
"""
from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request

from app.api.auth import _bearer_user_id, get_user
from app.kb.store import fetch_kb


@dataclass(frozen=True)
class CurrentUser:
    """已认证用户的最小画像（鉴权只需 id 与角色）。"""
    id: int
    username: str
    role: str

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


def get_current_user(request: Request) -> CurrentUser:
    """从 Authorization 头解析用户；未登录/无效/被禁用一律 401。"""
    settings = request.app.state.settings
    uid = _bearer_user_id(request, settings.auth_secret)
    if uid is None:
        raise HTTPException(status_code=401, detail="未登录或登录已过期")
    user = get_user(settings.database_url, uid)
    if user is None or user["disabled"]:
        raise HTTPException(status_code=401, detail="未登录或登录已过期")
    return CurrentUser(id=user["id"], username=user["username"], role=user["role"])


def require_admin(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    """管理员专属接口。非管理员 403（前端隐藏入口不是安全边界，这里才是）。"""
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return user


def assert_kb_access(kb: dict | None, user: CurrentUser, write: bool = False) -> dict:
    """知识库读/写权限判定（返回 kb 便于调用方接着用）。

    抽出来是为了让"路径参数带 kb_id"的依赖和"只有 doc_id、要反查 kb"的文档端点
    共用同一套语义，避免两处各判一遍、规则漂移。
    """
    if kb is None:
        raise HTTPException(status_code=404, detail="知识库不存在或无权访问")
    if user.is_admin or kb["owner_id"] == user.id:
        return kb
    if not write and kb["is_public"]:
        return kb
    # 私有库他人访问 → 404（不泄露存在性）；公开库他人写 → 403（资源可见，拒的是权限）
    if kb["is_public"]:
        raise HTTPException(status_code=403, detail="无权修改他人的知识库")
    raise HTTPException(status_code=404, detail="知识库不存在或无权访问")


def require_kb_access(write: bool = False):
    """依赖工厂：校验当前用户对路径参数 `kb_id` 指向的知识库的读/写权限。

    可见 = 自己的 / 公开的 / admin；可写 = 自己的 / admin。
    """
    def dep(request: Request,
            user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        raw = request.path_params.get("kb_id")
        if raw is None:
            raise HTTPException(status_code=404, detail="知识库不存在或无权访问")
        settings = request.app.state.settings
        kb = fetch_kb(settings.database_url, int(raw))
        assert_kb_access(kb, user, write=write)
        return user

    return dep
