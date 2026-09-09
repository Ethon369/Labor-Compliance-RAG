"""登录/注册路由 + 密码哈希 + 签名 token。零第三方依赖（标准库）。

为什么自制签名 token 而不是引 JWT 库：锁定清单（CLAUDE.md 规则 1）禁止新增依赖；
HMAC-SHA256 签名 + 过期时间戳对这个规模够用，且原理能用三段话讲清（payload →
base64url → HMAC 签名），面试更好讲。密码用 PBKDF2-HMAC-SHA256 加盐哈希，
同为标准库，比明文存密码强一个量级。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time

import psycopg
from fastapi import APIRouter, HTTPException, Request

from app.api.models import AuthRequest, AuthResponse, ChangePasswordRequest, MeResponse

router = APIRouter(prefix="/auth", tags=["auth"])

_TOKEN_TTL = 7 * 24 * 3600   # token 有效期：7 天（演示够用；要严格再收紧）
_PBKDF2_ITER = 100_000        # PBKDF2 迭代次数：标准推荐量级


# ---- 密码哈希（工具函数，离线可单测） ----

def hash_password(password: str) -> str:
    """PBKDF2-SHA256 加盐哈希，产出 "salt$digest" 格式。"""
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), _PBKDF2_ITER)
    return f"{salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """校验密码。compare_digest 防时序侧信道；畸形存储一律判 False（不抛 500）。"""
    try:
        salt, digest = stored.split("$", 1)
        calc = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), _PBKDF2_ITER)
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(calc.hex(), digest)


# ---- 签名 token（工具函数，离线可单测） ----

def make_token(user_id: int, secret: str, ttl: int = _TOKEN_TTL) -> str:
    """payload(uid,exp) → base64url → HMAC-SHA256 签名，拼成 token。"""
    payload = {"uid": user_id, "exp": int(time.time()) + ttl}
    p = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    sig = hmac.new(secret.encode(), p.encode(), hashlib.sha256).hexdigest()
    return f"{p}.{sig}"


def verify_token(token: str, secret: str) -> int | None:
    """校验签名 + 过期，返回 user_id；无效返回 None（任何异常都视为无效）。"""
    try:
        p, sig = token.split(".", 1)
    except ValueError:
        return None
    expect = hmac.new(secret.encode(), p.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expect):
        return None
    try:
        data = json.loads(base64.urlsafe_b64decode(p))
    except Exception:
        return None
    if data.get("exp", 0) < time.time():
        return None
    return data.get("uid")


def _bearer_user_id(request: Request, secret: str) -> int | None:
    """从 Authorization 头解析 Bearer token；无 header 或无效返回 None。"""
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        return None
    return verify_token(auth[len("Bearer "):], secret)


# ---- 建表 + 落库（幂等；沿用 vectorstore 的短连接风格） ----

def ensure_user_schema(dsn: str) -> None:
    """建 users / chat_history 两张表；CREATE IF NOT EXISTS，可反复调用。"""
    with psycopg.connect(dsn) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id            serial PRIMARY KEY,
                username      text NOT NULL UNIQUE,
                password_hash text NOT NULL,
                created_at    timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_history (
                id         serial PRIMARY KEY,
                user_id    integer NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                question   text NOT NULL,
                answer     text NOT NULL,
                citations  text NOT NULL DEFAULT '[]',
                created_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        # 多轮会话（阶段 10 增强）：一个会话包含多条 user/assistant 消息
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id         serial PRIMARY KEY,
                user_id    integer NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title      text NOT NULL DEFAULT '新对话',
                created_at timestamptz NOT NULL DEFAULT now(),
                updated_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id         serial PRIMARY KEY,
                session_id integer NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
                role       text NOT NULL CHECK (role IN ('user', 'assistant')),
                content    text NOT NULL,
                citations  text NOT NULL DEFAULT '[]',
                created_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )


def ensure_admin(dsn: str, username: str, password: str) -> int:
    """幂等确保管理员账号存在，返回 uid。

    - 用户名不存在 → 建号并置 role='admin'
    - 已存在 → 只把 role 补成 admin，**绝不覆盖已改过的密码**
      （否则每次重启都会把管理员改过的密码打回默认值，是真实事故来源）

    依赖 users.role 列已存在——调用前须先跑 ensure_user_schema + ensure_kb_schema。
    """
    with psycopg.connect(dsn) as conn:
        row = conn.execute(
            "SELECT id, role FROM users WHERE username = %s", (username,)
        ).fetchone()
        if row is None:
            created = conn.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (%s, %s, 'admin') "
                "RETURNING id",
                (username, hash_password(password)),
            ).fetchone()
            return created[0]
        uid, role = row
        if role != "admin":
            conn.execute("UPDATE users SET role = 'admin' WHERE id = %s", (uid,))
        return uid


def get_user(dsn: str, uid: int) -> dict | None:
    """按 id 取用户（含角色与禁用状态），不存在返回 None。"""
    with psycopg.connect(dsn) as conn:
        row = conn.execute(
            "SELECT id, username, role, disabled, created_at FROM users WHERE id = %s", (uid,)
        ).fetchone()
    if row is None:
        return None
    return {"id": row[0], "username": row[1], "role": row[2], "disabled": row[3],
            "created_at": row[4].isoformat()}


def change_password(dsn: str, uid: int, old_password: str, new_password: str) -> bool:
    """校验旧密码后更新为新密码；旧密码不对返回 False（不区分"用户不存在"）。"""
    with psycopg.connect(dsn) as conn:
        row = conn.execute("SELECT password_hash FROM users WHERE id = %s", (uid,)).fetchone()
        if row is None or not verify_password(old_password, row[0]):
            return False
        conn.execute(
            "UPDATE users SET password_hash = %s WHERE id = %s",
            (hash_password(new_password), uid),
        )
    return True


def list_users(dsn: str, limit: int, offset: int) -> tuple[list[dict], int]:
    """分页列出用户（新注册的在前）。返回 (records, total)。"""
    with psycopg.connect(dsn) as conn:
        total = conn.execute("SELECT count(*) FROM users").fetchone()[0]
        rows = conn.execute(
            "SELECT id, username, role, disabled, created_at FROM users "
            "ORDER BY id DESC LIMIT %s OFFSET %s",
            (limit, offset),
        ).fetchall()
    return [
        {"id": r[0], "username": r[1], "role": r[2], "disabled": r[3],
         "created_at": r[4].isoformat()}
        for r in rows
    ], total


def set_role(dsn: str, uid: int, role: str) -> bool:
    """改角色；用户不存在返回 False。"""
    with psycopg.connect(dsn) as conn:
        cur = conn.execute("UPDATE users SET role = %s WHERE id = %s", (role, uid))
        return cur.rowcount > 0


def reset_password(dsn: str, uid: int, new_password: str) -> bool:
    """管理员重置密码（不需要旧密码）；用户不存在返回 False。"""
    with psycopg.connect(dsn) as conn:
        cur = conn.execute(
            "UPDATE users SET password_hash = %s WHERE id = %s",
            (hash_password(new_password), uid),
        )
        return cur.rowcount > 0


def set_disabled(dsn: str, uid: int, disabled: bool) -> bool:
    """禁用/解禁账号；用户不存在返回 False。"""
    with psycopg.connect(dsn) as conn:
        cur = conn.execute("UPDATE users SET disabled = %s WHERE id = %s", (disabled, uid))
        return cur.rowcount > 0


def save_chat_history(dsn: str, user_id: int, question: str, answer: str, citations: list[dict]) -> None:
    """登录用户的一条问答落库。citations 存 JSON 文本，读取时再解析。"""
    with psycopg.connect(dsn) as conn:
        conn.execute(
            "INSERT INTO chat_history (user_id, question, answer, citations) VALUES (%s, %s, %s, %s)",
            (user_id, question, answer, json.dumps(citations, ensure_ascii=False)),
        )


def load_user_history(dsn: str, user_id: int, limit: int) -> tuple[list[dict], int]:
    """该用户最近 limit 条问答（倒序，最新在前）。返回 (records, total)。"""
    with psycopg.connect(dsn) as conn:
        total = conn.execute(
            "SELECT count(*) FROM chat_history WHERE user_id = %s", (user_id,)
        ).fetchone()[0]
        rows = conn.execute(
            "SELECT question, answer, citations FROM chat_history "
            "WHERE user_id = %s ORDER BY id DESC LIMIT %s",
            (user_id, limit),
        ).fetchall()
    records = [
        {"question": q, "answer": a, "citations": json.loads(c) if c else []}
        for q, a, c in rows
    ]
    return records, total


# ---- 多轮会话工具函数 ----

def create_session(dsn: str, user_id: int, title: str = "新对话",
                   kb_id: int | None = None) -> int:
    """创建会话，返回 session_id。kb_id 记下这个会话问答时检索哪个知识库。"""
    with psycopg.connect(dsn) as conn:
        row = conn.execute(
            "INSERT INTO chat_sessions (user_id, title, kb_id) VALUES (%s, %s, %s) RETURNING id",
            (user_id, title, kb_id),
        ).fetchone()
    return row[0]


def session_kb_id(dsn: str, session_id: int) -> int | None:
    """读会话绑定的知识库（可能为 NULL——库被删或从未绑定）。"""
    with psycopg.connect(dsn) as conn:
        row = conn.execute(
            "SELECT kb_id FROM chat_sessions WHERE id = %s", (session_id,)
        ).fetchone()
    return row[0] if row else None


def save_session_message(dsn: str, session_id: int, role: str, content: str, citations: list[dict] | None = None) -> None:
    """在指定会话中保存一条消息，并更新会话 updated_at。"""
    with psycopg.connect(dsn) as conn:
        conn.execute(
            "INSERT INTO chat_messages (session_id, role, content, citations) VALUES (%s, %s, %s, %s)",
            (session_id, role, content, json.dumps(citations or [], ensure_ascii=False)),
        )
        conn.execute("UPDATE chat_sessions SET updated_at = now() WHERE id = %s", (session_id,))


def list_user_sessions(dsn: str, user_id: int, limit: int = 50) -> list[dict]:
    """该用户最近 limit 个会话（倒序，最新更新在前）。"""
    with psycopg.connect(dsn) as conn:
        rows = conn.execute(
            "SELECT id, title, updated_at FROM chat_sessions WHERE user_id = %s "
            "ORDER BY updated_at DESC LIMIT %s",
            (user_id, limit),
        ).fetchall()
    return [{"id": r[0], "title": r[1], "updated_at": r[2].isoformat()} for r in rows]


def load_session_messages(dsn: str, session_id: int, user_id: int | None = None) -> list[dict]:
    """读取某会话的全部消息；若提供 user_id 则做拥有权校验。"""
    with psycopg.connect(dsn) as conn:
        if user_id is not None:
            owner = conn.execute(
                "SELECT user_id FROM chat_sessions WHERE id = %s", (session_id,)
            ).fetchone()
            if owner is None or owner[0] != user_id:
                return []
        rows = conn.execute(
            "SELECT id, role, content, citations, created_at FROM chat_messages "
            "WHERE session_id = %s ORDER BY id ASC",
            (session_id,),
        ).fetchall()
    return [
        {"id": r[0], "role": r[1], "content": r[2], "citations": json.loads(r[3]) if r[3] else [], "created_at": r[4].isoformat()}
        for r in rows
    ]


# ---- 路由 ----

@router.post("/register")
async def register(req: AuthRequest, request: Request):
    """注册：用户名查重 → 哈希入库 → 直接返回 token（免二次登录）。新用户恒为普通用户。"""
    settings = request.app.state.settings
    with psycopg.connect(settings.database_url) as conn:
        dup = conn.execute("SELECT 1 FROM users WHERE username = %s", (req.username,)).fetchone()
        if dup:
            raise HTTPException(status_code=409, detail="用户名已存在")
        row = conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (%s, %s) RETURNING id",
            (req.username, hash_password(req.password)),
        ).fetchone()
        user_id = row[0]
    token = make_token(user_id, settings.auth_secret)
    return AuthResponse(token=token, username=req.username, role="user")


@router.post("/login")
async def login(req: AuthRequest, request: Request):
    """登录：查用户 → 验密码 → 发 token。

    密码错与账号被禁用都返回 401 同一句话——不暴露"这个账号存在但被禁了"，
    否则等于给攻击者一个枚举有效用户名的信号。"""
    settings = request.app.state.settings
    with psycopg.connect(settings.database_url) as conn:
        row = conn.execute(
            "SELECT id, password_hash, role, disabled FROM users WHERE username = %s",
            (req.username,),
        ).fetchone()
    if row is None or row[3] or not verify_password(req.password, row[1]):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = make_token(row[0], settings.auth_secret)
    return AuthResponse(token=token, username=req.username, role=row[2])


@router.get("/me")
async def me(request: Request):
    """凭 token 查当前用户（前端登录态恢复 + 角色判断用）；无效 → 401。"""
    settings = request.app.state.settings
    uid = _bearer_user_id(request, settings.auth_secret)
    if uid is None:
        raise HTTPException(status_code=401, detail="未登录或登录已过期")
    user = get_user(settings.database_url, uid)
    if user is None or user["disabled"]:
        raise HTTPException(status_code=401, detail="用户不存在")
    return MeResponse(id=user["id"], username=user["username"], role=user["role"])


@router.post("/change-password")
async def change_password_endpoint(req: ChangePasswordRequest, request: Request):
    """修改自己的密码：验旧密码 → 更新。旧密码错 400（这里已确认用户身份，不必模糊化）。"""
    settings = request.app.state.settings
    uid = _bearer_user_id(request, settings.auth_secret)
    if uid is None:
        raise HTTPException(status_code=401, detail="未登录或登录已过期")
    if not change_password(settings.database_url, uid, req.old_password, req.new_password):
        raise HTTPException(status_code=400, detail="原密码不正确")
    return {"ok": True}
