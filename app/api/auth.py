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

from app.api.models import AuthRequest, AuthResponse

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

def create_session(dsn: str, user_id: int, title: str = "新对话") -> int:
    """创建会话，返回 session_id。"""
    with psycopg.connect(dsn) as conn:
        row = conn.execute(
            "INSERT INTO chat_sessions (user_id, title) VALUES (%s, %s) RETURNING id",
            (user_id, title),
        ).fetchone()
    return row[0]


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
    """注册：用户名查重 → 哈希入库 → 直接返回 token（免二次登录）。"""
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
    return AuthResponse(token=token, username=req.username)


@router.post("/login")
async def login(req: AuthRequest, request: Request):
    """登录：查用户 → 验密码 → 发 token。密码错统一 401（不暴露用户是否存在）。"""
    settings = request.app.state.settings
    with psycopg.connect(settings.database_url) as conn:
        row = conn.execute(
            "SELECT id, password_hash FROM users WHERE username = %s", (req.username,)
        ).fetchone()
    if row is None or not verify_password(req.password, row[1]):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = make_token(row[0], settings.auth_secret)
    return AuthResponse(token=token, username=req.username)


@router.get("/me")
async def me(request: Request):
    """凭 token 查当前用户名（前端登录态恢复用）；无效 → 401。"""
    settings = request.app.state.settings
    uid = _bearer_user_id(request, settings.auth_secret)
    if uid is None:
        raise HTTPException(status_code=401, detail="未登录或登录已过期")
    with psycopg.connect(settings.database_url) as conn:
        row = conn.execute("SELECT username FROM users WHERE id = %s", (uid,)).fetchone()
    if row is None:
        raise HTTPException(status_code=401, detail="用户不存在")
    return {"username": row[0]}
