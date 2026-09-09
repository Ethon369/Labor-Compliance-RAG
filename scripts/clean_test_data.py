"""清理测试残留数据：空知识库 + 自动生成的测试用户。

为什么会需要：早期测试（test_kb.py / test_api_kb.py / test_auth.py 等）创建了大量
0 文档空库与 `u_`/`tester_`/`smoke_`/`turn_` 开头的用户做断言，跑完不清理，几个月堆
积上千条。这些库全设为 is_public=true，污染所有用户的首页视图。

两件事分开跑（默认只清库，加 --users 才清用户）：
1. 空知识库：只删 0 文档的 kb，保留默认库（id=1，205 法条）。真实用户一旦上传过
   文档，他的库进入"安全区"，下次再跑也不会被误删。
2. 测试用户：删用户名匹配测试自动生成模式（`u_`/`tester_`/`smoke_`/`turn_` +
   hex 后缀）的用户，级联清掉他们的会话/历史/库。admin 与其余用户名永不删。

用法：
    python scripts/clean_test_data.py             # 交互确认：清空知识库
    python scripts/clean_test_data.py --users     # 交互确认：清测试用户
    python scripts/clean_test_data.py --dry-run   # 只看不删
    python scripts/clean_test_data.py --yes       # 跳过确认（CI/重置）
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.core.config import Settings  # noqa: E402
from app.core.db import pool_conn  # noqa: E402

# 默认库 id：205 条法条迁移来的（serial 序列第一条）。永删不得。
DEFAULT_KB_ID = 1
# 测试自动生成用户名的模式：前缀_hex（u_/tester_/smoke_/turn_ 等 + 至少 6 位 hex）
TEST_USER_RE = re.compile(r"^(?:u|tester|smoke|turn)_[0-9a-fA-F]{6,}$")
# 永不删除的用户：系统管理员；其余真实用户名（不匹配测试模式）天然不会被删
ALWAYS_KEEP = {"admin"}


# ---- 空知识库 ----

def preview_kbs(dsn: str) -> int:
    with pool_conn(dsn) as conn:
        return conn.execute(
            "SELECT count(*) FROM kb WHERE id <> %s AND doc_count = 0",
            (DEFAULT_KB_ID,),
        ).fetchone()[0]


def clean_kbs(dsn: str) -> int:
    with pool_conn(dsn) as conn:
        cur = conn.execute("DELETE FROM kb WHERE id <> %s AND doc_count = 0", (DEFAULT_KB_ID,))
    return cur.rowcount


# ---- 测试用户 ----

def preview_users(dsn: str) -> int:
    with pool_conn(dsn) as conn:
        rows = conn.execute("SELECT username FROM users").fetchall()
    return sum(1 for (name,) in rows if TEST_USER_RE.match(name))


def clean_users(dsn: str) -> int:
    """删测试用户（级联清会话/历史/库）。返回删除数。"""
    with pool_conn(dsn) as conn:
        rows = conn.execute("SELECT username FROM users").fetchall()
        victims = [name for (name,) in rows if TEST_USER_RE.match(name)]
        if not victims:
            return 0
        # != ALL 传整表名单：只删 victims，其余（admin/真实用户）全保留
        cur = conn.execute("DELETE FROM users WHERE username != ALL(%s)", (victims,))
    return cur.rowcount


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--users", action="store_true", help="同时清理自动生成的测试用户")
    ap.add_argument("--dry-run", action="store_true", help="只看不删")
    ap.add_argument("--yes", "-y", action="store_true", help="跳过交互确认")
    args = ap.parse_args(argv)

    dsn = Settings().database_url
    n_kbs = preview_kbs(dsn)
    n_users = preview_users(dsn) if args.users else 0
    todo = ["空知识库"] if n_kbs else []
    if args.users:
        todo.append("测试用户")
    print(f"将清理：{n_kbs} 个空知识库" + (f"，{n_users} 个测试用户" if args.users else ""))
    if not todo:
        print("没有可清理的数据。")
        return 0
    if args.dry_run:
        print("[dry-run] 已预览，实际未删")
        return 0
    if not args.yes:
        ans = input(f"确认清理{'、'.join(todo)}？[y/N] ").strip().lower()
        if ans != "y":
            print("已取消")
            return 0

    d_kbs = clean_kbs(dsn) if n_kbs else 0
    d_users = clean_users(dsn) if args.users and n_users else 0
    parts = []
    if n_kbs:
        parts.append(f"空知识库 {d_kbs} 个")
    if args.users and n_users:
        parts.append(f"测试用户 {d_users} 个")
    print(f"已清理：" + "，".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
