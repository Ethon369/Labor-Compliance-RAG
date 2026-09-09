"""清理测试残留数据：删除 0 文档的空知识库（保留默认库与所有非空库）。

为什么会需要：早期测试（test_kb.py / test_api_kb.py）创建了大量 0 文档的"空库"
做断言，跑完没清理，几个月下来堆积成百上千条，全是 `u_` 开头的测试用户 + 固定测试
名字（"公开库"/"分页库0"等）。这些库全设为 is_public=true 污染了所有用户的首页视图。

清理策略：只删 0 文档的 kb + 保留默认库（id=1，劳动法知识库，205 片）。这样：
- 测试残留（必是 0 文档）全被清
- 真实用户一旦上传了文档，他的库就进入"安全区"，下次再跑清理也不会被误删

用法：
    python scripts/clean_test_data.py            # 交互式确认
    python scripts/clean_test_data.py --dry-run  # 只看不删（预览将删多少）
    python scripts/clean_test_data.py --yes      # 跳过确认（CI/重置场景）
"""
from __future__ import annotations

import argparse

_PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_PROJECT_ROOT))

from app.core.config import Settings  # noqa: E402
from app.core.db import pool_conn  # noqa: E402

# 默认库 id：205 条法条迁移来的，id 必为 1（serial 序列第一条）。保留。
DEFAULT_KB_ID = 1


def preview(dsn: str) -> tuple[int, list[tuple[int, str, int, bool]]]:
    """返回 (会被删的总数, 前若干条预览 [(id, name, owner_id, is_public)])。"""
    with pool_conn(dsn) as conn:
        rows = conn.execute(
            """
            SELECT k.id, k.name, k.owner_id, k.is_public
            FROM kb k
            WHERE k.id <> %s AND k.doc_count = 0
            ORDER BY k.id
            """,
            (DEFAULT_KB_ID,),
        ).fetchall()
    return len(rows), [(r[0], r[1], r[2], r[3]) for r in rows]


def run(dsn: str) -> int:
    total, sample = preview(dsn)
    print(f"将删除 {total} 个 0 文档的空知识库（保留默认库 id={DEFAULT_KB_ID}）")
    if sample:
        print("预览前 10 条：")
        for r in sample[:10]:
            print(f"  id={r[0]:<5d} owner_id={r[2]:<4d} public={r[3]!s:<5s} name={r[1]!r}")
    with pool_conn(dsn) as conn:
        cur = conn.execute("DELETE FROM kb WHERE id <> %s AND doc_count = 0", (DEFAULT_KB_ID,))
    return cur.rowcount


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="只看不删")
    ap.add_argument("--yes", "-y", action="store_true", help="跳过交互确认")
    args = ap.parse_args(argv)

    dsn = Settings().database_url
    total, _ = preview(dsn)
    if total == 0:
        print("没有可清理的 0 文档知识库。")
        return 0

    if args.dry_run:
        print(f"[dry-run] 将删除 {total} 个 0 文档知识库（实际未删）")
        return 0

    if not args.yes:
        ans = input(f"确认删除 {total} 个空知识库？[y/N] ").strip().lower()
        if ans != "y":
            print("已取消")
            return 0

    deleted = run(dsn)
    print(f"已删除 {deleted} 个空知识库")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
