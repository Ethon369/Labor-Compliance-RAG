"""幂等建立 chunks 的 HNSW 向量索引，并支持建前/建后跑评测量化对比。

用法：
    python scripts/ensure_hnsw.py            # 建索引（不存在才建，可反复跑）
    python scripts/ensure_hnsw.py --check     # 只看当前索引状态，不改任何东西

为什么单独一个脚本而不是放进 ensure_ready()：建 HNSW 是**有代价**的——近似索引
在数据量小时可能反而略降召回（近似 vs 精确顺序扫描），且建索引本身会锁表、耗时。
所以它属于"运维动作"，不该在应用每次启动时悄悄做，而是由人显式触发，跑完评测
确认召回没退化再上线（对比方法见 docs/decisions/17）。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.core.config import Settings  # noqa: E402
from app.core.db import pool_conn  # noqa: E402

INDEX_NAME = "chunks_embedding_hnsw"


def index_exists(dsn: str) -> bool:
    with pool_conn(dsn) as conn:
        row = conn.execute(
            "SELECT 1 FROM pg_indexes WHERE tablename = 'chunks' AND indexname = %s",
            (INDEX_NAME,),
        ).fetchone()
    return row is not None


def create_index(dsn: str) -> bool:
    """建 HNSW 余弦索引。返回是否新建（False = 已存在）。"""
    if index_exists(dsn):
        return False
    with pool_conn(dsn) as conn:
        # m=16（默认，召回/内存平衡点）、ef_construction=64（默认）。数据量小，
        # 参数不必调优；索引的作用是让近邻查询走索引而非全表扫。
        conn.execute(
            f"CREATE INDEX {INDEX_NAME} ON chunks "
            "USING hnsw (embedding vector_cosine_ops)"
        )
    return True


def check(dsn: str) -> None:
    with pool_conn(dsn) as conn:
        exists = index_exists(dsn)
        n = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        embedded = conn.execute(
            "SELECT count(*) FROM chunks WHERE embedding IS NOT NULL").fetchone()[0]
    print(f"HNSW 索引 {INDEX_NAME}: {'已存在' if exists else '不存在'}")
    print(f"chunks 总数 {n}，有向量 {embedded}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="只看索引状态，不建")
    args = ap.parse_args(argv)

    dsn = Settings().database_url
    if args.check:
        check(dsn)
        return 0

    t0 = time.perf_counter()
    created = create_index(dsn)
    dt = time.perf_counter() - t0
    if created:
        print(f"已建 HNSW 索引 {INDEX_NAME}（{dt:.2f}s）")
    else:
        print(f"HNSW 索引 {INDEX_NAME} 已存在，跳过")
    check(dsn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
