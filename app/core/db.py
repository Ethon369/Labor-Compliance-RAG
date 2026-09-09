"""连接池：psycopg_pool 懒建、按 dsn 缓存，替换"每次查询开一条短连接"。

为什么需要池：改造前 auth / vectorstore / kb store / 聊天路由每执行一条 SQL 都
`psycopg.connect(dsn)` 建连、用完即弃。一条 SSE 问答要触达十几条 SQL，每次建连都要
走 TCP 握手 + PG 认证 + 后端进程 fork，延迟叠加后是最大瓶颈（决策 02 提过、V2.4 落地）。

为什么"按 dsn 缓存"而不是"全局单池"：测试用不同 dsn 连不同临时库，单池会串库。
用 dict 按 dsn 分池，生产环境只有一条 dsn，退化为单池，行为一致。
"""
from __future__ import annotations

from contextlib import contextmanager
from threading import Lock
from typing import Iterator

import psycopg
from psycopg_pool import ConnectionPool

# 默认池上限：≥ 并发问答线程数 + 上传并发，避免请求排队等连接。
# 单 worker 演示规模下 8 足够；真要横向扩需先评估 PG 后端 max_connections。
POOL_SIZE = 8

_pools: dict[str, ConnectionPool] = {}
_lock = Lock()


def get_pool(dsn: str) -> ConnectionPool:
    """取（必要时懒建）某 dsn 的连接池。线程安全：双重检查 + 锁内创建。"""
    pool = _pools.get(dsn)
    if pool is not None:
        return pool
    with _lock:
        pool = _pools.get(dsn)
        if pool is None:
            # open=False 再显式 open：首次调用才真正建连，配合懒加载语义。
            # timeout=30s：池满时 connection() 最多等 30 秒而非无限挂起。
            pool = ConnectionPool(dsn, min_size=1, max_size=POOL_SIZE,
                                  open=False, timeout=30)
            pool.open()
            _pools[dsn] = pool
    return pool


@contextmanager
def pool_conn(dsn: str) -> Iterator[psycopg.Connection]:
    """取一条池内连接，退出时归还。与 psycopg.connect 的 with 语义对齐：
    正常退出提交事务、异常退出回滚，调用方无需改事务处理逻辑。"""
    pool = get_pool(dsn)
    with pool.connection() as conn:
        yield conn


def close_pools() -> None:
    """关闭全部池（进程退出/测试收尾用）。"""
    with _lock:
        for pool in _pools.values():
            pool.close()
        _pools.clear()
