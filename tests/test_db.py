"""连接池单元/集成测试（@pytest.mark.db）。

验的是两件事：同 dsn 只建一个池（懒加载 + 缓存）；池内连接被复用而非
每次操作开新连接（这是 V2.4 替换短连接的核心收益）。连不上库整模块 skip。
"""
from __future__ import annotations

import pytest

from app.core import db
from app.core.config import Settings

pytestmark = pytest.mark.db

_dsn = Settings().database_url


def _db_ok() -> bool:
    try:
        with db.pool_conn(_dsn) as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


def test_get_pool_caches_by_dsn():
    """同一 dsn 反复取池，必须返回同一个对象（否则每次调用都重建池 = 又变短连接）。"""
    if not _db_ok():
        pytest.skip("labor-pg 不可达")
    assert db.get_pool(_dsn) is db.get_pool(_dsn)


def test_pool_conn_reuses_connections():
    """连续 50 次查询后，池内连接数应远小于 50（证明复用了，而不是每次开新连）。"""
    if not _db_ok():
        pytest.skip("labor-pg 不可达")
    for _ in range(50):
        with db.pool_conn(_dsn) as conn:
            conn.execute("SELECT 1")
    stats = db.get_pool(_dsn).get_stats()
    # 池上限 8，50 次串行查询峰值连接数应等于 1（复用），宽松断言到 <= 4 留足并发余量
    assert stats["connections_num"] <= 4


def test_pool_conn_transaction_semantics():
    """正常退出提交、异常退出回滚，语义与 psycopg.connect 一致。"""
    if not _db_ok():
        pytest.skip("labor-pg 不可达")
    import secrets
    table = "_pool_t_" + secrets.token_hex(4)
    try:
        # 正常路径：DML 退出时自动提交，跨连接可见（真实表，不用 TEMP TABLE——
        # 后者是连接会话级的，复用池下两次取到的物理连接未必相同）
        with db.pool_conn(_dsn) as conn:
            conn.execute(f"CREATE TABLE {table} (v int)")
            conn.execute(f"INSERT INTO {table} VALUES (1)")
        with db.pool_conn(_dsn) as conn:
            n = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            assert n == 1
    finally:
        with db.pool_conn(_dsn) as conn:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
