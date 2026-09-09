"""pytest 公共配置：让 tests 能以顶层模块方式 import scripts/ 下的脚本。

scripts/ 下的文件彼此以"同目录平级 import"的方式引用（如 ingest_laws.py
import fetch_laws），跑 pytest 时把 scripts/ 提前塞进 sys.path 保持行为一致。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


@pytest.fixture(autouse=True)
def _reset_rate_limits():
    """每个测试前清空限流状态：限流是进程级单例，不 reset 会让连续测试互相踩 429。"""
    from app.core.ratelimit import _limiter, login_guard
    _limiter.reset_all()
    login_guard.reset_all()
    yield
