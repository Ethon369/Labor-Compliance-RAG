"""ingest_laws 的离线测试：JSON 校验、dry-run 计划、SQL 常量、配置读取。

真实数据库路径（connect/建表/upsert）不进 pytest，走 docker 冒烟脚本——
本文件保证不设 DATABASE_URL 时整条链路也全绿。
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

import ingest_laws as ig

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


def test_dry_run_on_committed_data():
    """端到端闭环：fetch 产物 JSON → 模型校验 → 入库计划的条数与 PRD 一致。"""
    laws = ig.load_json_dir(RAW)
    plan = ig.dry_run_plan(laws)
    assert plan == {"labor_law": 107, "labor_contract_law": 98}


def test_load_json_rejects_invalid(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"law_id": 1}', encoding="utf-8")          # 字段类型不符
    with pytest.raises(ValueError, match="JSON 与模型不符"):
        ig.load_json_dir(tmp_path)


def test_load_json_empty_dir(tmp_path):
    with pytest.raises(ValueError, match="没有找到"):
        ig.load_json_dir(tmp_path)


def test_upsert_sql_uses_placeholders_not_fstring():
    """防注入底线：SQL 一律 %s 占位 + ON CONFLICT；若改成 f-string 拼接立即红。"""
    for sql in (ig.UPSERT_LAW_SQL, ig.UPSERT_ARTICLE_SQL):
        assert "%s" in sql
        assert "ON CONFLICT" in sql
        assert "{" not in sql.replace("%s", "")     # 参数一律占位符，禁止字面拼值
    assert "PRIMARY KEY (law_id, article_no)" in ig.SCHEMA_SQL


def test_missing_database_url_fails_cleanly(capsys, monkeypatch):
    """无 DATABASE_URL 时给可读提示（指向 dry-run），而不是炸 stacktrace。

    把 Settings 换成恒为空串的桩——避免本机 .env 真实存在时误读到配置。
    """
    monkeypatch.setattr(ig, "Settings", lambda: type("S", (), {"database_url": ""})())
    rc = ig.main(["--json-dir", str(RAW)])
    assert rc == 1
    assert "DATABASE_URL" in capsys.readouterr().err
