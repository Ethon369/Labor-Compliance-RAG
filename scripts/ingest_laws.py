"""把 data/raw/*.json（fetch_laws 的产物）以条款为单位写入 PostgreSQL。

条款即 chunk：一条法文一行，articles 表主键 (law_id, article_no) 天然支持
PRD 验收里的"按条号精确检索"。表结构与演进方向见 docs/decisions/02-入库表结构.md。

用法：
    python scripts/ingest_laws.py --dry-run   # 不碰数据库，只校验 + 打印入库计划
    python scripts/ingest_laws.py             # 真入库（连接串从 .env 的 DATABASE_URL 读）

建表是幂等的（CREATE TABLE IF NOT EXISTS），入库是 upsert（ON CONFLICT），
整文重跑安全。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from psycopg import connect
from psycopg.types.json import Jsonb
from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

import fetch_laws

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JSON_DIR = ROOT / "data" / "raw"

# ---------------------------------------------------------------------------
# 配置：一律来自环境变量 / .env（无硬编码）；Settings 结构后续阶段原样复用
# ---------------------------------------------------------------------------
class Settings(BaseSettings):
    database_url: str = ""

    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8")


# ---------------------------------------------------------------------------
# Schema：DDL 单独成常量——建表幂等可重跑，且便于在测试里静态断言
# 参数全部走 psycopg 占位符（%s），禁止任何 f-string 拼 SQL
# ---------------------------------------------------------------------------
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS laws (
    law_id     text PRIMARY KEY,
    name       text NOT NULL,
    version    text NOT NULL DEFAULT '',
    source     jsonb,
    fetched_at timestamptz
);

CREATE TABLE IF NOT EXISTS articles (
    law_id     text NOT NULL REFERENCES laws(law_id) ON DELETE CASCADE,
    article_no integer NOT NULL CHECK (article_no >= 1),
    chapter    text NOT NULL DEFAULT '',
    -- 条款文本即检索单元；阶段 2 加 pgvector embedding 列（见决策记录 02）
    text       text NOT NULL,
    PRIMARY KEY (law_id, article_no)
);
"""

UPSERT_LAW_SQL = """
INSERT INTO laws (law_id, name, version, source, fetched_at)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (law_id) DO UPDATE SET
    name = EXCLUDED.name,
    version = EXCLUDED.version,
    source = EXCLUDED.source,
    fetched_at = EXCLUDED.fetched_at
"""

UPSERT_ARTICLE_SQL = """
INSERT INTO articles (law_id, article_no, chapter, text)
VALUES (%s, %s, %s, %s)
ON CONFLICT (law_id, article_no) DO UPDATE SET
    chapter = EXCLUDED.chapter,
    text = EXCLUDED.text
"""


def load_json_dir(json_dir: Path) -> list[fetch_laws.Law]:
    """读 data/raw/*.json 并用 fetch_laws 的 Pydantic 模型校验。"""
    laws: list[fetch_laws.Law] = []
    for path in sorted(json_dir.glob("*.json")):
        try:
            laws.append(fetch_laws.Law.model_validate_json(path.read_text("utf-8")))
        except ValidationError as e:
            raise ValueError(f"{path}: JSON 与模型不符: {e}") from e
    if not laws:
        raise ValueError(f"{json_dir}: 没有找到 *.json")
    return laws


def dry_run_plan(laws: list[fetch_laws.Law]) -> dict[str, int]:
    """不连库的入库计划：统计每部法条数，供 dry-run 与测试断言。"""
    return {law.law_id: len(law.articles) for law in laws}


def init_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
    conn.commit()


def ingest(conn, laws: list[fetch_laws.Law]) -> int:
    """upsert 两部法 + 全部条款；返回写入（含覆盖）的行数。"""
    rows = 0
    with conn.cursor() as cur:
        for law in laws:
            cur.execute(UPSERT_LAW_SQL, [
                law.law_id, law.name, law.version,
                Jsonb(law.source.model_dump(mode="json")),
                law.source.fetched_at,
            ])
            article_rows = [
                (law.law_id, a.no, a.chapter, a.text) for a in law.articles
            ]
            cur.executemany(UPSERT_ARTICLE_SQL, article_rows)
            rows += len(article_rows)
    conn.commit()
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json-dir", type=Path, default=DEFAULT_JSON_DIR,
                    help=f"fetch_laws 产物目录（默认 {DEFAULT_JSON_DIR}）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只校验并打印计划，不连数据库")
    args = ap.parse_args(argv)

    try:
        laws = load_json_dir(args.json_dir)
    except ValueError as e:
        print(f"[fail] {e}", file=sys.stderr)
        return 1

    plan = dry_run_plan(laws)
    print("入库计划: " + ", ".join(f"{law.name} {n} 条"
                                    for law, n in zip(laws, plan.values())))
    if args.dry_run:
        return 0

    settings = Settings()
    if not settings.database_url:
        print("[fail] 未设置 DATABASE_URL（.env 或环境变量），先用 --dry-run 检查",
              file=sys.stderr)
        return 1
    try:
        with connect(settings.database_url) as conn:
            init_schema(conn)
            rows = ingest(conn, laws)
    except Exception as e:                      # psycopg 连接/执行错误统包一层
        print(f"[fail] 入库失败: {e}", file=sys.stderr)
        return 1
    print(f"[ok] 写入 {rows} 条条款（ON CONFLICT 覆盖，可重复执行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
