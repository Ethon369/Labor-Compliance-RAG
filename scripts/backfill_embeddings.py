"""给 chunks 表灌 embedding 向量（幂等，可重复执行）。

用法：
  python scripts/backfill_embeddings.py                 # 默认库，只补 embedding 为空的行
  python scripts/backfill_embeddings.py --force          # 全量重算（换 embedding 提供方后用它）
  python scripts/backfill_embeddings.py --kb-id 3        # 指定知识库

读取 app.core.config 的 Settings（embedding_mode 决定用占位向量还是真 bge-m3），
就绪与写入都走 app.rag.vectorstore.PgVectorStore，脚本只负责"选提供方、取文本、批量写"。
真库连通，失败即非 0 退出。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 让脚本能以"项目根包"方式 import app.*（跑 pytest 时由 pyproject pythonpath 注入，两处殊途同归）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import Settings  # noqa: E402
from app.kb.schema import DEFAULT_KB_ID  # noqa: E402
from app.rag.embedder import build_embedder  # noqa: E402
from app.rag.vectorstore import PgVectorStore  # noqa: E402


def _make_embedder(s: Settings):
    """保留原名供 migrate_default_kb 引用；实现下沉到 app.rag.embedder.build_embedder。"""
    return build_embedder(s)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="全量重算所有行，忽略是否已有向量")
    parser.add_argument("--kb-id", type=int, default=DEFAULT_KB_ID, help="目标知识库 id")
    args = parser.parse_args(argv)

    settings = Settings()
    if not settings.database_url:
        print("缺少 DATABASE_URL（检查 .env / 环境变量）", file=sys.stderr)
        return 1

    store = PgVectorStore(settings.database_url, settings.embedding_dim)
    store.ensure_ready()   # 幂等：启 pgvector 扩展 + 给 chunks 加 embedding 列
    rows = store.pending_rows([args.kb_id], force=args.force)   # (chunk_id, content)

    if not rows:
        print(f"无需回填（{'--force 全量' if args.force else '已全部有向量'}），退出")
        return 0

    embedder = _make_embedder(settings)
    print(f"回填 {len(rows)} 条，embedding={type(embedder).__name__} dim={embedder.dim}")
    vectors = embedder.embed([text for _, text in rows])   # 文本整体成批送，减少提供方往返
    store.set_embeddings([(cid, vec) for (cid, _), vec in zip(rows, vectors)])
    print(f"完成：{len(rows)} 行已写入 embedding 列")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
