"""文档入库流水线：解析 → 切分 → 向量化 → 落库。

为什么单独一层而不塞进路由或 store：这四个步骤各自可独立测试（解析器喂字节流、
切分器喂 Block 流、store 喂 dict），流水线只负责"按顺序串起来 + 把状态机走完"。

**契约：本函数永不抛异常**。它是 BackgroundTasks 里跑的后台任务——抛出去的异常没人
接、没人看，文档会永远卡在 pending。所以任何失败都要落进 `documents.error` +
`status='failed'`，让轮询接口和用户界面能看见原因。
"""
from __future__ import annotations

import logging
from pathlib import Path

from app.core.config import Settings
from app.kb import store
from app.kb.parser import UnsupportedContentError, parse_file
from app.kb.splitter import hash_content, split_document
from app.rag.embedder import build_embedder

logger = logging.getLogger(__name__)


def run(dsn: str, doc_id: int, path: str | Path, source_type: str, kb_row: dict) -> None:
    """处理一个已登记的文档，把状态推进到 ready 或 failed。

    kb_row：kb 行（至少含 id / chunk_size / chunk_overlap）——切分参数随库走，
    流水线不读全局配置，换库即换粒度。
    """
    path = Path(path)
    try:
        doc = store.get_document(dsn, doc_id)
        if doc is None:
            logger.error("文档 %s 不存在，跳过处理", doc_id)
            return

        store.set_document_status(dsn, doc_id, "parsing")
        parsed = parse_file(path, source_type)

        store.set_document_status(dsn, doc_id, "embedding")
        chunks = split_document(
            parsed,
            chunk_size=kb_row.get("chunk_size") or 500,
            overlap=kb_row.get("chunk_overlap") or 80,
        )
        if not chunks:
            # 解析出了文字却切不出片：宁可标 failed 也不要留一个 chunk_count=0 的 ready
            store.set_document_status(dsn, doc_id, "failed",
                                      error="解析后没有可入库的内容片段")
            return

        embedder = build_embedder(Settings())
        vectors = embedder.embed([c.content for c in chunks])

        payload = [
            {
                "seq": c.seq,
                "content": c.content,
                "heading": c.heading,
                "page": c.page,
                "char_count": c.char_count,
                "content_hash": hash_content(c.content),
            }
            for c in chunks
        ]
        # 内容与向量同事务写入：不留"内容已入库但还没有向量"的中间窗口
        store.replace_chunks(dsn, kb_row["id"], doc_id, payload, embeddings=vectors)
        store.set_document_status(dsn, doc_id, "ready", chunk_count=len(chunks))
        logger.info("文档 %s 入库完成：%s 片", doc_id, len(chunks))

    except UnsupportedContentError as exc:
        # 扫描件等"文件能开但取不到字"的情况：不是系统故障，提示用户即可
        store.set_document_status(dsn, doc_id, "failed", error=str(exc))
        logger.warning("文档 %s 无文字内容：%s", doc_id, exc)
    except Exception as exc:                      # noqa: BLE001 - 兜底是设计，不是偷懒
        store.set_document_status(dsn, doc_id, "failed", error=f"{type(exc).__name__}: {exc}")
        logger.exception("文档 %s 入库失败", doc_id)
    finally:
        # 临时文件由流水线负责清理：上传端点写盘、后台任务消费，职责成对了才不会漏
        path.unlink(missing_ok=True)
