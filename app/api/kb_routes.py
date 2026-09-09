"""知识库接口：库的 CRUD、文档上传与状态轮询、重建索引、管理员全局视图。

三条贯穿全文件的约束：

1. **写操作全部经 `require_kb_access(write=True)`**，读操作经 `write=False`。
   前端是否显示"管理"按钮只是体验，真正的边界在这里——漏一个 Depends，
   越权用户就能直接改别人的库。
2. **上传只登记、不等待**：请求里只做"校验扩展名 + 流式落临时盘 + 插 pending 行"，
   解析/切分/向量化交给 BackgroundTasks。用户不用盯着转圈等 30 秒。
3. **路径顺序**：`/kb/admin/all` 与 `/kb/documents/{id}` 必须排在 `/kb/{kb_id}` 之前，
   否则会被 `{kb_id}` 先吃掉（"admin"/"documents" 转不成 int）。
"""
from __future__ import annotations

import logging
import secrets
import tempfile
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Request, UploadFile

from app.api.deps import CurrentUser, assert_kb_access, get_current_user, require_admin, require_kb_access
from app.api.models import (
    CreateKbRequest,
    DocumentBrief,
    KbBrief,
    UpdateKbRequest,
)
from app.core.config import Settings
from app.kb import pipeline
from app.kb import store
from app.kb.parser import detect_source_type
from app.kb.schema import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from app.rag.embedder import build_embedder
from app.rag.vectorstore import PgVectorStore

router = APIRouter(prefix="/kb", tags=["kb"])
logger = logging.getLogger(__name__)

# 单文件上限：解析前先看大小，比解析到一半再失败更省资源（也避免撑爆临时盘）
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
_READ_CHUNK = 1024 * 1024
_TMP_DIR = Path(tempfile.gettempdir()) / "labor_kb_uploads"


# ---------------------------------------------------------------------------
# 知识库列表 / 新建（不依赖路径里的 kb_id）
# ---------------------------------------------------------------------------

@router.get("")
async def list_my_kbs(request: Request, page: int = 1, page_size: int = 20,
                      user: CurrentUser = Depends(get_current_user)):
    """我可见的知识库：自己的 + 公开的（admin 是全部）。"""
    dsn = request.app.state.settings.database_url
    rows, total = store.list_kbs(dsn, uid=user.id, page=page, page_size=page_size)
    return {"kbs": [KbBrief(**r) for r in rows], "total": total,
            "page": page, "page_size": page_size}


@router.post("")
async def create_kb(req: CreateKbRequest, request: Request,
                    user: CurrentUser = Depends(get_current_user)):
    dsn = request.app.state.settings.database_url
    kb_id = store.create_kb(
        dsn, user.id, req.name, description=req.description, is_public=req.is_public,
        chunk_size=req.chunk_size or DEFAULT_CHUNK_SIZE,
        chunk_overlap=req.chunk_overlap or DEFAULT_CHUNK_OVERLAP,
    )
    return {"id": kb_id, "name": req.name}


@router.get("/admin/all")
async def list_all_kbs(request: Request, page: int = 1, page_size: int = 20,
                       _admin: CurrentUser = Depends(require_admin)):
    """平台管理视图：全部知识库（含他人的私有库）。"""
    dsn = request.app.state.settings.database_url
    rows, total = store.list_kbs(dsn, uid=_admin.id, page=page, page_size=page_size)
    return {"kbs": [KbBrief(**r) for r in rows], "total": total,
            "page": page, "page_size": page_size}


# ---------------------------------------------------------------------------
# 文档级端点（只有 doc_id，权限靠反查所属库的规则判定）
# ---------------------------------------------------------------------------

def _doc_access(request: Request, doc_id: int, user: CurrentUser, write: bool) -> dict:
    """取文档并按其所属库校验权限，返回文档行。"""
    dsn = request.app.state.settings.database_url
    doc = store.get_document(dsn, doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="文档不存在或无权访问")
    kb = store.fetch_kb(dsn, doc["kb_id"])
    assert_kb_access(kb, user, write=write)     # 与 /kb/{kb_id} 系列同一套语义
    return doc


@router.get("/documents/{doc_id}")
async def get_document_status(doc_id: int, request: Request,
                              user: CurrentUser = Depends(get_current_user)):
    """轮询处理状态（前端到终态 ready/failed 停止）。"""
    return DocumentBrief(**_doc_access(request, doc_id, user, write=False))


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: int, request: Request,
                          user: CurrentUser = Depends(get_current_user)):
    _doc_access(request, doc_id, user, write=True)
    dsn = request.app.state.settings.database_url
    store.delete_document(dsn, doc_id)      # chunks 靠外键级联删，BM25 指纹自动失效
    return {"ok": True, "id": doc_id}


# ---------------------------------------------------------------------------
# 单个知识库
# ---------------------------------------------------------------------------

@router.get("/{kb_id}")
async def get_kb(request: Request, user: CurrentUser = Depends(require_kb_access())):
    dsn = request.app.state.settings.database_url
    kb_id = int(request.path_params["kb_id"])
    return KbBrief(**store.fetch_kb(dsn, kb_id))


@router.patch("/{kb_id}")
async def update_kb(req: UpdateKbRequest, request: Request,
                    _user: CurrentUser = Depends(require_kb_access(write=True))):
    dsn = request.app.state.settings.database_url
    kb_id = int(request.path_params["kb_id"])
    fields = req.model_dump(exclude_none=True)
    if not store.update_kb(dsn, kb_id, **fields):
        raise HTTPException(status_code=400, detail="没有可更新的字段")
    return {"ok": True, "id": kb_id}


@router.delete("/{kb_id}")
async def delete_kb(request: Request, _user: CurrentUser = Depends(require_kb_access(write=True))):
    dsn = request.app.state.settings.database_url
    kb_id = int(request.path_params["kb_id"])
    store.delete_kb(dsn, kb_id)
    return {"ok": True, "id": kb_id}


@router.get("/{kb_id}/documents")
async def list_documents(request: Request, page: int = 1, page_size: int = 20,
                         _user: CurrentUser = Depends(require_kb_access())):
    dsn = request.app.state.settings.database_url
    kb_id = int(request.path_params["kb_id"])
    rows, total = store.list_documents(dsn, kb_id, page=page, page_size=page_size)
    return {"documents": [DocumentBrief(**r) for r in rows], "total": total,
            "page": page, "page_size": page_size}


@router.post("/{kb_id}/documents")
async def upload_document(request: Request, background: BackgroundTasks,
                          file: UploadFile = File(...),
                          _user: CurrentUser = Depends(require_kb_access(write=True))):
    """上传文档：登记 pending 行 → 后台流水线处理 → 立即返回 doc_id 供轮询。

    流式读盘而不是 `await file.read()` 整体进内存：后者等于把内存上限交给客户端，
    10MB 的文件 × 并发上传足以把进程顶翻。
    """
    dsn = request.app.state.settings.database_url
    kb_id = int(request.path_params["kb_id"])

    source_type = detect_source_type(file.filename or "")
    if source_type is None:
        raise HTTPException(status_code=400, detail="不支持的文件类型（pdf/docx/txt/md）")

    _TMP_DIR.mkdir(parents=True, exist_ok=True)
    # 只用原始文件名的 basename 做标题，落盘用自己生成的名字——不让用户输入参与路径拼接
    title = Path(file.filename or "unnamed").name
    tmp_path = _TMP_DIR / f"{secrets.token_hex(8)}_{title}"

    size = 0
    try:
        with tmp_path.open("wb") as out:
            while True:
                chunk = await file.read(_READ_CHUNK)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="文件超过 10MB 上限")
                out.write(chunk)
    except HTTPException:
        tmp_path.unlink(missing_ok=True)
        raise
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="文件写入失败")

    if size == 0:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="文件为空")

    doc_id = store.create_document(dsn, kb_id, title, source_type, file_size=size)
    kb = store.fetch_kb(dsn, kb_id)
    background.add_task(pipeline.run, dsn, doc_id, str(tmp_path), source_type, kb)
    return {"id": doc_id, "kb_id": kb_id, "title": title,
            "status": "pending", "size": size}


@router.post("/{kb_id}/reindex")
async def reindex(request: Request, background: BackgroundTasks,
                  _user: CurrentUser = Depends(require_kb_access(write=True))):
    """重建本库向量（换 embedding 提供方、或想强制刷新时用）。

    只按已入库的切片内容重算向量，不重新解析原文件——原文件在入库后就删了。
    """
    settings = request.app.state.settings
    kb_id = int(request.path_params["kb_id"])
    background.add_task(_reindex_task, settings.database_url, kb_id,
                        settings.embedding_dim)
    return {"ok": True, "kb_id": kb_id}


def _reindex_task(dsn: str, kb_id: int, dim: int) -> None:
    """后台重算向量。异常同样不外抛（后台任务没人接），只记录日志。"""
    vs = PgVectorStore(dsn, dim)
    rows = vs.pending_rows([kb_id], force=True)      # (chunk_id, content)
    if not rows:
        return
    try:
        embedder = build_embedder(Settings())
        vectors = embedder.embed([text for _, text in rows])
        vs.set_embeddings([(cid, vec) for (cid, _), vec in zip(rows, vectors)])
    except Exception:
        logger.exception("知识库 %s 重建索引失败", kb_id)
