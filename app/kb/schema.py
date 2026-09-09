"""多知识库三层模型的 DDL 与幂等就绪（kb → documents → chunks）。

为什么是三层而不是"知识库直接挂切片"：删单个文档时只删它的切片、不牵连整库重建；
文档级处理状态（pending/parsing/ready/failed）可追踪；引用能定位到"哪个文档的哪一段、
第几页"。法律场景尤其吃这一点——回答要能说清出处。

与既有 laws/articles 表的关系：articles 继续作为默认知识库的**迁移数据源**保留，
检索层不再读它（见决策 11）。所以本文件的 ensure_kb_schema 必须在 ensure_user_schema
之后执行（kb.owner_id 外键指向 users），且必须在 PgVectorStore.ensure_ready 之前
（后者要给 chunks 加 embedding 列）。
"""
from __future__ import annotations

from app.core.db import pool_conn

# 默认知识库固定 id=1：内置两部法律迁移进来，游客与未选库的会话都落到它上面。
# 用常量而非"查第一个库"是因为迁移脚本、检索缺省值、测试都要引用同一个身份。
DEFAULT_KB_ID = 1

# 切分参数默认值：kb 表的列默认值与 splitter 的函数默认值共用这两个常量，
# 避免"表里写 500、代码里写 300"这种漂移（建库时未指定参数的新库就取它们）。
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 80

KB_DDL = f"""
CREATE TABLE IF NOT EXISTS kb (
    id            serial PRIMARY KEY,
    name          text NOT NULL,
    description   text NOT NULL DEFAULT '',
    owner_id      integer NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    -- 公开库：他人可见、可问答，但只有 owner/admin 能改
    is_public     boolean NOT NULL DEFAULT false,
    -- 切分参数随库走：不同语料（法条 / 长文档）适合的片长不同，写在库里而不是全局配置
    chunk_size    integer NOT NULL DEFAULT {DEFAULT_CHUNK_SIZE},
    chunk_overlap integer NOT NULL DEFAULT {DEFAULT_CHUNK_OVERLAP},
    embed_model   text NOT NULL DEFAULT '',
    doc_count     integer NOT NULL DEFAULT 0,
    chunk_count   integer NOT NULL DEFAULT 0,
    created_at    timestamptz NOT NULL DEFAULT now(),
    updated_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS documents (
    id            serial PRIMARY KEY,
    kb_id         integer NOT NULL REFERENCES kb(id) ON DELETE CASCADE,
    title         text NOT NULL,
    source_type   text NOT NULL CHECK (source_type IN ('pdf', 'docx', 'txt', 'md', 'text')),
    -- 仅内置法律迁移的文档有值（labor_law / labor_contract_law），上传文档为 NULL。
    -- 它是 eval gold 映射与"引用显示第 N 条还是第 N 段"的权威来源，不硬编码在代码里。
    source_law_id text,
    file_size     integer,
    status        text NOT NULL DEFAULT 'pending'
                  CHECK (status IN ('pending', 'parsing', 'embedding', 'ready', 'failed')),
    error         text,
    chunk_count   integer NOT NULL DEFAULT 0,
    created_at    timestamptz NOT NULL DEFAULT now(),
    updated_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id           bigserial PRIMARY KEY,
    kb_id        integer NOT NULL REFERENCES kb(id) ON DELETE CASCADE,
    doc_id       integer NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    seq          integer NOT NULL,          -- 片在文档内的顺序（内置法条=条号，可逆）
    page         integer,                   -- PDF 页码，其它来源为 NULL
    heading      text NOT NULL DEFAULT '',  -- 最近的上级标题，补上下文
    content      text NOT NULL,
    char_count   integer,
    -- 内容指纹：重传/reindex 时未变的片跳过重算 embedding（V2.4 启用）
    content_hash text,
    created_at   timestamptz NOT NULL DEFAULT now(),
    UNIQUE (doc_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_kb_owner    ON kb(owner_id);
CREATE INDEX IF NOT EXISTS idx_doc_kb      ON documents(kb_id);
CREATE INDEX IF NOT EXISTS idx_chunk_kb    ON chunks(kb_id);
CREATE INDEX IF NOT EXISTS idx_chunk_doc   ON chunks(doc_id);
"""

# 既有表的增量改造：全部 ADD COLUMN IF NOT EXISTS / CREATE INDEX IF NOT EXISTS，可反复执行
KB_ALTERS = """
ALTER TABLE users ADD COLUMN IF NOT EXISTS role text NOT NULL DEFAULT 'user'
    CHECK (role IN ('user', 'admin'));

-- 禁用：保留账号与历史（不删数据），但拒绝登录。比"删除用户"温和，管理员可反悔
ALTER TABLE users ADD COLUMN IF NOT EXISTS disabled boolean NOT NULL DEFAULT false;

-- ON DELETE SET NULL：删库不能被历史会话的外键阻断；该会话检索时回退默认库
ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS kb_id integer
    REFERENCES kb(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_sessions_user_updated ON chat_sessions(user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_session      ON chat_messages(session_id, id);
"""


def ensure_kb_schema(dsn: str) -> None:
    """建三层表 + 既有表增量列。幂等，可反复调用（启动时执行）。

    单事务执行：DDL 在 PG 里是事务性的，中途失败不会留下半套表。
    """
    with pool_conn(dsn) as conn:
        conn.execute(KB_DDL)
        conn.execute(KB_ALTERS)
