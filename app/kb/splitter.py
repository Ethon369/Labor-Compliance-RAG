"""文档切分：解析出的 Block 流 → 可入库的片段（带 seq/heading/page）。

策略是"递归退让"：先按语义边界切（标题分段 → 段落 → 句子），只有在某一段连
句子都切不开时才硬切字符。为什么执着于语义边界：半句话既伤检索（向量语义被
拦腰截断），也伤引用（用户看到的是残缺条文）。硬切是兜底，不是常态。

切分参数（chunk_size/overlap）由调用方从 kb 行传入，不读全局配置——法条适合整条
成片（短且自洽），上传的长文档要更小的片才不至于"一片里混了三个主题"。粒度是
**库的属性**，不是应用的属性。
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

from app.kb.parser import Block, ParsedDoc
from app.kb.schema import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE

# 切分边界，按"语义粒度从粗到细"排列：换行 → 中文/英文句末标点。
# 用后向断言切分，让标点留在前一句末尾（引用时读起来完整）。
_SEPARATORS = (
    re.compile(r"(?<=\n)"),
    re.compile(r"(?<=[。！？；!?;])"),
)


@dataclass(frozen=True)
class TextChunk:
    """一个待入库的片段。seq 从 1 起，在文档内连续（DB 里 (doc_id, seq) 唯一）。"""
    seq: int
    content: str
    heading: str = ""
    page: int | None = None

    @property
    def char_count(self) -> int:
        return len(self.content)


def hash_content(text: str) -> str:
    """内容指纹。用 md5 而非 sha256：这里只做"内容变没变"的等值判断，
    不涉及对抗性场景，md5 足够且更快（V2.4 重传去重要用）。
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def split_document(parsed: ParsedDoc, chunk_size: int = DEFAULT_CHUNK_SIZE,
                   overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[TextChunk]:
    """把 ParsedDoc 切成片段列表。空文档返回空列表（调用方按 failed 处理）。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须为正整数")
    # 重叠取到片长的一半为上限：再大就会"下一片几乎等于上一片"，切片无法向前推进
    overlap = min(max(overlap, 0), chunk_size // 2)

    chunks: list[TextChunk] = []
    seq = 0
    for heading, blocks in _sections(parsed.blocks):
        text = "\n".join(b.text for b in blocks)
        # 片跨页时只记起始页：精确到"每个字符属于第几页"需要逐块追踪，
        # 收益仅是引用里的页码更准，不值这个复杂度（见决策 15 已知取舍）
        page = next((b.page for b in blocks if b.page is not None), None)
        for content in _merge(_atomize(text, chunk_size), chunk_size, overlap):
            seq += 1
            chunks.append(TextChunk(seq=seq, content=content, heading=heading, page=page))
    return chunks


def _clean_heading(text: str) -> str:
    """标题行去装饰：Markdown 的 `#` 前缀不该进 heading 字段（正文里仍保留原样）。"""
    return text.lstrip("#").strip() or text


def _sections(blocks: list[Block]):
    """按标题切段：yield (最近上级标题, 该标题下的块列表)。

    标题行本身留在它自己那一段里——它是正文的一部分，丢掉会让片段看不出
    "这段属于哪一章"。
    """
    heading = ""
    buf: list[Block] = []
    for block in blocks:
        if block.is_heading:
            if buf:
                yield heading, buf
                buf = []
            heading = _clean_heading(block.text)
        buf.append(block)
    if buf:
        yield heading, buf


def _atomize(text: str, chunk_size: int) -> list[str]:
    """把文本拆成不超过 chunk_size 的原子片段（段落 → 句子 → 硬切，逐级退让）。"""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    for pattern in _SEPARATORS:
        parts = [p.strip() for p in pattern.split(text)]
        parts = [p for p in parts if p]
        if len(parts) > 1:
            out: list[str] = []
            for part in parts:
                out.extend(_atomize(part, chunk_size))  # 子片段可能仍超长，继续退让
            return out
    # 无任何可用边界（例如一整串没有标点的编号表）：只能按字符硬切
    return _hard_cut(text, chunk_size)


def _hard_cut(text: str, chunk_size: int) -> list[str]:
    """均分而非"切满再留残片"：250 字切 100，直接切得 100/100/50，均分得 84/84/82。

    50 字的半截片在检索里几乎没有价值（既不像一句完整的话，也没有独立语义），
    而均分后每片都接近上限。上取整保证单片仍不超过 chunk_size。
    """
    n = math.ceil(len(text) / chunk_size)
    size = math.ceil(len(text) / n)
    return [text[i:i + size] for i in range(0, len(text), size)]


def _merge(pieces: list[str], chunk_size: int, overlap: int) -> list[str]:
    """贪心合并原子片段，并在片间补 overlap 尾缀。保证每片长度 ≤ chunk_size。"""
    merged: list[str] = []
    for piece in pieces:
        if not merged:
            merged.append(piece)
            continue
        if len(merged[-1]) + 1 + len(piece) <= chunk_size:
            merged[-1] = f"{merged[-1]}\n{piece}"
            continue
        seed = merged[-1][-overlap:] if overlap else ""
        if seed and len(seed) + 1 + len(piece) > chunk_size:
            seed = ""   # 重叠不能把片推过上限：宁可少给上下文，也不能破坏片长契约
        merged.append(f"{seed}\n{piece}" if seed else piece)
    return merged
