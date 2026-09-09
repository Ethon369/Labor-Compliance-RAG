"""切分器离线单测：只喂 Block 流，验"边界是否合理、片长是否受控、元数据是否跟上"。"""
from __future__ import annotations

import pytest

from app.kb.parser import Block, ParsedDoc
from app.kb.splitter import TextChunk, hash_content, split_document


def _doc(*blocks: Block) -> ParsedDoc:
    return ParsedDoc(blocks=list(blocks))


def _text(*lines: str) -> ParsedDoc:
    # 标题判定归 parser（按来源格式：MD 的 `#`、DOCX 的 Heading 样式），splitter 只消费
    # is_heading、不重复识别语法。这里镜像 MD 规则，让构造出来的文档和真解析结果同形。
    return _doc(*(Block(text=t, is_heading=t.lstrip().startswith("#")) for t in lines))


# ---- 基本形态 ----

def test_empty_document_yields_nothing():
    assert split_document(_doc()) == []


def test_short_document_is_single_chunk():
    chunks = split_document(_text("第一条 为了公正及时解决劳动争议。"))
    assert len(chunks) == 1
    assert chunks[0] == TextChunk(seq=1, content="第一条 为了公正及时解决劳动争议。",
                                  heading="", page=None)


def test_seq_is_continuous_across_sections():
    chunks = split_document(_text("# 第一章", "甲" * 30, "# 第二章", "乙" * 30),
                            chunk_size=20, overlap=0)
    assert [c.seq for c in chunks] == list(range(1, len(chunks) + 1))


def test_heading_is_attached_and_markdown_marker_stripped():
    chunks = split_document(_text("# 第一章 总则", "甲" * 30, "## 第二节 范围", "乙" * 30),
                            chunk_size=40, overlap=0)
    assert [c.heading for c in chunks] == ["第一章 总则", "第二节 范围"]
    # heading 字段去掉了 `#`，但正文保留原样（引用里能看出它是标题行）
    assert chunks[0].content.startswith("# 第一章 总则")


def test_heading_propagates_to_every_chunk_of_section():
    chunks = split_document(_text("# 第一章", "甲" * 200), chunk_size=60, overlap=0)
    assert len(chunks) > 1
    assert {c.heading for c in chunks} == {"第一章"}


def test_page_comes_from_first_block_of_section():
    chunks = split_document(_doc(Block(text="甲" * 30, page=2), Block(text="乙" * 30, page=3)),
                            chunk_size=40, overlap=0)
    assert [c.page for c in chunks] == [2, 2]


def test_char_count_matches_content():
    chunks = split_document(_text("甲" * 10))
    assert chunks[0].char_count == 10


# ---- 片长与重叠契约 ----

def test_every_chunk_within_chunk_size():
    chunks = split_document(_text("甲" * 5000), chunk_size=500, overlap=80)
    assert len(chunks) > 1
    assert all(len(c.content) <= 500 for c in chunks)


def test_overlap_prepends_tail_of_previous_chunk():
    a, b, c = "甲" * 60, "乙" * 60, "丙" * 60
    chunks = split_document(_text(a, b, c), chunk_size=100, overlap=20)
    assert len(chunks) == 3
    assert chunks[1].content.startswith(a[-20:])
    assert chunks[2].content.startswith(b[-20:])


def test_overlap_never_pushes_chunk_over_limit():
    """重叠尾缀若会把片顶过上限，则宁可不给上下文——片长契约优先。"""
    chunks = split_document(_text("甲" * 100, "乙" * 100), chunk_size=100, overlap=50)
    assert all(len(c.content) <= 100 for c in chunks)


def test_oversized_overlap_is_clamped_without_hanging():
    chunks = split_document(_text("甲" * 400), chunk_size=100, overlap=10_000)
    assert chunks
    assert all(len(c.content) <= 100 for c in chunks)


def test_invalid_chunk_size_raises():
    with pytest.raises(ValueError):
        split_document(_text("甲"), chunk_size=0)


# ---- 切分边界 ----

def test_sentences_are_not_split_when_they_fit():
    sentences = [f"第{i}条 " + "甲" * 36 + "。" for i in range(1, 6)]
    chunks = split_document(_text("".join(sentences)), chunk_size=100, overlap=0)
    assert len(chunks) > 1
    # 每片都以句号收尾 = 没有一句话被劈成两半
    assert all(c.content.endswith("。") for c in chunks)
    assert chunks[0].content.startswith("第1条")


def test_punctuation_free_text_is_hard_cut_into_balanced_pieces():
    """完全没有边界的文本：硬切，且不留半截残片（均分而非切满）。"""
    chunks = split_document(_text("x" * 250), chunk_size=100, overlap=0)
    lengths = [len(c.content) for c in chunks]
    assert sum(lengths) == 250
    assert all(80 <= n <= 100 for n in lengths)


def test_long_paragraph_falls_back_to_sentence_then_hard_cut():
    """没有换行、只有句号的长段：先按句子切，而不是直接硬切。"""
    long_sentence = "甲" * 300
    text = long_sentence + "。" + "乙" * 40 + "。"
    chunks = split_document(_text(text), chunk_size=100, overlap=0)
    assert all(len(c.content) <= 100 for c in chunks)
    # 300 字的无标点长句被硬切，但后面的短句仍完整
    assert chunks[-1].content == "乙" * 40 + "。"


def test_blank_blocks_are_dropped():
    chunks = split_document(_text("第一条 内容", "   ", "\n"))
    assert [c.content for c in chunks] == ["第一条 内容"]


# ---- 内容指纹 ----

def test_hash_content_is_stable_and_content_sensitive():
    assert hash_content("第一条") == hash_content("第一条")
    assert hash_content("第一条") != hash_content("第二条")
