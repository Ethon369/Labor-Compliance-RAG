"""文档解析：文件 → 带页码与标题标记的段落流（Block）。

职责边界：本模块只负责"把文件里的文字和结构取出来"，不做切分、不碰数据库。
切分交给 splitter.py，编排交给 pipeline.py——这样解析器可以纯离线单测
（构造字节流即可，不需要真文件与真库）。
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

SUPPORTED_TYPES = ("pdf", "docx", "txt", "md", "text")

# 按扩展名推断 source_type；上传端点用它决定解析器，也用于入库的 CHECK 约束
_EXT_TO_TYPE = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".txt": "txt",
    ".md": "md",
    ".markdown": "md",
    ".text": "text",
}


class UnsupportedContentError(Exception):
    """文件能打开但取不到文字（典型是扫描版 PDF 没有文字层）。"""


class Block(BaseModel):
    """一段文本 + 它在原文件中的位置线索。

    page：PDF 页码（1 起），其它来源为 None；
    is_heading：是否是标题行——切分时用它给后续片段补"最近上级标题"。
    """
    page: int | None = None
    text: str
    is_heading: bool = False


class ParsedDoc(BaseModel):
    blocks: list[Block]
    page_count: int | None = None

    @property
    def char_count(self) -> int:
        return sum(len(b.text) for b in self.blocks)


def detect_source_type(filename: str) -> str | None:
    """按扩展名判断文档类型；不支持的类型返回 None（调用方按 400 处理）。"""
    return _EXT_TO_TYPE.get(Path(filename).suffix.lower())


def parse_file(path: str | Path, source_type: str) -> ParsedDoc:
    """按类型分派解析。取不到文字时抛 UnsupportedContentError（调用方标 failed）。"""
    p = Path(path)
    if source_type == "pdf":
        return _parse_pdf(p)
    if source_type == "docx":
        return _parse_docx(p)
    if source_type in ("txt", "md", "text"):
        return _parse_text(p, is_markdown=(source_type == "md"))
    raise UnsupportedContentError(f"不支持的文档类型：{source_type}")


def _read_text_any_encoding(p: Path) -> str:
    """按 utf-8 → gb18030 回退读取。中文老文档常见 GBK 编码，直接 utf-8 会炸。"""
    raw = p.read_bytes()
    for enc in ("utf-8", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _parse_pdf(p: Path) -> ParsedDoc:
    from pypdf import PdfReader

    reader = PdfReader(str(p))
    blocks: list[Block] = []
    for page_no, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        for line in text.splitlines():
            line = line.strip()
            if line:
                blocks.append(Block(page=page_no, text=line))
    if not blocks:
        raise UnsupportedContentError("PDF 无文字层（可能是扫描件），需要 OCR，当前版本不支持")
    return ParsedDoc(blocks=blocks, page_count=len(reader.pages))


def _parse_docx(p: Path) -> ParsedDoc:
    import docx

    document = docx.Document(str(p))
    blocks: list[Block] = []
    for para in document.paragraphs:
        text = (para.text or "").strip()
        if not text:
            continue
        style = (para.style.name or "") if para.style is not None else ""
        blocks.append(Block(text=text, is_heading=style.startswith("Heading") or style == "Title"))
    if not blocks:
        raise UnsupportedContentError("Word 文档里没有可提取的文字段落")
    return ParsedDoc(blocks=blocks)


def _parse_text(p: Path, is_markdown: bool) -> ParsedDoc:
    content = _read_text_any_encoding(p)
    blocks: list[Block] = []
    for line in content.splitlines():
        line = line.rstrip()
        if not line.strip():
            continue
        is_heading = is_markdown and line.lstrip().startswith("#")
        blocks.append(Block(text=line.strip(), is_heading=is_heading))
    if not blocks:
        raise UnsupportedContentError("文件内容为空")
    return ParsedDoc(blocks=blocks)
