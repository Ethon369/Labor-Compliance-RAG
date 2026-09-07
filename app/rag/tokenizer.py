"""中文切词：CJK 连续字符的滑窗二元组 + ASCII 词。零第三方依赖。

为什么是"二元组"而不是单字/整词/分词库：
- 不引 jieba（锁定清单外）；法律条文用词规范但未分词，"词"边界本身有歧义。
- 无词表也能把"加班费"切成 加班/班费，召回靠相邻字共现；BM25 的 idf 会自然
  压掉"支付/劳动"这类处处共现的高频噪音，占位向量同理。
- 代价：词粒度过粗、无关共现也成 token——bge-m3 语义向量上线后，这套切词
  只服务 BM25 这一路，语义路走 embedding，两路互补的边界正好在这里。
"""
from __future__ import annotations

import re

_HAN_RUN = re.compile(r"[一-鿿]+")  # 一段连续汉字
_ASCII_RUN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    """把文本切成检索词元列表。保留重复以携带词频（BM25 需要 tf）。"""
    tokens: list[str] = []
    for run in _HAN_RUN.findall(text):
        if len(run) == 1:
            tokens.append(run)  # 单字成词的残段（如"法"），避免整段词面丢失
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    tokens.extend(_ASCII_RUN.findall(text))
    return tokens
