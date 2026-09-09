"""文本→向量的两个提供方，对外是同一接口（embed_one/embed），切换不动下游。

为什么存在"离线占位"这条真路径而不是 mock：本机没有 key 时，要让 pgvector /
融合 / 检索全链路仍然离线可跑、可复现、可测试。占位向量是"词元集合"的确定性稠密
投影——召回能力约等于词面匹配，恰好让两条检索路对同一种"相关"给出一致结果，
便于验证融合逻辑；真正的语义（同义改写、语序无关）要等 bge-m3 上线才有。
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from typing import Protocol

from app.rag.tokenizer import tokenize


class Embedder(Protocol):
    dim: int

    def embed_one(self, text: str) -> list[float]: ...


def _l2_normalize(v: list[float]) -> list[float]:
    norm = sum(x * x for x in v) ** 0.5
    if norm == 0.0:
        return v  # 空文本 => 全零向量，谁都不相似，比除零崩溃好
    return [x / norm for x in v]


class HashEmbedder:
    """确定性词元→稠密向量：词元用 md5 稳定投到固定维度、累加词频后 L2 归一。

    取 md5 前 8 位十六进制做下标而非内建 hash()——后者每次进程随机加盐，
    会破坏"同文本必得同向量"的可复现性。
    """

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for term in tokenize(text):
            idx = int(hashlib.md5(term.encode("utf-8")).hexdigest()[:8], 16) % self.dim
            vec[idx] += 1.0
        return _l2_normalize(vec)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]


class SiliconFlowEmbedder:
    """真 bge-m3：走硅基流动的 OpenAI 兼容 /embeddings。

    用 urllib 而非 requests——后者不在锁定清单，前者够用且零依赖；
    把网络异常原样抛给调用方，调用方决定是中止还是降级。
    """

    def __init__(
        self,
        api_key: str,
        model: str = "BAAI/bge-m3",
        base_url: str = "https://api.siliconflow.cn/v1",
        dim: int = 1024,
        batch_size: int = 32,
    ):
        if not api_key:
            raise ValueError("siliconflow 模式需要 SILICONFLOW_API_KEY")
        self._api_key = api_key
        self._model = model
        self._endpoint = f"{base_url.rstrip('/')}/embeddings"
        self.dim = dim
        self._batch_size = batch_size

    def _call(self, inputs: list[str]) -> list[list[float]]:
        body = json.dumps({"model": self._model, "input": inputs}).encode("utf-8")
        req = urllib.request.Request(
            self._endpoint,
            data=body,
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return [item["embedding"] for item in payload["data"]]

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            out.extend(self._call(texts[i : i + self._batch_size]))
        return out

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


def build_embedder(settings) -> Embedder:
    """按配置选 embedding 提供方——全项目唯一的"选谁当 embedder"的入口。

    backfill 脚本、检索组装（build_retriever）、上传流水线（pipeline）都走这里，
    避免三处各写一遍 if siliconflow、换提供方时漏改一处。
    缺 key 当场抛错而非静默降级：静默降级会让线上检索质量无声劣化且难以察觉。
    """
    if settings.embedding_mode == "siliconflow":
        return SiliconFlowEmbedder(
            api_key=settings.siliconflow_api_key or "",
            model=settings.siliconflow_model,
            base_url=settings.siliconflow_base_url,
            dim=settings.embedding_dim,
        )
    return HashEmbedder(settings.embedding_dim)
