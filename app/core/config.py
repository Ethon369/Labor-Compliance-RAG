"""运行配置：一切外部依赖（DB 连接、embedding 提供方）从环境变量 + .env 读取。

app/rag 的组件不直接读这里——由 build_retriever()/回填脚本这类"组装点"读一次，
再以构造函数参数注入下层。好处：检索逻辑与配置解耦，单测可传显式参数、不碰 env。
"""
from __future__ import annotations

from typing import Literal

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# .env 按项目根定位而不是相对运行目录：脚本在任意 cwd 下执行都能读到，避免静默回落到默认值
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_PROJECT_ROOT / ".env", extra="ignore")

    database_url: str = "postgresql://labor:labor@localhost:5432/labor"

    # embedding 提供方：offline=确定性哈希占位（离线可测/演示）；siliconflow=真 bge-m3
    embedding_mode: Literal["offline", "siliconflow"] = "offline"
    # 向量维度与"离线、真模型"两路共用：列就按这个维度建，换真 bge-m3(1024) 不用重建表
    embedding_dim: int = 1024
    # 硅基流动 bge-m3（OpenAI 兼容 /embeddings），仅 mode=siliconflow 时用
    siliconflow_api_key: str | None = None
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"
    siliconflow_model: str = "BAAI/bge-m3"
