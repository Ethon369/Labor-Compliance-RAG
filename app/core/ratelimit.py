"""内存令牌桶限流：按 (端点, 客户端 IP) 分桶，超限返回 429；登录另配失败小窗口。

为什么内存而非 Redis：单 worker 演示规模，内存桶零依赖、判定 O(1)；多 worker 才需要
跨进程共享计数（Redis）。设计上把"判定"独立成纯函数，将来换 Redis 实现只改这一层。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock

from fastapi import HTTPException, Request


@dataclass
class _Bucket:
    tokens: float
    updated: float          # 上次补充 token 的时刻（monotonic）


class TokenBucketLimiter:
    """令牌桶：容量 capacity，每秒匀速补充 refill_per_sec 个 token。

    用"漏掉的桶要能漏掉"的经典模型：突发可以打满 capacity，但持续速率被
    refill_per_sec 卡死。桶按 key 惰性创建，长时间不用的桶会被 prune 掉。
    """

    def __init__(self) -> None:
        self._buckets: dict[str, _Bucket] = {}
        self._lock = Lock()

    def allow(self, key: str, capacity: float, refill_per_sec: float,
              cost: float = 1.0) -> bool:
        now = time.monotonic()
        with self._lock:
            b = self._buckets.get(key)
            if b is None:
                b = _Bucket(tokens=capacity, updated=now)
                self._buckets[key] = b
            b.tokens = min(capacity, b.tokens + (now - b.updated) * refill_per_sec)
            b.updated = now
            if b.tokens >= cost:
                b.tokens -= cost
                return True
            return False

    def prune(self, idle_sec: float = 600.0) -> None:
        """清理超过 idle_sec 没碰过的桶，防内存随 IP 数无限增长。"""
        now = time.monotonic()
        with self._lock:
            stale = [k for k, b in self._buckets.items() if now - b.updated > idle_sec]
            for k in stale:
                del self._buckets[k]

    def reset_all(self) -> None:
        """清空全部桶（测试隔离用：让各测试从干净状态起步，避免互相踩限流）。"""
        with self._lock:
            self._buckets.clear()


# 进程级单例：所有请求共享同一批桶
_limiter = TokenBucketLimiter()


def client_ip(request: Request) -> str:
    """取客户端 IP。信任 X-Forwarded-For 只在有反向代理时成立，本地直连取 peer。"""
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limit(capacity: float, refill_per_sec: float):
    """依赖工厂：对当前端点按客户端 IP 限流。超限抛 429。"""
    def dep(request: Request) -> None:
        key = f"{request.url.path}:{client_ip(request)}"
        if not _limiter.allow(key, capacity, refill_per_sec):
            raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")
    return dep


class LoginGuard:
    """登录/注册的失败小窗口：同 IP 连续失败超过 max_failures 次 → 封禁 window_sec。

    为什么单独一层：登录是暴力破解的入口，普通令牌桶防的是"高频合法请求"，
    防爆破要的是"失败次数的滑动窗口"——语义不同，不能复用同一个桶。
    成功后 reset，避免把正常用户的偶发失败累积成误封。
    """

    def __init__(self, max_failures: int = 5, window_sec: float = 60.0) -> None:
        self._failures: dict[str, list[float]] = {}
        self._lock = Lock()
        self.max_failures = max_failures
        self.window_sec = window_sec

    def is_blocked(self, ip: str) -> bool:
        now = time.monotonic()
        with self._lock:
            times = [t for t in self._failures.get(ip, []) if now - t < self.window_sec]
            self._failures[ip] = times
            return len(times) >= self.max_failures

    def record_failure(self, ip: str) -> None:
        now = time.monotonic()
        with self._lock:
            times = [t for t in self._failures.get(ip, []) if now - t < self.window_sec]
            times.append(now)
            self._failures[ip] = times

    def reset(self, ip: str) -> None:
        with self._lock:
            self._failures.pop(ip, None)

    def reset_all(self) -> None:
        """清空全部 IP 的失败记录（测试隔离用）。"""
        with self._lock:
            self._failures.clear()


login_guard = LoginGuard()
