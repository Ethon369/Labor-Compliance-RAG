"""限流单元测试（不碰 DB）。

验两件事：令牌桶按容量/速率正确放行与拒绝；登录失败小窗口在连续失败后封禁、
成功登录后清零。这是 V2.4 防刷与防爆破的两道闸。
"""
from __future__ import annotations

from app.core.ratelimit import LoginGuard, TokenBucketLimiter


def test_token_bucket_allow_within_capacity():
    lim = TokenBucketLimiter()
    # 容量 3：连续 3 次都放行
    for _ in range(3):
        assert lim.allow("k", capacity=3, refill_per_sec=0.0) is True
    # 第 4 次无 token 可放
    assert lim.allow("k", capacity=3, refill_per_sec=0.0) is False


def test_token_bucket_refills_over_time():
    import time
    lim = TokenBucketLimiter()
    assert lim.allow("k", capacity=1, refill_per_sec=10.0) is True   # 拿掉唯一 token
    assert lim.allow("k", capacity=1, refill_per_sec=10.0) is False  # 桶已空
    time.sleep(0.15)  # 0.15s × 10/s = 1.5 token，足够补满容量 1
    assert lim.allow("k", capacity=1, refill_per_sec=10.0) is True   # 补充后可再放行


def test_token_bucket_keys_isolated():
    lim = TokenBucketLimiter()
    lim.allow("a", capacity=1, refill_per_sec=0.0)
    # a 的桶已空，b 独立不受影响
    assert lim.allow("a", capacity=1, refill_per_sec=0.0) is False
    assert lim.allow("b", capacity=1, refill_per_sec=0.0) is True


def test_login_guard_blocks_after_failures():
    g = LoginGuard(max_failures=3, window_sec=60.0)
    ip = "1.2.3.4"
    for _ in range(3):
        assert g.is_blocked(ip) is False
        g.record_failure(ip)
    # 第 3 次失败后封禁
    assert g.is_blocked(ip) is True


def test_login_guard_reset_on_success():
    g = LoginGuard(max_failures=3, window_sec=60.0)
    ip = "5.6.7.8"
    g.record_failure(ip)
    g.record_failure(ip)
    g.reset(ip)  # 成功登录清零
    assert g.is_blocked(ip) is False
    g.record_failure(ip)
    g.record_failure(ip)
    assert g.is_blocked(ip) is False  # 2 次未达阈值
