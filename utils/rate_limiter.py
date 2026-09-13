from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

from loguru import logger

from config import Settings


@dataclass
class TokenBucket:
    capacity: int
    refill_rate: float
    tokens: float = 0.0
    last_refill: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.tokens = float(self.capacity)

    def consume(self, amount: int = 1) -> bool:
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

        if self.tokens >= amount:
            self.tokens -= amount
            return True
        return False

    def time_until_available(self, amount: int = 1) -> float:
        if self.tokens >= amount:
            return 0.0
        needed = amount - self.tokens
        return needed / self.refill_rate


@dataclass
class SlidingWindowCounter:
    window_size: float = 60.0
    max_requests: int = 10
    timestamps: list[float] = field(default_factory=list)

    def _cleanup(self) -> None:
        now = time.time()
        cutoff = now - self.window_size
        self.timestamps = [t for t in self.timestamps if t > cutoff]

    def check(self) -> bool:
        self._cleanup()
        return len(self.timestamps) < self.max_requests

    def record(self) -> None:
        self.timestamps.append(time.time())

    def current_count(self) -> int:
        self._cleanup()
        return len(self.timestamps)

    def time_until_available(self) -> float:
        self._cleanup()
        if len(self.timestamps) < self.max_requests:
            return 0.0
        oldest = min(self.timestamps)
        return self.window_size - (time.time() - oldest)


@dataclass
class EndpointLimit:
    per_minute: int
    burst: int
    counter: SlidingWindowCounter = field(init=False)
    bucket: TokenBucket = field(init=False)

    def __post_init__(self) -> None:
        self.counter = SlidingWindowCounter(window_size=60.0, max_requests=self.per_minute)
        self.bucket = TokenBucket(capacity=self.burst, refill_rate=self.per_minute / 60.0)


class RateLimiter:
    def __init__(self, settings: Settings) -> None:
        self._per_minute = settings.rate_limit_per_minute
        self._burst = settings.rate_limit_burst
        self._limits: dict[tuple[int, str], EndpointLimit] = {}
        self._global_counts: dict[str, int] = defaultdict(int)

    def _get_limit(self, user_id: int, endpoint: str) -> EndpointLimit:
        key = (user_id, endpoint)
        if key not in self._limits:
            self._limits[key] = EndpointLimit(
                per_minute=self._per_minute,
                burst=self._burst,
            )
        return self._limits[key]

    def check(self, user_id: int, endpoint: str = "default") -> bool:
        limit = self._get_limit(user_id, endpoint)
        allowed = limit.counter.check() and limit.bucket.consume()
        if not allowed:
            logger.debug("Rate limit hit for user {} on {}", user_id, endpoint)
        return allowed

    def record(self, user_id: int, endpoint: str = "default") -> None:
        limit = self._get_limit(user_id, endpoint)
        limit.counter.record()

    def get_wait_time(self, user_id: int, endpoint: str = "default") -> float:
        limit = self._get_limit(user_id, endpoint)
        counter_wait = limit.counter.time_until_available()
        bucket_wait = limit.bucket.time_until_available(1)
        return max(counter_wait, bucket_wait)

    def get_remaining(self, user_id: int, endpoint: str = "default") -> int:
        limit = self._get_limit(user_id, endpoint)
        return max(0, limit.counter.max_requests - limit.counter.current_count())

    def get_usage_info(self, user_id: int, endpoint: str = "default") -> dict:
        limit = self._get_limit(user_id, endpoint)
        remaining = self.get_remaining(user_id, endpoint)
        wait_time = self.get_wait_time(user_id, endpoint)
        return {
            "remaining": remaining,
            "limit": limit.counter.max_requests,
            "wait_seconds": round(wait_time, 2),
            "is_limited": remaining == 0,
        }

    def cleanup(self, max_age: float = 600.0) -> None:
        now = time.time()
        stale_keys = []
        for key, limit in self._limits.items():
            if now - limit.counter.timestamps[-1] > max_age if limit.counter.timestamps else True:
                stale_keys.append(key)
        for key in stale_keys:
            del self._limits[key]

    def set_custom_limit(self, user_id: int, endpoint: str, per_minute: int, burst: int | None = None) -> None:
        key = (user_id, endpoint)
        self._limits[key] = EndpointLimit(
            per_minute=per_minute,
            burst=burst or per_minute,
        )

    def get_all_user_limits(self) -> dict[int, dict[str, dict]]:
        user_limits: dict[int, dict[str, dict]] = {}
        for (user_id, endpoint), limit in self._limits.items():
            if user_id not in user_limits:
                user_limits[user_id] = {}
            remaining = max(0, limit.counter.max_requests - limit.counter.current_count())
            user_limits[user_id][endpoint] = {
                "remaining": remaining,
                "limit": limit.counter.max_requests,
            }
        return user_limits
