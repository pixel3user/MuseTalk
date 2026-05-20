"""Reusable exponential backoff with full jitter.

This module provides a small, deterministic, testable helper for backoff
schedules. It is intentionally pure (no I/O) so unit tests can verify the
schedule without sleeping.

Why this exists
---------------
Both PersonaPlexMirrorClient and PersonaPlexChatBridge previously slept a
fixed `reconnect_delay_seconds` between reconnect attempts. When PersonaPlex
is down for an extended period, that produces a thundering-herd reconnect
storm: hundreds of attempts per minute.

The right policy is exponential backoff with full jitter:
    delay = random_uniform(0, min(cap, base * 2 ** attempt))

This is the AWS/Google-recommended algorithm. It bounds the worst-case load
on the upstream service while keeping individual reconnect latency low when
the outage is short.

Reference: https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class BackoffPolicy:
    """Exponential backoff with full jitter.

    Attributes:
        base_seconds: Initial delay (also the minimum on attempt 0).
        cap_seconds: Maximum delay regardless of attempt count.
        multiplier: Growth factor per attempt (default 2.0 = doubling).
    """

    base_seconds: float = 0.5
    cap_seconds: float = 30.0
    multiplier: float = 2.0

    def __post_init__(self) -> None:
        if self.base_seconds < 0:
            raise ValueError("base_seconds must be >= 0")
        if self.cap_seconds < self.base_seconds:
            raise ValueError("cap_seconds must be >= base_seconds")
        if self.multiplier < 1.0:
            raise ValueError("multiplier must be >= 1.0")

    def deterministic_max(self, attempt: int) -> float:
        """Maximum possible delay for a given attempt number, before jitter.

        Useful for tests and assertions; not used at runtime.
        """
        if attempt < 0:
            attempt = 0
        # Guard against overflow for very large attempt numbers.
        # 2^60 already exceeds any realistic cap; clamp earlier.
        if attempt > 60:
            return self.cap_seconds
        return min(self.cap_seconds, self.base_seconds * (self.multiplier ** attempt))


def compute_backoff(
    attempt: int,
    policy: BackoffPolicy,
    *,
    rng: "random.Random | None" = None,
) -> float:
    """Return the delay (seconds) to wait before the given attempt.

    Implements full jitter: returns a uniform random value in [0, max_for_attempt].
    Pass a seeded `random.Random` instance from tests for determinism.

    Args:
        attempt: 0-based attempt counter. The first reconnect uses attempt=0.
        policy: Backoff parameters.
        rng: Optional random number generator (for tests).

    Returns:
        Non-negative float seconds to sleep.
    """
    upper = policy.deterministic_max(attempt)
    if upper <= 0:
        return 0.0
    r = rng if rng is not None else random
    return r.uniform(0.0, upper)
