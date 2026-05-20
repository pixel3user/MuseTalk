"""Shared constants for the WebRTC signaling app."""

SESSION_TOKEN_HEADER = "x-session-token"

# Idempotency support for /offer retries.
# When a browser fetch('/offer') is retried after a network glitch, we want
# the second request to return the same answer SDP (and not create a new
# session that immediately replaces the first one in single-session mode).
IDEMPOTENCY_KEY_HEADER = "x-idempotency-key"
IDEMPOTENCY_TTL_SECONDS = 30.0
IDEMPOTENCY_CACHE_MAX_ENTRIES = 256
