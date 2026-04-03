"""
Unit tests for _ResponseCache — TTL expiry, eviction, key normalisation.

get() returns (response_text, confidence) or None.
Use _text() to extract just the response text for equality checks.
"""

import time
import pytest
from app.orchestrator import _ResponseCache


def _text(result) -> str:
    """Unwrap (response, confidence) → response, or propagate None."""
    if result is None:
        return None
    return result[0]


@pytest.fixture
def cache():
    return _ResponseCache(ttl_seconds=60, max_size=5)


class TestCacheBasics:
    def test_set_and_get(self, cache):
        cache.set("hello", "world")
        assert _text(cache.get("hello")) == "world"

    def test_get_missing_returns_none(self, cache):
        assert cache.get("not_here") is None

    def test_size_increments_on_set(self, cache):
        cache.set("a", "1")
        cache.set("b", "2")
        assert cache.size == 2

    def test_overwrite_same_key(self, cache):
        cache.set("key", "first")
        cache.set("key", "second")
        assert _text(cache.get("key")) == "second"
        assert cache.size == 1

    def test_confidence_stored_and_returned(self, cache):
        cache.set("msg", "resp", confidence=0.75)
        result = cache.get("msg")
        assert result is not None
        text, conf = result
        assert text == "resp"
        assert conf == 0.75

    def test_default_confidence_is_1(self, cache):
        cache.set("msg", "resp")
        _text, conf = cache.get("msg")
        assert conf == 1.0


class TestCacheNormalisation:
    def test_case_insensitive_key(self, cache):
        cache.set("Hello World", "response")
        assert _text(cache.get("hello world")) == "response"
        assert _text(cache.get("HELLO WORLD")) == "response"

    def test_leading_trailing_whitespace_stripped(self, cache):
        cache.set("  query  ", "response")
        assert _text(cache.get("query")) == "response"
        assert _text(cache.get("  query  ")) == "response"

    def test_different_messages_different_keys(self, cache):
        cache.set("blood sugar", "r1")
        cache.set("blood pressure", "r2")
        assert _text(cache.get("blood sugar")) == "r1"
        assert _text(cache.get("blood pressure")) == "r2"


class TestCacheTTL:
    def test_expired_entry_returns_none(self):
        short_cache = _ResponseCache(ttl_seconds=1, max_size=10)
        short_cache.set("query", "response")
        assert _text(short_cache.get("query")) == "response"
        time.sleep(1.1)
        assert short_cache.get("query") is None

    def test_unexpired_entry_survives(self):
        long_cache = _ResponseCache(ttl_seconds=3600, max_size=10)
        long_cache.set("query", "response")
        assert _text(long_cache.get("query")) == "response"


class TestCacheEviction:
    def test_max_size_evicts_oldest(self):
        cache = _ResponseCache(ttl_seconds=3600, max_size=3)
        cache.set("a", "1")
        cache.set("b", "2")
        cache.set("c", "3")
        assert cache.size == 3
        cache.set("d", "4")  # triggers eviction
        assert cache.size == 3
        # "d" must be present; "a" should have been evicted (FIFO)
        assert _text(cache.get("d")) == "4"
        assert cache.get("a") is None

    def test_expired_entries_evicted_before_oldest(self):
        cache = _ResponseCache(ttl_seconds=1, max_size=3)
        cache.set("old_a", "1")
        cache.set("old_b", "2")
        time.sleep(1.1)
        # Both old entries expired; adding a new one should evict expired first
        cache.set("new_c", "3")
        assert _text(cache.get("new_c")) == "3"
        assert cache.size <= 3
