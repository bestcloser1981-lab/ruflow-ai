import asyncio
import os
import sys
import types
from unittest.mock import Mock

os.environ.setdefault("GEMINI_API_KEY", "test-key")

# sentence-transformers imports scipy/sklearn at module import time, which is
# blocked in this environment. Stub the package so we can exercise the cache
# logic without loading the heavy embedding model stack.
if "sentence_transformers" not in sys.modules:
    sentence_transformers = types.ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = object
    sys.modules["sentence_transformers"] = sentence_transformers

from backend.services.memory_optimizer import MemoryOptimizer


async def _run_cache_expiry_checks() -> None:
    optimizer = MemoryOptimizer(
        memory_service=Mock(),
        embedding_service=Mock(),
        cache_ttl_seconds=0.10,
        cache_max_size=8,
    )

    await optimizer.cache_set("warm-key", {"value": "keep"})
    assert await optimizer.cache_get("warm-key") == {"value": "keep"}

    await asyncio.sleep(0.06)
    assert await optimizer.cache_get("warm-key") == {"value": "keep"}

    await asyncio.sleep(0.06)
    assert await optimizer.cache_get("warm-key") == {"value": "keep"}

    await asyncio.sleep(0.12)
    assert await optimizer.cache_get("warm-key") is None

    await optimizer.cache_set("fresh-key", {"value": "fresh"})
    await asyncio.sleep(0.12)
    assert await optimizer.cache_get("fresh-key") is None


def test_cache_ttl_refreshes_on_hot_access() -> None:
    asyncio.run(_run_cache_expiry_checks())
