"""LRU cache of voice-clone prompts (Phase 2b, Tier-A).

OmniVoice's :py:meth:`omnivoice_core.models.OmniVoice.create_voice_clone_prompt`
performs three expensive steps every time it is called:

1. ``load_audio`` — disk read + decode + resample to model rate (24 kHz).
2. ``remove_silence`` / ``trim_long_audio`` — host-side numpy work.
3. ``audio_tokenizer.encode`` — GPU forward pass producing the (C, T)
   reference token tensor.

For a XGEN-style workload (a small, fixed roster of voice profiles
called over and over), steps 1–3 are pure functions of
``(ref_audio_path, ref_text, preprocess_prompt)``. Caching the resulting
:class:`omnivoice_core.models.omnivoice.VoiceClonePrompt` removes the
per-request reference-encoding latency from every call after the first.

Output equivalence
------------------

This optimisation is Tier-A: the cached prompt is the *exact* same
object the upstream method would have returned for the same inputs, so
``model.generate(voice_clone_prompt=cached)`` produces bit-identical
PCM (under the same RNG seed) to ``model.generate(...)`` with a
freshly-built prompt. The output-equivalence regression gate
(``compare_audio check --atol 1e-4``) verifies this on every PR.

Cache invalidation
------------------

The cache key includes the file's ``st_mtime_ns`` and ``st_size`` so any
on-disk change to a reference audio invalidates its entry on the next
lookup, without operators having to flush manually.

Concurrency
-----------

The cache is guarded by an ``asyncio.Lock`` per (cache instance) — but
the model.create_voice_clone_prompt call itself is dispatched to the
shared executor under the engine's existing concurrency semaphore, so
two concurrent misses on the *same* key collapse into one expensive
build (the second waiter sees a populated entry on wakeup).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Tuple

logger = logging.getLogger("server.ref_cache")


_CacheKey = Tuple[str, str, bool, int, int]  # path, ref_text, preprocess, mtime_ns, size


@dataclass
class _Entry:
    prompt: Any                # VoiceClonePrompt (kept opaque to avoid import)
    last_used_monotonic: float


class RefCache:
    """Bounded LRU of :class:`VoiceClonePrompt` keyed by ref-audio identity.

    Not thread-safe; intended to be driven from a single asyncio loop.
    """

    def __init__(self, *, max_size: int) -> None:
        if max_size < 0:
            raise ValueError("max_size must be ≥ 0")
        self._max_size = int(max_size)
        self._store: "OrderedDict[_CacheKey, _Entry]" = OrderedDict()
        self._lock = asyncio.Lock()
        # Per-key build locks coalesce concurrent misses on the same ref.
        self._build_locks: dict[_CacheKey, asyncio.Lock] = {}
        # Stats are exposed via /diag/cache for the perf gate.
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def size(self) -> int:
        return len(self._store)

    def stats(self) -> dict:
        total = self.hits + self.misses
        hit_ratio = (self.hits / total) if total else 0.0
        return {
            "size": len(self._store),
            "max_size": self._max_size,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_ratio": round(hit_ratio, 4),
        }

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()
            self._build_locks.clear()

    @staticmethod
    def _key_for(ref_audio_path: str, ref_text: Optional[str],
                 preprocess_prompt: bool) -> _CacheKey:
        try:
            st = os.stat(ref_audio_path)
            mtime_ns = int(st.st_mtime_ns)
            size = int(st.st_size)
        except OSError:
            # File missing: still build a key so the build attempt raises
            # a meaningful error from the upstream loader, not from us.
            mtime_ns = -1
            size = -1
        return (str(ref_audio_path), str(ref_text or ""),
                bool(preprocess_prompt), mtime_ns, size)

    async def get_or_build(
        self,
        *,
        ref_audio_path: str,
        ref_text: Optional[str],
        preprocess_prompt: bool,
        builder: Callable[[], Awaitable[Any]],
    ) -> Any:
        """Return a cached prompt or build + cache one.

        ``builder`` is an async callable that produces a fresh
        :class:`VoiceClonePrompt`. It is invoked at most once per
        cache-key while a miss is in flight; concurrent waiters on the
        same key share the result.
        """
        if self._max_size == 0:
            # Cache disabled — straight passthrough.
            self.misses += 1
            return await builder()

        key = self._key_for(ref_audio_path, ref_text, preprocess_prompt)

        # Fast path: hit.
        async with self._lock:
            entry = self._store.get(key)
            if entry is not None:
                entry.last_used_monotonic = time.monotonic()
                self._store.move_to_end(key)
                self.hits += 1
                logger.debug("ref-cache hit key=%s", key[0])
                return entry.prompt

            # Miss: take per-key build lock to coalesce concurrent misses.
            build_lock = self._build_locks.get(key)
            if build_lock is None:
                build_lock = asyncio.Lock()
                self._build_locks[key] = build_lock

        async with build_lock:
            # Double-check under the build lock in case someone filled it
            # while we were waiting.
            async with self._lock:
                entry = self._store.get(key)
                if entry is not None:
                    entry.last_used_monotonic = time.monotonic()
                    self._store.move_to_end(key)
                    self.hits += 1
                    return entry.prompt
                self.misses += 1

            logger.info("ref-cache miss; building prompt for %s", ref_audio_path)
            prompt = await builder()

            async with self._lock:
                self._store[key] = _Entry(
                    prompt=prompt, last_used_monotonic=time.monotonic(),
                )
                self._store.move_to_end(key)
                while len(self._store) > self._max_size:
                    evicted_key, _ = self._store.popitem(last=False)
                    self.evictions += 1
                    logger.info(
                        "ref-cache eviction (size > max=%d): %s",
                        self._max_size, evicted_key[0],
                    )
                self._build_locks.pop(key, None)
                return prompt
