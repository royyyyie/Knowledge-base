from __future__ import annotations

import asyncio
import inspect
import math
import os
import threading
import time
import unicodedata
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from numbers import Real
from typing import Any, Protocol

from backend.providers.core import (
    CancellationProbe,
    ProviderCallContext,
    ProviderExecutor,
    ProviderOperation,
    ProviderPolicy,
)
from backend.providers.loop_bridge import ProviderLoopBridge


Vector = list[float]
SparseVector = dict[int, float]
ModelFactory = Callable[[], Any]


class EmbeddingMode(StrEnum):
    QUERY = "query"
    DOCUMENT = "document"


@dataclass(frozen=True)
class EmbeddingScope:
    """Cache isolation fields supplied by the retrieval or indexing caller."""

    namespace: str = "default"
    tenant_id: str = "default"
    index_id: str = "default"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "namespace", _normalize_scope_part(self.namespace, "default")
        )
        object.__setattr__(
            self, "tenant_id", _normalize_scope_part(self.tenant_id, "default")
        )
        object.__setattr__(
            self, "index_id", _normalize_scope_part(self.index_id, "default")
        )


@dataclass(frozen=True)
class EmbeddingReadiness:
    ready: bool
    closed: bool
    model_loaded: bool
    dimension: int | None
    queue_depth: int
    inflight: int


@dataclass(frozen=True)
class EmbeddingRuntimeStats:
    cache_hits: int
    cache_misses: int
    inflight_joins: int
    batches: int
    encoded_texts: int


@dataclass(frozen=True)
class BgeM3Features:
    dense: Vector
    sparse: SparseVector


class EmbeddingProvider(Protocol):
    async def embed_query(
        self,
        text: str,
        *,
        scope: EmbeddingScope | None = None,
        deadline: float | None = None,
        cancellation: CancellationProbe | None = None,
    ) -> Vector: ...

    async def embed_documents(
        self,
        texts: Sequence[str],
        *,
        scope: EmbeddingScope | None = None,
        deadline: float | None = None,
        cancellation: CancellationProbe | None = None,
    ) -> list[Vector]: ...


@dataclass(frozen=True)
class _CacheKey:
    model_name: str
    model_revision: str
    namespace: str
    tenant_id: str
    index_id: str
    mode: EmbeddingMode
    text: str


@dataclass(frozen=True)
class _WorkItem:
    key: _CacheKey
    text: str
    future: asyncio.Future[tuple[float, ...]]


class EmbeddingRuntime:
    """Async-native local embedding Adapter with bounded execution.

    One runtime belongs to one event loop. Model construction and encoding run
    only on its dedicated executor. Query and document work have separate
    micro-batch queues but share one encoder concurrency gate.
    """

    def __init__(
        self,
        *,
        model_name: str | None = None,
        model_revision: str | None = None,
        device: str | None = None,
        backend: str = "sentence_transformers",
        query_max_tokens: int = 128,
        document_max_tokens: int = 512,
        provider_name: str = "embedding-model",
        model_factory: ModelFactory | None = None,
        provider_executor: ProviderExecutor | None = None,
        policy: ProviderPolicy | None = None,
        encoder_concurrency: int = 2,
        executor_workers: int | None = None,
        max_batch_size: int = 32,
        max_queue_size: int = 256,
        max_request_size: int = 4096,
        cache_size: int = 2048,
        microbatch_window_seconds: float = 0.02,
        max_text_characters: int = 32768,
        expected_dimension: int | None = None,
        default_timeout_seconds: float | None = None,
        cancellation_poll_seconds: float = 0.025,
        clock: Callable[[], float] = time.monotonic,
        async_sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        workers = executor_workers or encoder_concurrency
        _validate_positive_int("encoder_concurrency", encoder_concurrency)
        _validate_positive_int("executor_workers", workers)
        _validate_positive_int("max_batch_size", max_batch_size)
        _validate_positive_int("max_queue_size", max_queue_size)
        _validate_positive_int("max_request_size", max_request_size)
        _validate_positive_int("max_text_characters", max_text_characters)
        if backend not in {"sentence_transformers", "bge_m3"}:
            raise ValueError("unsupported embedding backend")
        if backend == "bge_m3" and encoder_concurrency != 1:
            raise ValueError("the BGE-M3 model requires one encoder at a time")
        if backend == "bge_m3" and workers != 1:
            raise ValueError("the BGE-M3 model requires one executor worker")
        for name, value in (
            ("query_max_tokens", query_max_tokens),
            ("document_max_tokens", document_max_tokens),
        ):
            _validate_positive_int(name, value)
            if value > 8192:
                raise ValueError(f"{name} must not exceed 8192")
        if cache_size < 0:
            raise ValueError("cache_size cannot be negative")
        if microbatch_window_seconds < 0 or not math.isfinite(
            microbatch_window_seconds
        ):
            raise ValueError(
                "microbatch_window_seconds must be finite and non-negative"
            )
        if cancellation_poll_seconds <= 0 or not math.isfinite(
            cancellation_poll_seconds
        ):
            raise ValueError("cancellation_poll_seconds must be positive and finite")
        if expected_dimension is not None:
            _validate_positive_int("expected_dimension", expected_dimension)
        if default_timeout_seconds is not None and (
            default_timeout_seconds <= 0 or not math.isfinite(default_timeout_seconds)
        ):
            raise ValueError("default_timeout_seconds must be positive and finite")

        self.model_name = (
            model_name or os.getenv("EMBEDDING_MODEL") or "BAAI/bge-m3"
        ).strip()
        self.model_revision = (
            model_revision
            if model_revision is not None
            else (os.getenv("EMBEDDING_MODEL_REVISION") or "").strip()
        )
        self.device = (device or os.getenv("EMBEDDING_DEVICE") or "cpu").strip()
        self.backend = backend
        self._query_max_tokens = query_max_tokens
        self._document_max_tokens = document_max_tokens
        self.provider_name = provider_name
        if not self.model_name:
            raise ValueError("model_name cannot be empty")
        if not self.device:
            raise ValueError("device cannot be empty")

        self._model_factory = model_factory or self._create_default_model
        self._provider_executor = provider_executor or ProviderExecutor(
            clock=clock,
            async_sleeper=async_sleeper,
        )
        self._policy = policy or ProviderPolicy(max_attempts=2)
        self._encoder_concurrency = encoder_concurrency
        self._max_batch_size = max_batch_size
        self._max_queue_size = max_queue_size
        self._max_request_size = max_request_size
        self._cache_size = cache_size
        self._microbatch_window_seconds = float(microbatch_window_seconds)
        self._max_text_characters = max_text_characters
        self._cancellation_poll_seconds = float(cancellation_poll_seconds)
        self._dimension = expected_dimension
        self._default_timeout_seconds = default_timeout_seconds
        self._clock = clock

        self._executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="embedding-encoder",
        )
        self._executor_shutdown = False
        self._owner_loop: asyncio.AbstractEventLoop | None = None
        self._model: Any | None = None
        self._model_lock: asyncio.Lock | None = None
        self._cache_lock: asyncio.Lock | None = None
        self._encoder_gate: asyncio.Semaphore | None = None
        self._capacity: asyncio.BoundedSemaphore | None = None
        self._queues: dict[EmbeddingMode, asyncio.Queue[_WorkItem]] = {}
        self._workers: list[asyncio.Task[None]] = []
        self._cache: OrderedDict[_CacheKey, tuple[float, ...]] = OrderedDict()
        self._inflight: dict[_CacheKey, asyncio.Future[tuple[float, ...]]] = {}
        self._close_task: asyncio.Task[None] | None = None
        self._closing = False
        self._closed = False
        self._ready = False
        self._cache_hits = 0
        self._cache_misses = 0
        self._inflight_joins = 0
        self._batches = 0
        self._encoded_texts = 0

    async def embed_query(
        self,
        text: str,
        *,
        scope: EmbeddingScope | None = None,
        namespace: str | None = None,
        tenant_id: str | None = None,
        index_id: str | None = None,
        deadline: float | None = None,
        cancellation: CancellationProbe | None = None,
        policy: ProviderPolicy | None = None,
    ) -> Vector:
        self._raise_if_closed()
        normalized = self._normalize_input(text)
        resolved_scope = _resolve_scope(
            scope,
            namespace=namespace,
            tenant_id=tenant_id,
            index_id=index_id,
        )
        resolved_deadline = self._resolve_deadline(deadline)
        context = ProviderCallContext(
            provider=self.provider_name,
            operation=ProviderOperation.EMBEDDING,
            deadline=resolved_deadline,
            cancellation=cancellation,
        )
        vectors = await self._provider_executor.acall(
            lambda: self._embed_once(
                [normalized],
                mode=EmbeddingMode.QUERY,
                scope=resolved_scope,
                cancellation=cancellation,
            ),
            context=context,
            policy=policy or self._policy,
        )
        return vectors[0]

    async def embed_documents(
        self,
        texts: Sequence[str],
        *,
        scope: EmbeddingScope | None = None,
        namespace: str | None = None,
        tenant_id: str | None = None,
        index_id: str | None = None,
        deadline: float | None = None,
        cancellation: CancellationProbe | None = None,
        policy: ProviderPolicy | None = None,
    ) -> list[Vector]:
        self._raise_if_closed()
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings")
        if len(texts) > self._max_request_size:
            raise ValueError(
                f"embedding request exceeds max_request_size={self._max_request_size}"
            )
        normalized = [self._normalize_input(text) for text in texts]
        if not normalized:
            return []
        resolved_scope = _resolve_scope(
            scope,
            namespace=namespace,
            tenant_id=tenant_id,
            index_id=index_id,
        )
        resolved_deadline = self._resolve_deadline(deadline)
        context = ProviderCallContext(
            provider=self.provider_name,
            operation=ProviderOperation.EMBEDDING,
            deadline=resolved_deadline,
            cancellation=cancellation,
        )
        return await self._provider_executor.acall(
            lambda: self._embed_once(
                normalized,
                mode=EmbeddingMode.DOCUMENT,
                scope=resolved_scope,
                cancellation=cancellation,
            ),
            context=context,
            policy=policy or self._policy,
        )

    async def embed_m3_features(
        self,
        texts: Sequence[str],
        *,
        mode: EmbeddingMode,
        deadline: float | None = None,
        cancellation: CancellationProbe | None = None,
    ) -> list[BgeM3Features]:
        """Return both retrieval heads from one BGE-M3 inference pass."""
        if self.backend != "bge_m3":
            raise RuntimeError("BGE-M3 features require the bge_m3 backend")
        self._raise_if_closed()
        if isinstance(texts, (str, bytes)) or len(texts) > self._max_request_size:
            raise ValueError("invalid BGE-M3 feature request size")
        normalized = [self._normalize_input(text) for text in texts]
        if not normalized:
            return []
        context = ProviderCallContext(
            provider=self.provider_name,
            operation=ProviderOperation.EMBEDDING,
            deadline=self._resolve_deadline(deadline),
            cancellation=cancellation,
        )
        return await self._provider_executor.acall(
            lambda: self._encode_m3_features_once(normalized, mode, cancellation),
            context=context,
            policy=self._policy,
        )

    async def score_m3_colbert(
        self,
        query: str,
        documents: Sequence[str],
        *,
        cancellation: CancellationProbe | None = None,
    ) -> list[float]:
        """Raw bounded inference; the reranker Adapter owns the Provider deadline."""
        if self.backend != "bge_m3":
            raise RuntimeError("ColBERT requires the bge_m3 backend")
        self._raise_if_closed()
        if isinstance(documents, (str, bytes)) or len(documents) > 500:
            raise ValueError("invalid ColBERT candidate count")
        normalized_query = self._normalize_input(query)
        normalized_documents = [self._normalize_input(text) for text in documents]
        if not normalized_documents:
            return []
        await self._ensure_started()
        self._raise_if_cancelled(cancellation)
        gate = self._require_encoder_gate()
        async with gate:
            model = await self._get_model()
            loop = asyncio.get_running_loop()
            stop = threading.Event()
            try:
                raw = await loop.run_in_executor(
                    self._executor,
                    self._score_m3_colbert_sync,
                    model,
                    normalized_query,
                    normalized_documents,
                    stop,
                )
            except asyncio.CancelledError:
                stop.set()
                raise
        self._raise_if_cancelled(cancellation)
        if len(raw) != len(documents) or any(
            isinstance(score, bool)
            or not isinstance(score, (float, int))
            or not math.isfinite(float(score))
            for score in raw
        ):
            raise ValueError("BGE-M3 returned invalid ColBERT scores")
        return [float(score) for score in raw]

    async def warmup(
        self,
        text: str = "embedding warmup",
        *,
        deadline: float | None = None,
        cancellation: CancellationProbe | None = None,
    ) -> EmbeddingReadiness:
        await self.embed_query(
            text,
            scope=EmbeddingScope(namespace="runtime", index_id="warmup"),
            deadline=deadline,
            cancellation=cancellation,
        )
        return self.readiness()

    def readiness(self) -> EmbeddingReadiness:
        return EmbeddingReadiness(
            ready=self._ready and not self._closed,
            closed=self._closed,
            model_loaded=self._model is not None,
            dimension=self._dimension,
            queue_depth=sum(queue.qsize() for queue in self._queues.values()),
            inflight=len(self._inflight),
        )

    def stats(self) -> EmbeddingRuntimeStats:
        return EmbeddingRuntimeStats(
            cache_hits=self._cache_hits,
            cache_misses=self._cache_misses,
            inflight_joins=self._inflight_joins,
            batches=self._batches,
            encoded_texts=self._encoded_texts,
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._assert_owner_loop()
        if self._close_task is None:
            self._close_task = asyncio.create_task(
                self._close_impl(),
                name="embedding-runtime-close",
            )
        close_task = self._close_task
        try:
            await asyncio.shield(close_task)
        except asyncio.CancelledError:
            await asyncio.shield(close_task)
            raise
        finally:
            if (
                close_task.done()
                and not self._closed
                and self._close_task is close_task
            ):
                self._close_task = None

    async def _close_impl(self) -> None:
        self._closing = True
        self._ready = False
        try:
            workers = tuple(self._workers)
            for worker in workers:
                worker.cancel()
            if workers:
                await asyncio.gather(*workers, return_exceptions=True)
            self._workers.clear()

            for queue in self._queues.values():
                while True:
                    try:
                        item = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    await self._cancel_item(item)
                    queue.task_done()
                    if self._capacity is not None:
                        self._capacity.release()

            if self._cache_lock is not None:
                async with self._cache_lock:
                    remaining = tuple(self._inflight.values())
                    self._inflight.clear()
                    self._cache.clear()
            else:
                remaining = tuple(self._inflight.values())
                self._inflight.clear()
                self._cache.clear()
            for future in remaining:
                if not future.done():
                    future.cancel()

            if not self._executor_shutdown:
                self._executor_shutdown = True
                await asyncio.to_thread(
                    self._executor.shutdown,
                    wait=True,
                    cancel_futures=True,
                )
            self._model = None
        except BaseException:
            self._closing = False
            raise
        self._closed = True
        self._closing = False

    async def _embed_once(
        self,
        texts: Sequence[str],
        *,
        mode: EmbeddingMode,
        scope: EmbeddingScope,
        cancellation: CancellationProbe | None,
    ) -> list[Vector]:
        await self._ensure_started()
        self._raise_if_cancelled(cancellation)
        tasks = [
            asyncio.create_task(
                self._resolve_vector(
                    text,
                    mode=mode,
                    scope=scope,
                    cancellation=cancellation,
                )
            )
            for text in texts
        ]
        try:
            vectors = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return [list(vector) for vector in vectors]

    async def _resolve_vector(
        self,
        text: str,
        *,
        mode: EmbeddingMode,
        scope: EmbeddingScope,
        cancellation: CancellationProbe | None,
    ) -> tuple[float, ...]:
        key = self._cache_key(text, mode=mode, scope=scope)
        cache_lock = self._require_cache_lock()
        async with cache_lock:
            cached = self._cache_get(key)
            if cached is not None:
                self._cache_hits += 1
            shared = self._inflight.get(key)
            if cached is None and shared is not None:
                self._inflight_joins += 1
        if cached is not None:
            return cached
        if shared is not None:
            return await self._await_shared(shared, cancellation)

        self._raise_if_cancelled(cancellation)
        capacity = self._require_capacity()
        await capacity.acquire()
        queued = False
        cached = None
        shared = None
        try:
            self._raise_if_cancelled(cancellation)
            async with cache_lock:
                self._raise_if_closed()
                cached = self._cache_get(key)
                if cached is not None:
                    self._cache_hits += 1
                shared = self._inflight.get(key)
                if cached is None and shared is not None:
                    self._inflight_joins += 1
                if cached is None and shared is None:
                    shared = asyncio.get_running_loop().create_future()
                    shared.add_done_callback(_consume_future_exception)
                    self._inflight[key] = shared
                    if mode is EmbeddingMode.QUERY:
                        self._cache_misses += 1
                    self._queues[mode].put_nowait(
                        _WorkItem(key=key, text=text, future=shared)
                    )
                    queued = True
        finally:
            if not queued:
                capacity.release()

        if cached is not None:
            return cached
        if shared is None:
            raise AssertionError("embedding work was neither cached nor queued")
        return await self._await_shared(shared, cancellation)

    async def _await_shared(
        self,
        future: asyncio.Future[tuple[float, ...]],
        cancellation: CancellationProbe | None,
    ) -> tuple[float, ...]:
        shielded = asyncio.shield(future)
        try:
            if cancellation is None:
                return await shielded
            while True:
                self._raise_if_cancelled(cancellation)
                done, _ = await asyncio.wait(
                    {shielded}, timeout=self._cancellation_poll_seconds
                )
                if done:
                    return shielded.result()
        except BaseException:
            shielded.cancel()
            raise

    async def _ensure_started(self) -> None:
        self._raise_if_closed()
        loop = asyncio.get_running_loop()
        if self._owner_loop is not None:
            if self._owner_loop is not loop:
                raise RuntimeError("embedding runtime belongs to another event loop")
            return

        self._owner_loop = loop
        self._model_lock = asyncio.Lock()
        self._cache_lock = asyncio.Lock()
        self._encoder_gate = asyncio.Semaphore(self._encoder_concurrency)
        self._capacity = asyncio.BoundedSemaphore(self._max_queue_size)
        self._queues = {
            EmbeddingMode.QUERY: asyncio.Queue(),
            EmbeddingMode.DOCUMENT: asyncio.Queue(),
        }
        self._workers = [
            asyncio.create_task(
                self._batch_worker(mode),
                name=f"embedding-{mode.value}-batcher",
            )
            for mode in EmbeddingMode
        ]

    async def _batch_worker(self, mode: EmbeddingMode) -> None:
        queue = self._queues[mode]
        while True:
            first = await queue.get()
            batch = [first]
            try:
                await self._collect_microbatch(queue, batch)
                await self._process_batch(mode, batch)
            except asyncio.CancelledError:
                for item in batch:
                    await self._cancel_item(item)
                raise
            finally:
                for _ in batch:
                    queue.task_done()
                    self._require_capacity().release()

    async def _collect_microbatch(
        self,
        queue: asyncio.Queue[_WorkItem],
        batch: list[_WorkItem],
    ) -> None:
        loop = asyncio.get_running_loop()
        expires_at = loop.time() + self._microbatch_window_seconds
        while len(batch) < self._max_batch_size:
            try:
                batch.append(queue.get_nowait())
                continue
            except asyncio.QueueEmpty:
                pass

            remaining = expires_at - loop.time()
            if remaining <= 0:
                return
            try:
                batch.append(await asyncio.wait_for(queue.get(), timeout=remaining))
            except TimeoutError:
                return

    async def _process_batch(
        self,
        mode: EmbeddingMode,
        batch: Sequence[_WorkItem],
    ) -> None:
        try:
            gate = self._require_encoder_gate()
            async with gate:
                model = await self._get_model()
                self._batches += 1
                loop = asyncio.get_running_loop()
                raw = await loop.run_in_executor(
                    self._executor,
                    self._encode_sync,
                    model,
                    mode,
                    [item.text for item in batch],
                )
            vectors = self._validate_vectors(raw, expected_count=len(batch))
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            await self._fail_batch(batch, exc)
            return

        cache_lock = self._require_cache_lock()
        async with cache_lock:
            for item, vector in zip(batch, vectors, strict=True):
                if self._cache_size > 0 and mode is EmbeddingMode.QUERY:
                    self._cache[item.key] = vector
                    self._cache.move_to_end(item.key)
                    while len(self._cache) > self._cache_size:
                        self._cache.popitem(last=False)
                if self._inflight.get(item.key) is item.future:
                    self._inflight.pop(item.key, None)
            self._encoded_texts += len(vectors)
            self._ready = True

        for item, vector in zip(batch, vectors, strict=True):
            if not item.future.done():
                item.future.set_result(vector)

    async def _fail_batch(
        self,
        batch: Sequence[_WorkItem],
        exc: BaseException,
    ) -> None:
        cache_lock = self._require_cache_lock()
        async with cache_lock:
            for item in batch:
                if self._inflight.get(item.key) is item.future:
                    self._inflight.pop(item.key, None)
        for item in batch:
            if not item.future.done():
                item.future.set_exception(exc)

    async def _cancel_item(self, item: _WorkItem) -> None:
        if self._cache_lock is not None:
            async with self._cache_lock:
                if self._inflight.get(item.key) is item.future:
                    self._inflight.pop(item.key, None)
        if not item.future.done():
            item.future.cancel()

    async def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        model_lock = self._require_model_lock()
        async with model_lock:
            if self._model is None:
                loop = asyncio.get_running_loop()
                model = await loop.run_in_executor(
                    self._executor,
                    self._model_factory,
                )
                if model is None:
                    raise RuntimeError("embedding model factory returned no model")
                self._model = model
        return self._model

    async def _encode_m3_features_once(
        self,
        texts: Sequence[str],
        mode: EmbeddingMode,
        cancellation: CancellationProbe | None,
    ) -> list[BgeM3Features]:
        await self._ensure_started()
        self._raise_if_cancelled(cancellation)
        gate = self._require_encoder_gate()
        async with gate:
            model = await self._get_model()
            loop = asyncio.get_running_loop()
            raw = await loop.run_in_executor(
                self._executor, self._encode_m3_sync, model, mode, texts, True
            )
        self._raise_if_cancelled(cancellation)
        if not isinstance(raw, Mapping):
            raise ValueError("BGE-M3 returned an invalid feature result")
        vectors = self._validate_vectors(
            raw.get("dense_vecs"), expected_count=len(texts)
        )
        weights = raw.get("lexical_weights")
        if not isinstance(weights, list) or len(weights) != len(texts):
            raise ValueError("BGE-M3 returned an invalid sparse vector count")
        features: list[BgeM3Features] = []
        for dense, lexical in zip(vectors, weights, strict=True):
            if not isinstance(lexical, Mapping) or not lexical:
                raise ValueError("BGE-M3 returned an empty sparse vector")
            sparse: SparseVector = {}
            for raw_id, raw_weight in lexical.items():
                if not isinstance(raw_id, str) or not raw_id.isdecimal():
                    raise ValueError("BGE-M3 returned an invalid sparse token ID")
                token_id = int(raw_id)
                if (
                    token_id in sparse
                    or isinstance(raw_weight, bool)
                    or not isinstance(raw_weight, Real)
                    or not math.isfinite(float(raw_weight))
                    or float(raw_weight) <= 0
                ):
                    raise ValueError("BGE-M3 returned an invalid sparse weight")
                sparse[token_id] = float(raw_weight)
            features.append(BgeM3Features(dense=list(dense), sparse=sparse))
        return features

    def _encode_m3_sync(
        self,
        model: Any,
        mode: EmbeddingMode,
        texts: Sequence[str],
        return_sparse: bool,
    ) -> Any:
        method = (
            model.encode_queries if mode is EmbeddingMode.QUERY else model.encode_corpus
        )
        return method(
            list(texts),
            batch_size=self._max_batch_size,
            max_length=(
                self._query_max_tokens
                if mode is EmbeddingMode.QUERY
                else self._document_max_tokens
            ),
            return_dense=True,
            return_sparse=return_sparse,
            return_colbert_vecs=False,
        )

    def _score_m3_colbert_sync(
        self,
        model: Any,
        query: str,
        documents: Sequence[str],
        stop: threading.Event,
    ) -> list[float]:
        query_vectors = model.encode_queries(
            [query],
            batch_size=self._max_batch_size,
            max_length=self._query_max_tokens,
            return_dense=False,
            return_sparse=False,
            return_colbert_vecs=True,
        )["colbert_vecs"]
        if len(query_vectors) != 1:
            raise ValueError("BGE-M3 returned an invalid ColBERT vector count")
        scores: list[float] = []
        batch_size = min(self._max_batch_size, 4)
        for start in range(0, len(documents), batch_size):
            if stop.is_set():
                return []
            batch = list(documents[start : start + batch_size])
            document_vectors = model.encode_corpus(
                batch,
                batch_size=self._max_batch_size,
                max_length=self._document_max_tokens,
                return_dense=False,
                return_sparse=False,
                return_colbert_vecs=True,
            )["colbert_vecs"]
            if len(document_vectors) != len(batch):
                raise ValueError("BGE-M3 returned an invalid ColBERT vector count")
            if stop.is_set():
                return []
            scores.extend(
                float(model.colbert_score(query_vectors[0], vector))
                for vector in document_vectors
            )
        return scores

    def _encode_sync(
        self,
        model: Any,
        mode: EmbeddingMode,
        texts: Sequence[str],
    ) -> Any:
        if self.backend == "bge_m3":
            raw = self._encode_m3_sync(model, mode, texts, False)
            if not isinstance(raw, Mapping):
                raise ValueError("BGE-M3 returned an invalid dense result")
            return raw.get("dense_vecs")
        method_name = (
            "encode_query" if mode is EmbeddingMode.QUERY else "encode_document"
        )
        method = getattr(model, method_name, None)
        if not callable(method):
            raise RuntimeError(f"embedding model does not implement {method_name}")

        kwargs: dict[str, Any] = {}
        try:
            parameters = inspect.signature(method).parameters.values()
        except (TypeError, ValueError):
            parameters = ()
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters
        )
        names = {parameter.name for parameter in parameters}
        candidates = {
            "normalize_embeddings": True,
            "convert_to_numpy": True,
            "show_progress_bar": False,
        }
        for name, value in candidates.items():
            if accepts_kwargs or name in names:
                kwargs[name] = value
        return method(list(texts), **kwargs)

    def _validate_vectors(
        self,
        raw: Any,
        *,
        expected_count: int,
    ) -> list[tuple[float, ...]]:
        value = raw.tolist() if callable(getattr(raw, "tolist", None)) else raw
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise ValueError("embedding model returned a non-sequence result")
        if len(value) != expected_count:
            raise ValueError("embedding model returned an unexpected vector count")

        vectors: list[tuple[float, ...]] = []
        dimension = self._dimension
        for raw_vector in value:
            vector_value = (
                raw_vector.tolist()
                if callable(getattr(raw_vector, "tolist", None))
                else raw_vector
            )
            if isinstance(vector_value, (str, bytes)) or not isinstance(
                vector_value, Sequence
            ):
                raise ValueError("embedding model returned an invalid vector")
            if not vector_value:
                raise ValueError("embedding model returned an empty vector")

            vector: list[float] = []
            for element in vector_value:
                if isinstance(element, bool):
                    raise ValueError("embedding vector contains a boolean")
                try:
                    number = float(element)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(
                        "embedding vector contains a non-numeric value"
                    ) from exc
                if not math.isfinite(number):
                    raise ValueError("embedding vector contains a non-finite value")
                vector.append(number)

            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                raise ValueError("embedding vector dimension does not match runtime")
            vectors.append(tuple(vector))

        if dimension is None:
            raise ValueError("embedding model returned no vectors")
        self._dimension = dimension
        return vectors

    def _cache_key(
        self,
        text: str,
        *,
        mode: EmbeddingMode,
        scope: EmbeddingScope,
    ) -> _CacheKey:
        return _CacheKey(
            model_name=self.model_name,
            model_revision=self.model_revision,
            namespace=scope.namespace,
            tenant_id=scope.tenant_id,
            index_id=scope.index_id,
            mode=mode,
            text=text,
        )

    def _cache_get(self, key: _CacheKey) -> tuple[float, ...] | None:
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
        return cached

    def _normalize_input(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("embedding text must be a string")
        normalized = " ".join(unicodedata.normalize("NFKC", text).split())
        if not normalized:
            raise ValueError("embedding text cannot be empty")
        if len(normalized) > self._max_text_characters:
            raise ValueError("embedding text exceeds the configured character limit")
        return normalized

    def _raise_if_closed(self) -> None:
        if self._closed or self._closing:
            raise RuntimeError("embedding runtime is closed")

    def _resolve_deadline(self, deadline: float | None) -> float | None:
        if deadline is not None or self._default_timeout_seconds is None:
            return deadline
        return self._clock() + self._default_timeout_seconds

    def _create_default_model(self) -> Any:
        if self.backend == "bge_m3":
            from FlagEmbedding import BGEM3FlagModel
            from huggingface_hub import snapshot_download

            model_path = snapshot_download(
                repo_id=self.model_name,
                revision=self.model_revision or None,
            )
            return BGEM3FlagModel(
                model_path,
                devices=self.device,
                use_fp16=self.device.startswith("cuda"),
            )
        from sentence_transformers import SentenceTransformer

        kwargs: dict[str, Any] = {"device": self.device}
        if self.model_revision:
            kwargs["revision"] = self.model_revision
        return SentenceTransformer(self.model_name, **kwargs)

    def _assert_owner_loop(self) -> None:
        if (
            self._owner_loop is not None
            and self._owner_loop is not asyncio.get_running_loop()
        ):
            raise RuntimeError("embedding runtime belongs to another event loop")

    def _require_model_lock(self) -> asyncio.Lock:
        if self._model_lock is None:
            raise RuntimeError("embedding runtime has not started")
        return self._model_lock

    def _require_cache_lock(self) -> asyncio.Lock:
        if self._cache_lock is None:
            raise RuntimeError("embedding runtime has not started")
        return self._cache_lock

    def _require_encoder_gate(self) -> asyncio.Semaphore:
        if self._encoder_gate is None:
            raise RuntimeError("embedding runtime has not started")
        return self._encoder_gate

    def _require_capacity(self) -> asyncio.BoundedSemaphore:
        if self._capacity is None:
            raise RuntimeError("embedding runtime has not started")
        return self._capacity

    @staticmethod
    def _raise_if_cancelled(cancellation: CancellationProbe | None) -> None:
        if cancellation is not None and cancellation():
            raise asyncio.CancelledError("embedding call cancelled")


class EmbeddingService:
    """Synchronous embedding Interface backed by ``ProviderLoopBridge``."""

    def __init__(
        self,
        state_path: str | None = None,
        *,
        runtime: EmbeddingRuntime | None = None,
        bridge: ProviderLoopBridge,
        **runtime_kwargs: Any,
    ) -> None:
        del state_path
        if runtime is not None and runtime_kwargs:
            raise ValueError("runtime_kwargs cannot be used with an existing runtime")
        self._runtime = runtime or EmbeddingRuntime(**runtime_kwargs)
        self._bridge = bridge

    @property
    def runtime(self) -> EmbeddingRuntime:
        return self._runtime

    def embed_query(
        self,
        text: str,
        *,
        scope: EmbeddingScope | None = None,
        deadline: float | None = None,
        cancellation: CancellationProbe | None = None,
    ) -> Vector:
        return self._bridge.call_sync(
            lambda: self._runtime.embed_query(
                text,
                scope=scope,
                deadline=deadline,
                cancellation=cancellation,
            ),
            cancellation=cancellation,
        )

    def embed_m3_query(
        self,
        text: str,
        *,
        deadline: float | None = None,
        cancellation: CancellationProbe | None = None,
    ) -> BgeM3Features:
        return self._bridge.call_sync(
            lambda: self._runtime.embed_m3_features(
                [text],
                mode=EmbeddingMode.QUERY,
                deadline=deadline,
                cancellation=cancellation,
            ),
            cancellation=cancellation,
        )[0]

    def embed_m3_documents(
        self,
        texts: Sequence[str],
        *,
        deadline: float | None = None,
        cancellation: CancellationProbe | None = None,
    ) -> list[BgeM3Features]:
        return self._bridge.call_sync(
            lambda: self._runtime.embed_m3_features(
                texts,
                mode=EmbeddingMode.DOCUMENT,
                deadline=deadline,
                cancellation=cancellation,
            ),
            cancellation=cancellation,
        )

    def get_embeddings(
        self,
        texts: Sequence[str],
        *,
        scope: EmbeddingScope | None = None,
        deadline: float | None = None,
        cancellation: CancellationProbe | None = None,
    ) -> list[Vector]:
        return self._bridge.call_sync(
            lambda: self._runtime.embed_documents(
                texts,
                scope=scope,
                deadline=deadline,
                cancellation=cancellation,
            ),
            cancellation=cancellation,
        )

    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        scope: EmbeddingScope | None = None,
        deadline: float | None = None,
        cancellation: CancellationProbe | None = None,
    ) -> list[Vector]:
        return self.get_embeddings(
            texts,
            scope=scope,
            deadline=deadline,
            cancellation=cancellation,
        )

    def warmup(
        self,
        *,
        deadline: float | None = None,
        cancellation: CancellationProbe | None = None,
    ) -> EmbeddingReadiness:
        return self._bridge.call_sync(
            lambda: self._runtime.warmup(
                deadline=deadline,
                cancellation=cancellation,
            ),
            cancellation=cancellation,
        )

    def readiness(self) -> EmbeddingReadiness:
        return self._runtime.readiness()

    def close(self, *, close_bridge: bool = False) -> None:
        if not self._runtime.readiness().closed:
            self._bridge.call_sync(self._runtime.close)
        if close_bridge:
            self._bridge.close()


def _resolve_scope(
    scope: EmbeddingScope | None,
    *,
    namespace: str | None,
    tenant_id: str | None,
    index_id: str | None,
) -> EmbeddingScope:
    if scope is not None:
        if namespace is not None or tenant_id is not None or index_id is not None:
            raise ValueError("scope cannot be combined with individual scope fields")
        return scope
    return EmbeddingScope(
        namespace=namespace or "default",
        tenant_id=tenant_id or "default",
        index_id=index_id or "default",
    )


def _normalize_scope_part(value: str, fallback: str) -> str:
    if not isinstance(value, str):
        raise TypeError("embedding scope fields must be strings")
    normalized = " ".join(unicodedata.normalize("NFKC", value).split())
    return normalized or fallback


def _validate_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _consume_future_exception(future: asyncio.Future[tuple[float, ...]]) -> None:
    if not future.cancelled():
        future.exception()


__all__ = [
    "EmbeddingMode",
    "EmbeddingProvider",
    "EmbeddingReadiness",
    "EmbeddingRuntime",
    "EmbeddingRuntimeStats",
    "EmbeddingScope",
    "EmbeddingService",
]
