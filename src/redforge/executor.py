"""Executor — bounded-concurrency runner with retry and budget. DESIGN.md §6.3."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from typing import Literal

from pydantic import BaseModel

from redforge.types import AttackPrompt, ErrorClass, TargetResponse


class ExecutorConfig(BaseModel):
    max_concurrency: int = 8  # target-call parallelism
    judge_concurrency: int = 4  # separate semaphore for judge calls
    per_request_timeout_s: float = 30.0
    max_retries: int = 3
    retry_initial_delay_s: float = 1.0
    retry_jitter_factor: float = 0.3  # full-jitter exponential backoff
    max_total_duration_s: float = 300.0
    max_cost_usd: float | None = None
    cost_cap_action: Literal["fail", "degrade_to_heuristic"] = "degrade_to_heuristic"


def classify_error(exc: BaseException) -> ErrorClass:
    """Map a raised exception to an ErrorClass for ScanResult observability."""
    if isinstance(exc, asyncio.TimeoutError):
        return ErrorClass.TIMEOUT
    if isinstance(exc, asyncio.CancelledError):
        return ErrorClass.CANCELLED
    # HTTP-status-aware classification (when SDKs raise structured errors with a
    # `.status_code` or `.response.status_code` attribute).
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    if isinstance(status, int):
        if status == 429:
            return ErrorClass.RATE_LIMIT
        if 500 <= status < 600:
            return ErrorClass.HTTP_5XX
        if 400 <= status < 500:
            return ErrorClass.HTTP_4XX
    return ErrorClass.USER_CALLABLE_EXCEPTION


def _is_retryable(error_class: ErrorClass) -> bool:
    return error_class in (
        ErrorClass.TIMEOUT,
        ErrorClass.RATE_LIMIT,
        ErrorClass.HTTP_5XX,
    )


class Executor:
    """Runs prompts against a target with concurrency, retry, budget enforcement.

    Domain-agnostic — knows nothing about attacks or scoring.
    """

    def __init__(
        self,
        target: Callable[[str], Awaitable[TargetResponse]],
        config: ExecutorConfig,
    ) -> None:
        self._target = target
        self._config = config
        self._sem = asyncio.Semaphore(config.max_concurrency)

    async def _call_with_retry(
        self, prompt_text: str
    ) -> tuple[TargetResponse | BaseException, int, ErrorClass]:
        attempts = 0
        last_error: BaseException | None = None
        last_class = ErrorClass.NONE
        while attempts <= self._config.max_retries:
            try:
                resp = await asyncio.wait_for(
                    self._target(prompt_text),
                    timeout=self._config.per_request_timeout_s,
                )
                return resp, attempts, ErrorClass.NONE
            except asyncio.CancelledError:
                raise
            except BaseException as e:  # noqa: BLE001 — we re-categorize
                last_error = e
                last_class = classify_error(e)
                if not _is_retryable(last_class) or attempts == self._config.max_retries:
                    return e, attempts, last_class
                # Full-jitter exponential backoff.
                base = self._config.retry_initial_delay_s * (2 ** attempts)
                jitter = base * self._config.retry_jitter_factor
                await asyncio.sleep(random.uniform(0, base + jitter))
                attempts += 1
        assert last_error is not None
        return last_error, attempts, last_class

    async def run(
        self, prompts: Iterable[AttackPrompt]
    ) -> AsyncIterator[
        tuple[AttackPrompt, TargetResponse | BaseException, float, int, ErrorClass]
    ]:
        """Yield (prompt, response_or_error, duration_ms, retry_attempts, error_class)
        for each prompt as it completes."""

        async def _one(
            p: AttackPrompt,
        ) -> tuple[AttackPrompt, TargetResponse | BaseException, float, int, ErrorClass]:
            async with self._sem:
                started = time.perf_counter()
                result, attempts, err_class = await self._call_with_retry(p.prompt)
                duration_ms = (time.perf_counter() - started) * 1000
                return p, result, duration_ms, attempts, err_class

        tasks = [asyncio.create_task(_one(p)) for p in prompts]
        deadline = (
            time.monotonic() + self._config.max_total_duration_s
            if self._config.max_total_duration_s > 0
            else None
        )
        for task in asyncio.as_completed(tasks):
            if deadline is not None and time.monotonic() > deadline:
                # Total-budget exceeded. Cancel remaining and stop yielding.
                for t in tasks:
                    if not t.done():
                        t.cancel()
                break
            yield await task
