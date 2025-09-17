import asyncio
from types import SimpleNamespace

from trend_analysis.gui import utils


def test_debounce_cancels_pending_call(monkeypatch):
    """Debounce should cancel prior tasks and only execute latest
    invocation."""

    # Controlled time progression for debounce window checks
    time_values = iter([0.0, 0.05, 0.35, 0.70, 1.0])
    monkeypatch.setattr(utils.time, "time", lambda: next(time_values))

    async def fast_sleep(_):
        return None

    monkeypatch.setattr(utils.asyncio, "sleep", fast_sleep)

    tasks: list[SimpleNamespace] = []
    original_create_task = asyncio.create_task

    def tracking_create_task(coro):
        task = original_create_task(coro)
        wrapper = SimpleNamespace(_task=task, cancelled=False)

        def cancel() -> None:
            wrapper.cancelled = True
            task.cancel()

        wrapper.cancel = cancel
        tasks.append(wrapper)
        return wrapper

    monkeypatch.setattr(utils.asyncio, "create_task", tracking_create_task)

    calls: list[str] = []

    @utils.debounce(300)
    def handler(value: str) -> None:
        calls.append(value)

    async def run() -> None:
        await handler("first")
        await handler("second")
        await asyncio.sleep(0)

    asyncio.run(run())

    assert calls == ["second"]
    assert tasks and tasks[0].cancelled is True


def test_debounce_awaits_async_handler(monkeypatch):
    """Async callbacks returned by debounce should be awaited before
    finishing."""

    time_values = iter([0.0, 0.5])
    monkeypatch.setattr(utils.time, "time", lambda: next(time_values))

    async def fast_sleep(_):
        return None

    monkeypatch.setattr(utils.asyncio, "sleep", fast_sleep)

    calls: list[str] = []

    @utils.debounce(50)
    async def handler(value: str) -> None:
        calls.append(value)

    async def run() -> None:
        await handler("async")
        await asyncio.sleep(0)

    asyncio.run(run())

    assert calls == ["async"]


def test_debounce_waits_for_elapsed_time():
    calls: list[str] = []

    @utils.debounce(10)
    def handler(value: str) -> None:
        calls.append(value)

    async def run() -> None:
        await handler("first")
        await asyncio.sleep(0.05)

    asyncio.run(run())

    assert calls == ["first"]


def test_debounce_sync_handler_returning_coroutine(monkeypatch):
    """Synchronous handlers that return coroutines should be awaited."""

    time_values = iter([0.0, 0.05, 0.10, 0.15])
    monkeypatch.setattr(utils.time, "time", lambda: next(time_values))

    async def fast_sleep(_):
        return None

    monkeypatch.setattr(utils.asyncio, "sleep", fast_sleep)

    calls: list[str] = []

    @utils.debounce(10)
    def handler(value: str):
        async def inner() -> None:
            calls.append(value)

        return inner()

    async def run() -> None:
        await handler("coroutine")
        await asyncio.sleep(0)

    asyncio.run(run())

    assert calls == ["coroutine"]


def test_debounce_skips_call_when_wait_not_elapsed(monkeypatch):
    """Callbacks should be suppressed if the debounce window has not passed."""

    time_values = iter([0.0, 0.05])
    monkeypatch.setattr(utils.time, "time", lambda: next(time_values))

    async def fast_sleep(_: float) -> None:
        return None

    monkeypatch.setattr(utils.asyncio, "sleep", fast_sleep)

    calls: list[str] = []

    @utils.debounce(200)
    def handler(value: str) -> None:
        calls.append(value)

    async def run() -> None:
        await handler("first")
        await asyncio.sleep(0)

    asyncio.run(run())

    assert calls == []
