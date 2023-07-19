import asyncio
import sys
import time
from collections import abc
from functools import singledispatch
from inspect import isawaitable, iscoroutinefunction
from typing import (
    Any,
    AsyncIterable,
    AsyncIterator,
    Callable,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
)

from IPython import get_ipython

from ._async_thread import AsyncThread

T = TypeVar("T")

ZMQ_POLLOUT = 2  # zmq.POLLOUT without zmq dependency


class KernelWrapper:
    _current: Optional["KernelWrapper"] = None

    def __init__(self, shell, loop) -> None:
        kernel = shell.kernel

        self._shell = shell
        self._kernel = kernel
        self._loop = loop
        self._original_parent = (
            kernel._parent_ident,
            kernel.get_parent() if hasattr(kernel, "get_parent") else kernel._parent_header,  # ipykernel 6+  # ipykernel < 6
        )
        self._events: List[Tuple[Any, Any, Any]] = []
        self._backup_execute_request = kernel.shell_handlers["execute_request"]
        self._aproc = None

        if iscoroutinefunction(self._backup_execute_request):  # ipykernel 6+
            kernel.shell_handlers["execute_request"] = self._execute_request_async
        else:
            # ipykernel < 6
            kernel.shell_handlers["execute_request"] = self._execute_request

        shell.events.register("post_execute", self._post_execute_hook)

    def restore(self):
        if self._backup_execute_request is not None:
            self._kernel.shell_handlers["execute_request"] = self._backup_execute_request
            self._backup_execute_request = None

    def _reset_output(self):
        self._kernel.set_parent(*self._original_parent)

    def _execute_request(self, stream, ident, parent):
        # store away execute request for later and reset io back to the original cell
        self._events.append((stream, ident, parent))
        self._reset_output()

    async def _execute_request_async(self, stream, ident, parent):
        self._execute_request(stream, ident, parent)

    async def replay(self):
        kernel = self._kernel
        self.restore()

        sys.stdout.flush()
        sys.stderr.flush()
        shell_stream = getattr(kernel, "shell_stream", None)  # ipykernel 6 vs 5 differences

        for stream, ident, parent in self._events:
            kernel.set_parent(ident, parent)
            if kernel._aborting:
                kernel._send_abort_reply(stream, parent, ident)
            else:
                rr = kernel.execute_request(stream, ident, parent)
                if isawaitable(rr):
                    await rr

                # replicate shell_dispatch behaviour
                sys.stdout.flush()
                sys.stderr.flush()
                if shell_stream is not None:  # 6+
                    kernel._publish_status("idle", "shell")
                    shell_stream.flush(ZMQ_POLLOUT)
                else:
                    kernel._publish_status("idle")

    async def do_one_iteration(self):
        try:
            rr = self._kernel.do_one_iteration()
            if isawaitable(rr):
                await rr
        except Exception:  # pylint: disable=broad-except
            # it's probably a bug in ipykernel,
            # .do_one_iteration() should not throw
            return
        finally:
            # reset stdio back to original cell
            self._reset_output()

    def _post_execute_hook(self, *args, **kw):
        self._shell.events.unregister("post_execute", self._post_execute_hook)
        self.restore()
        KernelWrapper._current = None
        asyncio.ensure_future(self.replay(), loop=self._loop)

    async def _poll_async(self, n=1):
        for _ in range(n):
            await self.do_one_iteration()

    async def __aenter__(self):
        return self._poll_async

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    def __enter__(self):
        if self._aproc is not None:
            raise ValueError("Nesting not supported")
        self._aproc = AsyncThread()
        return self._aproc.wrap(self._poll_async)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._aproc.terminate()
        self._aproc = None

    @staticmethod
    def get() -> "KernelWrapper":
        if KernelWrapper._current is None:
            KernelWrapper._current = KernelWrapper(get_ipython(), asyncio.get_event_loop())
        return KernelWrapper._current


class IteratorWrapperAsync(abc.AsyncIterable, Generic[T]):
    def __init__(
        self,
        its: AsyncIterable[T],
        n: int = 1,
    ):
        self._its = its
        self._n = n

    def __aiter__(self) -> AsyncIterator[T]:
        async def _loop(kernel: KernelWrapper, its: AsyncIterable[T], n: int) -> AsyncIterator[T]:
            async with kernel as poll:
                async for x in its:
                    await poll(n)
                    yield x

        return _loop(KernelWrapper.get(), self._its, self._n)


class IteratorWrapper(abc.Iterable, Generic[T]):
    def __init__(
        self,
        its: Iterable[T],
        n: int = 1,
    ):
        self._its = its
        self._n = n

    def __iter__(self) -> Iterator[T]:
        def _loop(kernel: KernelWrapper, its: Iterable[T], n: int) -> Iterator[T]:
            with kernel as poll:
                try:
                    for x in its:
                        poll(n)
                        yield x
                except GeneratorExit:
                    pass
                except Exception as e:
                    raise e

        return _loop(KernelWrapper.get(), self._its, self._n)

    def __aiter__(self) -> AsyncIterator[T]:
        async def _loop(kernel: KernelWrapper, its: Iterable[T], n: int) -> AsyncIterator[T]:
            async with kernel as poll:
                for x in its:
                    await poll(n)
                    yield x

        return _loop(KernelWrapper.get(), self._its, self._n)


def ui_events():
    """
    Gives you a function you can call to process UI events while running a long
    task inside a Jupyter cell.

    .. code-block:: python

       with ui_events() as ui_poll:
          while some_condition:
             ui_poll(10)  # Process upto 10 UI events if any happened
             do_some_more_compute()

    Async mode is also supported:

    .. code-block:: python

       async with ui_events() as ui_poll:
          while some_condition:
             await ui_poll(10)  # Process upto 10 UI events if any happened
             do_some_more_compute()


    #. Call ``kernel.do_one_iteration()`` taking care of IO redirects
    #. Intercept ``execute_request`` IPython kernel events and delay their execution
    #. Schedule replay of any blocked ``execute_request`` events when
       cell execution is finished.
    """
    return KernelWrapper.get()


@singledispatch
def with_ui_events(its, n: int = 1):
    """
    Deal with kernel ui events while processing a long sequence

    Iterable returned from this can be used in both async and sync contexts.

    .. code-block:: python

       for x in with_ui_events(some_data_stream, n=10):
          do_things_with(x)

       async for x in with_ui_events(some_data_stream, n=10):
          await do_things_with(x)


    This is basically equivalent to:

    .. code-block:: python

       with ui_events() as poll:
         for x in some_data_stream:
             poll(10)
             do_things_with(x)


    :param its:
      Iterator to pass through, this should be either
      :class:`~collections.abc.Iterable` or :class:`~collections.abc.AsyncIterable`

    :param n:
      Number of events to process in between items

    :returns:
      :class:`~collections.abc.AsyncIterable` when input is
      :class:`~collections.abc.AsyncIterable`

    :returns:
      Object that implements both :class:`~collections.abc.Iterable` and
      :class:`~collections.abc.AsyncIterable` interfaces when input is normal
      :class:`~collections.abc.Iterable`
    """
    raise TypeError("Expect Iterable[T]|AsyncIterable[T]")


@with_ui_events.register(abc.Iterable)
def with_ui_events_sync(its: Iterable[T], n: int = 1) -> IteratorWrapper[T]:
    return IteratorWrapper(its, n=n)


@with_ui_events.register(abc.AsyncIterable)
def with_ui_events_async(its: AsyncIterable[T], n: int = 1) -> AsyncIterable[T]:
    return IteratorWrapperAsync(its, n=n)


def run_ui_poll_loop(f: Callable[[], Optional[T]], sleep: float = 0.02, n: int = 1) -> T:
    """
    Repeatedly call ``f()`` until it returns something other than ``None``
    while also responding to widget events.

    This blocks execution of cells below in the notebook while still preserving
    interactivity of jupyter widgets.

    :param f:
       Function to periodically call (``f()`` should not block for long)

    :param sleep:
       Amount of time to sleep in between polling (in seconds, 1/50 is the default)

    :param n:
       Number of events to process per iteration

    :returns:
       First non-``None`` value returned from ``f()``
    """

    def as_iterator(f: Callable[[], Optional[T]], sleep: float) -> Iterator[Optional[T]]:
        x = None
        while x is None:
            if sleep is not None:
                time.sleep(sleep)

            x = f()
            yield x

    for x in with_ui_events(as_iterator(f, sleep), n):
        if x is not None:
            return x

    raise RuntimeError("hm...")  # for mypy sake
