"""
Tools for working with async tasks
https://github.com/Kirill888/jupyter-ui-poll/blob/develop/jupyter_ui_poll/_async_thread.py
"""
import asyncio
import threading


class AsyncThread:
    @staticmethod
    def _worker(loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()
        loop.close()

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=AsyncThread._worker, args=(self._loop,))
        self._thread.start()

    def terminate(self):
        def _stop(loop):
            loop.stop()

        if self._loop is None:
            return

        self.call_soon(_stop, self._loop)
        self._thread.join()
        self._loop, self._thread = None, None

    def __del__(self):
        self.terminate()

    def submit(self, func, *args, **kwargs):
        """
        Run async func with args/kwargs in separate thread, returns Future object.
        """
        return asyncio.run_coroutine_threadsafe(func(*args, **kwargs), self._loop)

    def wrap(self, func):
        def sync_func(*args, **kwargs):
            return self.submit(func, *args, **kwargs).result()

        return sync_func

    def call_soon(self, func, *args):
        """
        Call normal (non-async) function with arguments in the processing thread
        it's just a wrapper over `loop.call_soon_threadsafe()`

        Returns a handle with `.cancel`, not a full on Future
        """
        return self._loop.call_soon_threadsafe(func, *args)

    @property
    def loop(self):
        return self._loop
