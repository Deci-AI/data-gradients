import concurrent
from concurrent.futures import ThreadPoolExecutor


class ThreadManager:
    """Class responsible to manage and execute multiple functions concurrently using a thread pool.

    :Example:

        >>> from time import sleep
        >>> def wait_function(wait_time):
        ...     print(f"Waiting for {wait_time} seconds")
        ...     sleep(wait_time)
        ...     print("Function execution complete")
        ...
        >>> manager = ThreadManager()
        >>> manager.submit(wait_function, 2)
        >>> manager.submit(wait_function, 1)
        >>> manager.wait_complete()
        Waiting for 1 seconds
        Waiting for 2 seconds
        Function execution complete
        Function execution complete
    """

    def __init__(self):
        self.thread = ThreadPoolExecutor()
        self.futures = []

    def submit(self, fn: callable, *args):
        """Submit a function to be executed in the thread pool.

        :param fn:      Function to execute.
        :param args:    Arguments to pass to the function.
        """
        self.futures.append(self.thread.submit(fn, *args))

    def wait_complete(self):
        """Wait until all submitted functions have completed execution."""
        concurrent.futures.wait(self.futures, return_when=concurrent.futures.ALL_COMPLETED)
