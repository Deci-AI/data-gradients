import concurrent
from concurrent.futures import ThreadPoolExecutor


class ThreadManager:
    def __init__(self):
        self.thread = ThreadPoolExecutor()
        self.futures = []

    def submit(self, fn, *args):
        self.futures.append(self.thread.submit(fn, *args))

    def wait_complete(self):
        concurrent.futures.wait(self.futures, return_when=concurrent.futures.ALL_COMPLETED)
