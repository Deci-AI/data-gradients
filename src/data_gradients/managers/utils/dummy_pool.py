class DummyPool:
    def __init__(self, num_workers):
        if num_workers > 0:
            raise ValueError("DummyPool can't have workers")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def imap_unordered(self, func, iterable, chunksize=1):
        for sample in iterable:
            yield func(sample)

    def imap(self, func, iterable):
        for sample in iterable:
            yield func(sample)

    def starmap(self, func, iterable):
        return [func(*args) for args in iterable]
