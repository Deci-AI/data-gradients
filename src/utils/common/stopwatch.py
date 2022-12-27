from timeit import default_timer as timer


class Stopwatch:
    def __init__(self) -> None:
        super().__init__()
        self._timer = timer()
        self._ticks = [self._timer]

    def tick(self):
        self._ticks.append(timer())
        return self._ticks[-1] - self._ticks[-2]

    def total(self, make_tick=True):
        if make_tick:
            self.tick()
            now = self._ticks[-1]
        else:
            now = timer()
        return now - self._ticks[0]

    def tick_and_print(self, what):
        print("{} took: {:.4f}s".format(what, self.tick()))