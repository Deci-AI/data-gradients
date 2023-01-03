import datetime
from timeit import default_timer as timer

import numpy as np


class Stopwatch:
    def __init__(self) -> None:
        super().__init__()
        self._timer = timer()
        self._ticks = [self._timer]

    @property
    def ticks(self):
        return self._ticks

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

    def average(self):
        return np.average(self._ticks)

    def tick_and_print(self, what):
        print("{} took: {:.4f}s".format(what, self.tick()))

    def get_total_time(self, train_len, val_len) -> str:
        total_seconds = val_len * self._ticks[-1] + (train_len - val_len) * self._ticks[-1] * 0.75
        total_time = str(datetime.timedelta(seconds=total_seconds))
        return total_time