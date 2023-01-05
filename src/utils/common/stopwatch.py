import datetime
from timeit import default_timer as timer

import numpy as np


class Stopwatch:
    def __init__(self) -> None:
        super().__init__()
        self._ticks = [timer()]

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

    def estimate_total_time(self, train_len, val_len) -> str:
        total_seconds = val_len * self._ticks[-1] + (train_len - val_len) * self._ticks[-1] * 0.75
        total_time = str(datetime.timedelta(seconds=total_seconds))
        return total_time