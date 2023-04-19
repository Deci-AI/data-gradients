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
        train_load = self._ticks[-3]
        val_load = self._ticks[-2]
        features_half = self._ticks[-1] / 2
        total_train = train_load + features_half
        total_val = val_load + features_half

        total_seconds = (val_len - 1) * total_val + (train_len - 1) * total_train
        total_time = str(datetime.timedelta(seconds=total_seconds))
        return total_time


from typing import List


class Timer:
    def __init__(self) -> None:
        self.start_time = timer()
        self.stop_time = None

    def stop(self):
        if self.stop_time is not None:
            raise RuntimeError("Timer already stopped")
        self.stop_time = timer() - self.start_time

    @property
    def elapsed(self):
        return self.stop_time - self.start_time
