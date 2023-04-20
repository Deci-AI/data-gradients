from abc import ABC, abstractmethod
import datetime as dt

log_dir = "logs/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")


class ResultsLogger(ABC):
    def __init__(self):
        self.logdir = log_dir

    @abstractmethod
    def log(self, title: str, data):
        pass

    @abstractmethod
    def close(self):
        pass
