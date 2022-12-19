from abc import ABC, abstractmethod
import datetime as dt


class ResultsLogger(ABC):
    def __init__(self):
        self.logdir = "logs/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")

    @abstractmethod
    def log(self, title: str, data):
        pass

    @abstractmethod
    def close(self):
        pass
