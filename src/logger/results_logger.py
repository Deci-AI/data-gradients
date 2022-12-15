from abc import ABC, abstractmethod
import datetime as dt


class ResultsLogger(ABC):
    def __init__(self):
        self.logdir = "logs/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")

    @abstractmethod
    def log_graph(self, title: str, graph):
        pass

    @abstractmethod
    def log_text(self, title: str, text):
        pass
