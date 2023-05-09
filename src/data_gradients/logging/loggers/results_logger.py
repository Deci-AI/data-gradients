from abc import ABC, abstractmethod


class ResultsLogger(ABC):
    def __init__(self, log_dir: str):
        self.logdir = log_dir

    @abstractmethod
    def log(self, title: str, data):
        pass

    @abstractmethod
    def close(self):
        pass
