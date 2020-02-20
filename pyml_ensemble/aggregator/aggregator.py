import abc

class Aggregator(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def combine(self, predictions):
        pass
