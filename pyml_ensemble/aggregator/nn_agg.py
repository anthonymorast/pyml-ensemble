from .aggregator import Aggregator

class AnnAggregator(Aggregator):
    def __init__(self):
        super().__init__()

    def combine(self, predictions):
        pass

    def train(self, x, y):
        pass
