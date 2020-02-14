from .aggregator import Aggregator

class MeanAggregator(Aggregator):
    def __init__(self):
        super().__init__()

    def combine(self, predictions):
        s = sum(predictions)
        return s/len(predictions)
