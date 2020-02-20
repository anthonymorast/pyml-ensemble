from .aggregator import Aggregator

class KalmanAggregator(Aggregator):
    def __init__(self):
        super().__init__()

    def combine(self, predictions):
        pass
