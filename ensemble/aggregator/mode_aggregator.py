from .aggregator import Aggregator
from scipy import stats

class ModeAggregator(Aggregator):
    def __init__(self):
        super().__init__()

    def combine(self, predictions):
        return stats.mode(predictions)[0][0]
