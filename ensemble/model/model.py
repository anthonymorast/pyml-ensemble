import abc

class Model(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def train(self, x, y):
        pass

    @abc.abstractmethod
    def get_prediction(self, x):
        pass

    def custom_stat(self, stat):
        """ Used to define custom aggregators. This stat can be used to match
            with a stat in the aggregator. For example, if this were the k-NN
            aggregator this would store the mean/variance values (somehow) so
            that they can be compared with the data chunk's mean/variance.

            Need to think of the best way to handle custom aggregators. Could
            also use child classes of the model too...
        """
        pass
