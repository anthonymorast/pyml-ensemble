from .model.model import Model
from .aggregator.aggregator import Aggregator

class Ensemble(object):
    def __init__(self):
        self.models = []
        self.aggregator = None

    def add_models(self, models):
        for model in models:
            self.add_model(model)

    def add_model(self, model):
        if not isinstance(model, Model):
            raise TypeError("model must be of type pyml_ensemble.model.Model")
        self.models.append(model)

    def set_aggregator(self, aggregator):
        if not isinstance(aggregator, Aggregator):
            raise TypeError("aggregator must be of type pyml_ensemble.aggregator.Aggregator")
        self.aggregator = aggregator

    def train(self, x, y):
        """
            par x - a list of training data samples, one list entry for each model.
            par y - a list of training data targets, one list entry for each model.
                    Note: x[i] <-> y[i]
        """
        for i in range(0, len(self.models)):
            self.models[i].train(x[i], y[i])

    def predict(self, x):
        if self.aggregator is None:
            raise RuntimeError("no aggregator defined for ensemble")
        predictions = []
        for model in self.models:
            predictions.append(model.get_prediction(x))
        return self.aggregator.combine(predictions)

    def call_all(self, function_name):
        """
            Calls a function for each ensemble method by name and returns a list
            of return values from the method call.
        """
        return_list = []
        for model in self.models:
            return_list.append(getattr(model, function_name)())
        return return_list
