import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics

from ensemble import Ensemble
from ensemble.aggregator import MeanAggregator
from ensemble.model import ANNModel

def y(x):
    return (x**2)

if __name__ == '__main__':
    ensemble = Ensemble()
    aggregator = MeanAggregator()
    ensemble.set_aggregator(aggregator)

    x = np.linspace(0, 20, num=2000)
    y = y(x)

    trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.33)

    num_models = 2
    for i in range(num_models):
        ann = ANNModel(1, 2, [5, 5], 1, epochs=2500, batch_size=16)
        ensemble.add_model(ann)

    # train each model with the same data in this case
    ensemble.train([trainx for _ in range(num_models)], [trainy for _ in range(num_models)])
    # for x_val in np.nditer(testx):
    #     y_val = ensemble.predict(np.atleast_2d(x_val))
    #     print(x_val, "^2", "=", y_val)
    yhat = ensemble.predict(testx)
    print(metrics.mean_squared_error(testy, yhat))


    """
        To adapt, keep a running list of previous (x, y) pairs.
        Once the list is long enough, add a new model to the ensemble and
        train just that model (adding model_count() and train_model_at() to
        accomodate this functionality).

        Another way to adapt the ensemble is by retraining with new data. In
        this case append new data points onto the original trainx, trainy dataframes
        then just call ensemble.train() again with the updated lists of training data.

        Should we create an Adaptation model that performs these two types of
        adaptation?
    """
