from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn import metrics

from pyml_ensemble import Ensemble
from pyml_ensemble.aggregator import ModeAggregator
from pyml_ensemble.model import TreeModel

ensemble = Ensemble()
aggregator = ModeAggregator()
ensemble.set_aggregator(aggregator)

iris = load_iris()
trainx, testx, trainy, testy = train_test_split(iris.data, iris.target, test_size=0.33)

num_models = 30
for i in range(num_models):
    tree = TreeModel()
    ensemble.add_model(tree)

ensemble.train([trainx for _ in range(num_models)], [trainy for _ in range(num_models)])
y_hat = ensemble.predict(testx)
print(metrics.mean_squared_error(testy, y_hat))
