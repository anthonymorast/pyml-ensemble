from pyml_ensemble.model import Model
from ann import ANN

class ANNModel(Model):
    def __init__(self, input_size, num_hidden_layers, hidden_layer_sizes,
                output_size, epochs=50, batch_size=1, fit_verbose=2,
                variables=None, weight_file=''):
        super().__init__()
        self.weight_file = weight_file
        self.model = ANN(input_size, num_hidden_layers, hidden_layer_sizes,
                        output_size, epochs=epochs, batch_size=batch_size,
                        fit_verbose=fit_verbose, variables=variables)

    def train(self, x, y):
        self.model.train(x, y)

    def get_prediction(self, x):
        return self.model.predict(x)

    def load_weights(self):
        self.model.set_weights(self.weight_file)

    def save_weights(self):
        self.model.save_weights(self.weight_file)

    def set_weight_filename(self, filename):
        self.weight_file = filename
