from pyml_ensemble.model import Model
from lstm import MyLSTM

class LSTMModel(Model):
    def __init__(self, input_size, num_hidden_layers, hidden_layer_sizes,
                output_size, epochs=50, batch_size=1, fit_verbose=2,
                num_timesteps=None, weight_file=''):
        super().__init__()

        self.model = MyLSTM(input_size, num_hidden_layers, hidden_layer_sizes,
                          output_size, epochs=epochs, batch_size=batch_size, 
                          fit_verbose=fit_verbose, num_timesteps=num_timesteps)

        self.weight_file = weight_file
        if weight_file == '':
            self.model.build_model()

    def train(self, x, y):
        self.model.train(x, y)

    def get_prediction(self, x):
        return self.model.predict(x)

