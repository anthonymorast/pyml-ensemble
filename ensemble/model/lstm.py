from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN, Dropout


class MyLSTM(object):
    def __init__(self, input_size, num_hidden_layers, hidden_layer_sizes, output_size,
                 epochs=50, batch_size=1, fit_verbose=2, variables=None):
        self.input_size = input_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = fit_verbose
        self.variables = variables

        self.build_model()

    def build_model(self):
        self.model = Sequential()
        self.model.add(LSTM(self.hidden_layer_sizes[0], input_shape=(self.input_size, self.variables),
                            return_sequences=True))
        # self.model.add(Dropout(0.2))
        for i in range(1, self.num_hidden_layers - 1):
            self.model.add(LSTM(self.hidden_layer_sizes[i], return_sequences=True))
            # self.model.add(Dropout(0.2))
        self.model.add(LSTM(self.hidden_layer_sizes[len(self.hidden_layer_sizes) - 1]))
        # self.model.add(Dropout(0.2))
        self.model.add(Dense(self.output_size))
        self.model.compile(loss='mean_squared_error', optimizer='adam')


    def predict(self, data):
        """
            Runs the data in the data parameter through the network and
            returns a list of predicted values.

             data - a matrix of data (explanatory variables) to be sent through the LSTM
        """
        return self.model.predict(data, batch_size=data.shape[1])


    def get_weights(self):
        """
            Returns the weights for each layer in the network (list of arrays).
        """
        return self.model.get_weights()


    def set_weights(self, weights):
        """
            Sets the weights of the network.
        """
        self.model.set_weights(weights)

    def save_model_weights(self, filename):
        """
            Saves the model weights to a file. File name should have extension
            'h5'.
        """
        self.model.save_weights(filename)


    def load_model_weights(self, filename):
        """
            Loads the model weights from a file. File name should have extension
            'h5'.
        """
        self.model.load_weights(filename)


    def train(self, train_x, train_y, optimzer='adam'):
        """
            Trains the model using the Adam optimization algortihm (more to be implemented
            later). Creates a 'history' attr of the LSTM.

            train_x - a matrix of explanatory variables for training
            train_y - a matrix of dependent variables to train on
            optimizer - optimization algorithm (Adam is the only one implemented)
         """
        self.history = self.model.fit(train_x, train_y, epochs=self.epochs, batch_size=self.batch_size,
                                      verbose=self.verbose, shuffle=False)
