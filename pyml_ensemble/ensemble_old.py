"""
	The Ensemble class sets up the ensemble including training style,
	number of base networks, train size of the data, segment size for
	each network, and other general parameters for the networks.

	Author: Anthony Morast
	Date: 4/23/2018
"""

from method import *
from data_handler import *
from errors import *
from lstm import *

import numpy as np

class Ensemble(object):
    def __init__(self, train_style='overlap', num_segments=5, base_size=-1,
                 trainsize=2000, lookback=1, k=-1, verbose=0, load_train='t',
                 dataset='s'):
        """
            train_style - defines how the data will be segmented for the ensemble methods
                         (sequential, overlap, random_segments)
            num_segments - the number of segments for the training data
            base_size - only valid for 'overlap' train style. Determine the size
                        of the training data for each method
            trainsize - number of samples to use for training
            lookback - how many past data points to use as inputs. Usually 1 works best
            k - kNN; number of nearest neighbors to use for prediction
            verbose - determine the level of output (0, 1, 2)
            load_train - 't' to train, 'l' to load
            dataset - 's' for sunspot, 'm' for mackey-glass, 'e' for eurusd network params
        """
        if train_style != 'overlap' and base_size > 0:
            print("Warning: base_size parameter ignored for training style...")

        self.methods = []
        self.train_style = train_style
        self.num_segments = num_segments
        self.base_size = base_size
        self.trainsize = trainsize
        self.look_back = lookback
         # only used for overlap training
        if num_segments > 1:
            self.shift = int((self.trainsize - self.base_size)/(self.num_segments-1))
        else:
            self.shift = int((self.trainsize - self.base_size)/(self.num_segments))
        if train_style == 'overlap':
            self.segment_size = base_size
        elif train_style == 'sequential':
            self.segment_size = int(self.trainsize/self.num_segments)
        self.k = k
        if k == -1:
            self.k = num_segments
        self.data_set = False
        self.verbose = verbose
        self.load_train = load_train
        self.params = None
        self.dataset = dataset
        self.weight_file_count = 0

        self.trainx = []
        self.trainy = []
        self.testx = []
        self.testy = []

    def get_predictions(self, adaptive=True, window_size=None):
        """
            Iterates the test data getting predictions for each timestep. Returns
            a list containing the predictions over the test set.

            adaptive - if true, trains new networks as data become available
            window_size - size of window for current variance and mean values
        """
        if window_size is None:
            window_size = self.segment_size

        ## Trains way too many networks
        # if self.train_style == 'overlap':
        #     histx = self.trainx[len(self.trainx)-1][-(self.segment_size - self.shift):].tolist()
        #     histy = self.trainy[len(self.trainy)-1][-(self.segment_size - self.shift):].tolist()
        # else:
        #     histx, histy = [], []
        histx, histy = [], []
        predictions = []
        win_size = window_size
        print("Getting predictions, this may take some time...")
        for i in range(self.testsize - 1):
            # get sliding window for statistical properties kNN
            idx = self.trainsize + self.look_back - win_size + i + 1
            window = self.x[idx:idx + win_size]
            xp, yp = self.testx[i], self.testy[i]
            # observations to train new networks
            histy.append(yp)
            histx.append(xp.tolist())
            xp = xp.reshape((1, xp.shape[0], xp.shape[1]))
            # print("TARGET: ", yp)
            if adaptive and len(histx) == self.segment_size:
                # train new network
                self.train_or_load(np.asarray(histx), np.asarray(histy))
                histx, histy = [], []
                # if self.train_style == 'overlap':
                #     histx = histx[-(self.segment_size - self.shift):]
                #     histy = histy[-(self.segment_size - self.shift):]
                # else:
                #     histx, histy = [], []
                # print("MSE:", mse(self.testy.tolist()[0:i], predictions))
            prediction = self.get_prediction(xp, window.mean(), window.var())
            predictions.append(prediction[0])

        return predictions


    def get_prediction(self, x, mean, variance):
        """
            Gets the prediction from the Ensemble

            x - input
            mean - sliding window mean
            variance - sliding window variance
        """
        prediction = 0
        weights = []
        for method in self.methods:
            distance = method.get_distance(mean, variance)
            weight = .5
            if distance != 0:
                weight = 1/(distance)#**2)
            weights.append(weight)
            prediction += (method.get_prediction(x).reshape(1)*weight)
        #     print(method.get_prediction(x).reshape(1), method.variance, method.mean, variance, mean, distance, weight)
        # print('Prediction:', prediction/sum(weights))
        return prediction / sum(weights)


    def create_methods(self, batch_size=100, verbose=2, epochs=1850, params='s'):
        """
            Creates the methods that make up the ensemble. The method class
            includes the methods variance and mean so the data must be defined
            first.
        """
        if not self.data_set:
            print("Must set data on ensemble before creating the methods.\n\
                   \tensemble.set_data_from_file(filename)\n\
                   \tensemble.create_datasets()")
            exit()

        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.params = params

        for i in range(len(self.trainx)):
            x, y = self.trainx[i], self.trainy[i]
            lstm = self.get_method(x, params)
            self.methods.append(Method(lstm, x.mean(), x.var()))
        return

    def get_method(self, x, params='s'):
        """
            Returns the method being used in the ensemble.
        """
		# different parameters for different datasets
        sunspots_params = [4, [27, 25, 3, 45]]
        eur_usd_params = [4, [7, 16, 39, 9]]
        mackey_params = [9, [28, 4, 36, 47, 2, 4, 5, 1, 37]]

        param_set = sunspots_params
        if params == 'e':
            param_set = eur_usd_params
        elif params == 'm':
            param_set = mackey_params
        return MyLSTM(x.shape[1], param_set[0], param_set[1], 1,
                      epochs=self.epochs, batch_size=self.batch_size,
                      fit_verbose=self.verbose, variables=x.shape[2])

    def create_datasets(self):
        """
            Segments the data based on the ensemble's training style
        """
        train = self.x[:self.trainsize+self.look_back]
        test = self.x[self.trainsize+1:]

        trainx, trainy = self.create_lookback_dataset(train)
        testx, testy = self.create_lookback_dataset(test)

        # segment the data based on traning style
        if self.train_style == 'sequential':
            trainx = self.chunk_data(trainx)
            trainy = self.chunk_data(trainy)
        elif self.train_style == 'random':
            print("'random' training not yet implemented.")
            exit()
        elif self.train_style == 'overlap':
            trainx = [trainx[(i*self.shift):(self.base_size + (i*self.shift))]
                      for i in range(self.num_segments)]
            trainy = [trainy[(i*self.shift):(self.base_size + (i*self.shift))]
                      for i in range(self.num_segments)]
        else:
            print("Invalid training style for ensemble.")
            exit()

        self.trainx, self.trainy = trainx, trainy
        self.testx, self.testy = testx, testy
        return


    def train_methods(self):
        """
            Trains the methods in the ensemble.
        """
        if len(self.methods) == 0:
            print("No methods to train! Create the methods first:\n\
                   \tensemble.create_methods()")
            exit()

        # load or train the networks weight based on load_train parameter
        for i in range(len(self.methods)):
            x, y = self.trainx[i], self.trainy[i]
            method = self.methods[i].network
            if self.load_train == 'l':
                print('Loading weights for method ', i+1, '...')
                if self.dataset == 'm':
                    method.load_model_weights('./weights/mackey/mackey' +
                                            str(self.weight_file_count) + '.h5')
                elif self.dataset == 'e':
                    method.load_model_weights('./weights/eurusd/eurusd' +
                                            str(self.weight_file_count) + '.h5')
                else:
                    method.load_model_weights('./weights/sunspots/sunspots' +
                                            str(self.weight_file_count) + '.h5')
                self.weight_file_count += 1
            else:
                print('Training method', i+1, 'out of', len(self.methods), '...')
                method.train(x, y)
        return


    def train_or_load(self, x, y):
        """
            Loads or trains a new method based on the ensembles parameters.
        """
        lstm = self.get_method(x, self.params)
        if self.load_train == 'l':
            print('Loading weights of new ensemble method...')
            if self.dataset == 'm':
                lstm.load_model_weights('./weights/mackey/mackey' +
                                        str(self.weight_file_count) + '.h5')
            elif self.dataset == 'e':
                lstm.load_model_weights('./weights/eurusd/eurusd' +
                                        str(self.weight_file_count) + '.h5')
            else:
                lstm.load_model_weights('./weights/sunspots/sunspots' +
                                        str(self.weight_file_count) + '.h5')
            self.weight_file_count += 1
        else:
            print("Training new ensemble method...")
            lstm.train(x, y)

        self.methods.append(Method(lstm, x.mean(), x.var()))


    def set_data_from_file(self, filename):
        """
            Sets the x and y data from the given filename.
        """
        self.dh = DataHandler(filename)
        self.dh.timeSeriesToSupervised()
        self.data = self.dh.tsdata.values
        self.x, self.y = self.data[:,1], self.data[:,3]
        self.testsize = len(self.data) - self.trainsize - self.look_back
        self.data_set = True


    def chunk_data(self, data):
        """
            Splits data into n chunks, all evenly sized except (perhaps) the last one.
        """
        n = self.segment_size
        return [data[i:i+n] for i in range(0, data.shape[0], n)]


    def create_lookback_dataset(self, data):
        """
            Builds arrays containing previous information for the x inputs.
        """
        lookback = self.look_back

        datax, datay = [], []
        if lookback == 0:
            lookback = 1
        rng = len(data)-lookback
        for i in range(rng):
            datax.append(data[i:i+lookback])
            datay.append(data[i + lookback])
        arr = np.asarray(datax)
        return arr.reshape(arr.shape[0], 1, arr.shape[1]), np.asarray(datay)


    def save_method_weights(self, base_file_name):
        """
            Save the weights of all of the models in the ensemble.
        """
        for i in range(len(self.methods)):
            method = self.methods[i].network
            filename = base_file_name + str(i) + '.h5'
            method.save_model_weights(filename)
