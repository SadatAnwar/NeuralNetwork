import copy
import time

import climate
import numpy as np
import theanets

import utils
from RawData import RawData
from TrainingTimeSeries import TrainingTimeSeries

climate.enable_default_logging()


def test_RawData_read_csv_multiple_features(algo, layers, train, valid, test):
    net = theanets.Regressor(layers=layers, loss='mse')

    iterations, error = net.train(train=train,
                                  valid=valid,
                                  algo=algo, patience=10, max_updates=5000, learning_rate=0.02,
                                  min_improvement=0.001, momentum=0.9)
    result = []
    original = []
    for targets in test[1]:
        original.append(data.deNormalizeValue(targets))
    for training in test[0]:
        training = np.reshape(training, (1, len(training)))
        result.append(data.deNormalizeValue(net.predict(training)[0]))

    result = np.asarray(result)
    original = np.asarray(original)
    return original, result, error, net.num_params, iterations


algos = ['nag']
lags = [([1, 6],), ([1, 24], [168, 168 + 24]), ([1, 48], [168, 168 + 24]), ([1, 24], 48, 168), ([1, 24], [48, 72]),
        ([1, 48],)]
rawData = RawData('../../DataManipulation/files/load_weather.csv')
timeSeries = rawData.readAllValuesCSV(targetCol=2)

j = 1
futures = 120
for lag in lags:
    ts = copy.deepcopy(timeSeries)
    data = TrainingTimeSeries(ts, lags=lag, futures=futures)
    train = [data.getTrainingTrain(), data.getTrainingTarget()]
    valid = [data.getValidationTrain(), data.getValidationTarget()]
    test = [data.getTestTrain(), data.getTestTarget()]
    inLayer = theanets.layers.Input(data.trainLength, name='inputLayer')
    for algo in algos:
        for hiddenNeuron in range(70, 126, 5):
            hiddenLayer = theanets.layers.Feedforward(hiddenNeuron, inputs=inLayer.size, activation='sigmoid',
                                                      name='hiddenLayer')
            outLayer = theanets.layers.Feedforward(data.getOutputCount(), inputs=hiddenLayer.size, activation='linear')
            layers = [inLayer, hiddenLayer, outLayer]
            start_time = time.time()
            orig, result, error, n_params, iterations = test_RawData_read_csv_multiple_features(algo, layers, train,
                                                                                                valid, test)
            trainTime = time.time() - start_time
            rmse = utils.calculateRMSE(orig, result)
            mape = utils.calculateMAPE(orig, result)
            smape = utils.calculateSMAPE(orig, result)
            # Start the plotting
            title = 'algo:%s, lags:%s, hidden neurons:%s, testSample:%s TrainTime:%.2f sec' % (
                algo, lag, hiddenNeuron, len(result), trainTime)
            utils.plotFigures(orig, result, title, j)
            j += 1
            utils.benchmark(str(lag), inLayer.size, hiddenNeuron, outLayer.size, error[0]['err'], error[1]['err'],
                            n_params, rmse, mape, smape, trainTime, iterations)
