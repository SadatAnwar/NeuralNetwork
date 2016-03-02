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
    assert len(train) == len(test) == len(valid)
    networks = []
    for i in range(len(train)):
        networks.append(theanets.Regressor(layers=layers, loss='mse'))
        for j, (tm, vm) in enumerate(networks[i].itertrain(train=train[i],
                                                           valid=valid[i],
                                                           algo=algo, patience=10, max_updates=5000, learning_rate=0.02,
                                                           min_improvement=0.001, momentum=0.9)):
            pass
    result = []
    original = []
    for i in range(len(test[0][0])):
        orig = []
        res = []
        training = np.reshape(test[0][0][i], (1, len(test[0][0][i])))
        for j in range(len(test)):
            orig.append(data.deNormalizeValue(test[j][1][i]))
            res.append(data.deNormalizeValue(networks[j].predict(training)[0]))
        original.append(np.asarray(orig).flatten())
        result.append(np.asarray(res).flatten())

    result = np.asarray(result)
    original = np.asarray(original)
    return original, result, [tm, vm], net.num_params, i


algos = ['nag']
lags = [([1, 6],), ([1, 24],), ([1, 48],),
        ([1, 24], [48, 72]),
        ([1, 24], [168, 168 + 6]),
        ([1, 24], [168, 168 + 24]),
        ([1, 48], [168, 168 + 12]),
        ([1, 48], [168, 168 + 24]),
        ([1, 24], 48, 168)]
rawData = RawData('../../DataManipulation/files/load_weather_jan.csv')
timeSeries = rawData.readAllValuesCSV(targetCol=2)

k = 1
futures = 119
outputCount = 24
modelCount = (futures + 1) / outputCount

for lag in lags:
    ts = copy.deepcopy(timeSeries)
    data = []
    train = []
    valid = []
    test = []
    for i in range(modelCount):
        j = i * 24
        data.append(TrainingTimeSeries(ts, lags=lag, futures=(j, j + 24)))
        train.append([data[i].getTrainingTrain(), data[i].getTrainingTarget()[:, j:(j + 24)]])
        valid.append([data[i].getValidationTrain(), data[i].getValidationTarget()[:, j:(j + 24)]])
        test.append([data[i].getTestTrain(), data[i].getTestTarget()[:, j:(j + 24)]])
    inLayer = theanets.layers.Input(data[0].trainLength, name='inputLayer')
    for algo in algos:
        for hiddenNeuron in range(70, 126, 5):
            hiddenLayer = theanets.layers.Feedforward(hiddenNeuron, inputs=inLayer.size, activation='sigmoid',
                                                      name='hiddenLayer')
            outLayer = theanets.layers.Feedforward(outputCount, inputs=hiddenLayer.size, activation='linear')
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
            utils.plotFigures(orig, result, title, k, locationToSaveImages='../results/multiple_model/')
            k += 1
            utils.benchmark(str(lag).replace(',', ':'), inLayer.size, hiddenNeuron, outLayer.size, error[1][0]['err'],
                            error[1][1]['err'],
                            n_params, rmse, mape, smape, trainTime, iterations,
                            fileName='../performance/neuralNetBenchmark_m_m.csv')
