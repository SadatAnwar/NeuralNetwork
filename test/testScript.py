import copy
import numpy as np
import theanets
import climate
import time

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
rawData = RawData('../../DataManupulation/files/load_weather.csv')
timeSeries = rawData.readAllValuesCSV(targetCol=2)
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as pl

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
            pl.style.use('bmh')
            fig_x_scale = range(0, futures + 1, 6)
            title = 'algo:%s, lags:%s, hidden neurons:%s, \n testSample:%s TrainTime:%.2f sec' % (
                algo, lag, hiddenNeuron, len(result), trainTime)
            x_lable = 'forecast as t+x'
            # Figure
            pl.rcParams['figure.dpi'] = 100
            fig_size = pl.rcParams["figure.figsize"]
            fig_size[0] = 15
            fig_size[1] = 8
            pl.rcParams["figure.figsize"] = fig_size
            ax1 = host_subplot(111, axes_class=AA.Axes)
            pl.subplots_adjust(right=0.85, left=0.09)
            ncolors = pl.rcParams['axes.prop_cycle']
            b = ncolors._left[0]['color']
            r = ncolors._left[1]['color']
            g = ncolors._left[2]['color']
            ax1.plot(mape, '-', color=b)
            ax1.set_xlabel(x_lable)
            ax1.set_xticks(fig_x_scale)
            # Make the y-axis label and tick labels match the line color.
            ax1.set_ylabel('MAPE', color=b)
            ax2 = ax1.twinx()
            ax2.plot(smape, '-', color=r)
            ax2.set_ylabel('SMAPE', color=r)
            ax3 = ax1.twinx()
            offset = 60
            new_fixed_axis = ax3.get_grid_helper().new_fixed_axis
            ax3.axis["right"] = new_fixed_axis(loc="right",
                                               axes=ax3,
                                               offset=(offset, 0))
            ax3.axis["right"].toggle(all=True)
            ax3.plot(rmse, '-', color=g)
            ax3.set_ylabel('RMSE', color=g)

            pl.title(title)
            pl.legend(['MAPE', 'SMAPE', 'RMSE'])
            pl.savefig('../results/mac/profile%s.pdf' % j)
            pl.close()
            j += 1
            utils.benchmark(str(lag), inLayer.size, hiddenNeuron, outLayer.size, error[0]['err'], error[1]['err'],
                            n_params, rmse, mape, smape, trainTime, iterations)
