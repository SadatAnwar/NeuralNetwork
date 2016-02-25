import unittest

import itertools
import numpy as np
import theanets

from Logger import Logger
from RawData import RawData
# create logger
from TrainingTimeSeries import TrainingTimeSeries

logger = Logger(__name__)


class test_RawData(unittest.TestCase):
    def test_RawData_read_csv_single_Col(self):
        logger.debug('Testing Raw data CSV reader')
        rawData = RawData('../../DataManupulation/files/load.csv')
        timeSeries = rawData.readAllValuesCSV()
        self.assertEqual(len(timeSeries), 35032)

    def test_RawData_read_csv_multiple_features(self):
        logger.debug('Testing Raw data CSV reader with multiple features')
        rawData = RawData('../../DataManupulation/files/load_weather.csv')
        timeSeries = rawData.readAllValuesCSV(targetCol=2)
        self.assertEqual(len(timeSeries[1].getValue()), 4)
        logger.info('Stacking all the values into an ND array')
        tts = TrainingTimeSeries(timeSeries, lags=([1, 5], 24, 48), futures=5)
        net = theanets.Regressor(layers=[28, 10, 6], loss='mse')
        for target, train in itertools.izip(tts.getTargetData(), tts.getTrainingData()):
            for monitor in net.itertrain([np.reshape(train, (1,len(train))), target], algo='sgd', patience=10, max_updates=10000, learning_rate=0.05,min_improvement=0.001, momentum=0.5):
                pass
            print monitor


