import datetime
import time

import climate
import numpy as np


class TrainingTimeSeries:
    """
    A training time series internally consists of a numeric ndarray, and supports operations specific to a
    training data set

    """
    _data = np.ndarray

    def __init__(self, timeSeries, lags=0, futures=0, normalization=True, trainSize=70, validationSize=15):
        """
        The trainingTimeSeries can be created using a TimeSeries and optionally a lags that indicates the number of lags
        for the training data, and a validationSplit to indicate the amount of data to be used as validation
        eg: TrainingTimeSeries(timeSeries, lags=24, validationSplit=0.8)
        """
        self._isMeanCalculated = False
        self._isMeanCalculated = False
        self._lags = lags
        self._futures = futures
        self._norm = normalization
        if normalization:
            self._timeSeries = self._normalize(timeSeries)
        else:
            self._timeSeries = timeSeries
        self._trainingData = None
        self._targetData = None
        self._data = None
        climate.logging.info('Creating training time series with %s lags and %s futures' % (lags, futures))
        start = time.time()
        for tp in self._timeSeries:
            # Convert the dateTime to features
            features = self._timeSeries.getDatetimeFeatures(tp.getDateTime())
            # Get the lagged targets
            laggedTargets = self._getLagTargets(self._timeSeries, tp, self._lags)
            if np.isnan(laggedTargets).any():
                continue
            futureFeatures, futureTargets = self._getFutures(tp.getDateTime(), futures)
            if np.isnan(futureFeatures).any() or np.isnan(futureTargets).any():
                continue
            futureVals = np.concatenate((futureFeatures, futureTargets))
            newData = np.concatenate((features, laggedTargets, futureVals))
            if self._data is None:
                self._data = np.asarray(np.column_stack(newData))
            else:
                if len(self._data[0]) != len(newData):
                    continue
                self._data = np.append(self._data, np.column_stack(newData), axis=0)
        endTime = time.time() - start
        climate.logging.info('Training data completed in %.2f sec' % endTime)
        climate.logging.info('Average time per row %.2f sec' % (endTime / len(self._timeSeries)))
        np.random.shuffle(self._data)
        dataTrain = int(float(trainSize) / 100 * len(self._data))
        dataValidation = int(float(validationSize) / 100 * len(self._data))
        self._trainingData = self._data[:dataTrain, :]
        climate.logging.info('Training data size %s' % len(self._trainingData))
        self._validationData = self._data[dataTrain:dataTrain + dataValidation, :]
        climate.logging.info('Validation data size %s' % len(self._validationData))
        self._testData = self._data[dataTrain + dataValidation:, :]
        climate.logging.info('Test data size %s' % len(self._testData))
        self.trainLength = len(self._data[0]) - self.targetLength

    def getTrainingTrain(self):
        return self._trainingData[:, :self.trainLength]

    def getTrainingTarget(self):
        return self._trainingData[:, self.trainLength:]

    def getValidationTrain(self):
        return self._validationData[:, :self.trainLength]

    def getValidationTarget(self):
        return self._validationData[:, self.trainLength:]

    def getTestTrain(self):
        return self._testData[:, :self.trainLength]

    def getTestTarget(self):
        return self._testData[:, self.trainLength:]

    def getOutputCount(self):
        """
       returns the count of the number of neurons in the OP layer, basically futures + 1
        :return:
        """
        return self._futures + 1

    def _getLaggedExtraFeatures(self, tp, lags=5):
        laggedTimes = [tp.getDateTime() - datetime.timedelta(hours=x) for x in range(0, lags)]
        laggedTimePoints = [self._timeSeries[laggedTimes[x]] for x in range(0, len(laggedTimes))]
        t = None
        for point in laggedTimePoints:
            if point is np.nan:
                return np.nan
            if t is None:
                t = np.asarray(point.getValue()[:-1])
            else:
                t = np.append(t, tp.getValue()[:-1])
        return t

    def _getLagTargets(self, timeSeries, timePoint, lags):
        currentTime = timePoint.getDateTime()
        if not isinstance(lags, tuple):
            raise InvalidLagsError('%s is not a valid type of lag' % lags)
        else:
            # Make sure we have values begin requested
            lastLag = lags[-1]
            if isinstance(lastLag, list):
                if timeSeries[currentTime - datetime.timedelta(hours=lastLag[1])] is np.nan:
                    return np.nan
            elif isinstance(lastLag, int):
                if timeSeries[currentTime - datetime.timedelta(hours=lastLag)] is np.nan:
                    return np.nan
            else:
                raise InvalidLagsError('%s is not a valid type of lag' % lastLag)

        targets = np.asarray([])
        for period in lags:
            if isinstance(period, list):
                # Check the limit of this range, if the last (max lag) is not nan, we can get valid data from the lags
                # if not, we cant use it, so we return nan.
                if timeSeries[currentTime - datetime.timedelta(hours=period[1] + 1)] is np.nan:
                    return np.nan
                # laggedTimes = [currentTime - datetime.timedelta(hours=x) for x in range(period[0], (period[1] + 1))]
                vals = timeSeries.getValuesBetween(currentTime - datetime.timedelta(hours=period[0]),
                                                   currentTime - datetime.timedelta(hours=(period[1])))
                if np.isnan(vals).any():
                    return np.nan
                targets = np.append(targets, vals[:, -1])
            else:
                laggedTarget = timeSeries[currentTime - datetime.timedelta(hours=period)]
                if laggedTarget is not np.nan:
                    lt = np.asarray([laggedTarget.getValue(-1)])
                else:
                    # if anything is nan, return nan, we cant work with this data
                    return np.nan
                targets = np.append(targets, lt)
        return np.asarray(targets)

    def _getFutures(self, currentTimePoint, numberOfFutures):
        """
        This function gets future values for the target at each time instant
        if a future is not calculable it will replace it with a nan which has to be later omitted by the train and target
         series
        :param numberOfFutures: int for the amount of future values desired (NOTE: futures are consecutive )
        :return:
        """
        index = self._timeSeries.getIndex(currentTimePoint)
        if isinstance(numberOfFutures, int):
            vals = self._timeSeries.getValuesBetween(index, index + numberOfFutures)
            if np.isnan(vals).any():
                futuresTargets = np.asarray([np.nan * numberOfFutures])
                futuresFeatures = np.asarray([np.nan * numberOfFutures])
            else:
                futuresTargets = vals[:, -1]
                futuresFeatures = vals[:min(48, len(vals)), :-1]
                self.targetLength = len(futuresTargets.flatten())
            return futuresFeatures.flatten(), futuresTargets.flatten()
        if isinstance(numberOfFutures, tuple) or isinstance(numberOfFutures, list):
            vals = self._timeSeries.getValuesBetween(index + numberOfFutures[0],
                                                     index + numberOfFutures[0] + numberOfFutures[1])
            if np.isnan(vals).any():
                futuresTargets = np.asarray([np.nan * (numberOfFutures[0] - numberOfFutures[1])])
                futuresFeatures = np.asarray([np.nan * (numberOfFutures[0] - numberOfFutures[1])])
            else:
                futuresTargets = vals[:, -1]
                futuresFeatures = vals[:min(48, len(vals)), :-1]
                self.targetLength = len(futuresTargets.flatten())
            return futuresFeatures.flatten(), futuresTargets.flatten()

    def _normalize(self, timeSeries):
        climate.logging.info('Normalizing TimeSeries...')
        start = time.time()
        self._getMeanOfTimeSeries(timeSeries)
        self._getDeviationTimeSeries(timeSeries)
        self.normalizeWithMeanSD(timeSeries)
        climate.logging.info('Normalization complete in %.2f seconds' % (time.time() - start))
        return timeSeries

    def _getMeanOfTimeSeries(self, timeSeries):
        self.mean = timeSeries.sum_values / len(timeSeries)
        self.mean[0] = 0
        self._isMeanCalculated = True

    def _getDeviationTimeSeries(self, timeSeries):
        self.deviation = np.std(timeSeries.getValuesBetween(0, len(timeSeries) - 1))
        self.deviation[0] = 1
        self._isDeviationCalculated = True

    def normalizeWithMeanSD(self, timeSeries):
        if not self._isMeanCalculated:
            self._getMeanOfTimeSeries(timeSeries)
        if not self._isDeviationCalculated:
            self._getDeviationTimeSeries(timeSeries)
        timeSeries.applyMaskOnValues(lambda x: np.nan_to_num((x - self.mean) / self.deviation))

    def deNormalizeValue(self, x):
        if self._norm:
            return (x * self.deviation[-1]) + self.mean[-1]
        else:
            return x


class InvalidLagsError(Exception):
    pass
