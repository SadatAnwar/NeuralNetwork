import csv
import os
import numpy as np


def antiLog(values):
    return values ** 10


def logTimePointValues(values):
    return np.append(values[:-1], np.log10(values[-1]))


def linearEvaluationFunction(x):
    """
    A linear evaluation function that converts 2D data 1 point
    :param x: index of prediction
    :return:
    """
    length = len(x)
    step = float(1) / float(length)
    a = np.arange(1, 0, step * -1)[:length]
    s = 0
    for i in range(0, len(x)):
        s += x[i] * a[i]
    return s / sum(a)


def nonlinearEvaluationFunction(x):
    """
    A linear evaluation function that converts 2D data 1 point
    :param x: index of prediction
    :return:
    """
    a = np.ones(len(x))
    # the first 48 hrs are significant boyond that we have exponential decay
    if len(x) > 48:
        for i in range(48, len(x)):
            a[i] = np.exp(1) ** (-0.01 * (i - 48))
    s = 0
    for i in range(0, len(x)):
        s += x[i] * a[i]
    return s / sum(a)


def _findTimeOfDayCol(timeSeries):
    """
    Find the index of the column containing the TOD field, if there is no TOD field return None
    We deduce this by looking at the max and min of each col, the TOD is the only col that contains values from 0-23
    :param timeSeries:
    :return: int/None: 0 base index of the col containing the TOD
    """
    # Make sure the input is an np.array
    if type(timeSeries).__module__ != np.__name__:
        timeSeries = np.asarray(timeSeries)
    if timeSeries.shape[0] > 0:
        # The series should have more then one col
        if timeSeries.shape[1] > 1:
            for i in range(0, timeSeries.shape[1]):
                if timeSeries[:, i].min() == 0 and timeSeries[:, i].max() == 23:
                    return i
    return None


def normalizeTimeSeries(InputTimeSeries, minnTS=None, maxTS=None):
    if InputTimeSeries.shape[1] > 1:
        return _normalizeExtendedTimeSeries(InputTimeSeries, minnTS=None, maxTS=None)
    else:
        return _normalizeSimpleTimeSeries(InputTimeSeries, minnTS=None, maxTS=None)


def _normalizeExtendedTimeSeries(timeSeries, minnTS=None, maxTS=None):
    """
    Incoming series format
    [[time(hr(0-23)],[holiday(1/0)],[mon_fri(1/0)],[tue_thu(1/0)], [weekend(1/0)],[power]]
    [[holiday(1/0)],[mon_fri(1/0)],[tue_thu(1/0)], [weekend(1/0)],[power]]
    NOTE: The power is always present in the LAST col
    Need to only normalize the power column and the time
    For time we will divide by 23 to get all values between 0-1
    To normalize the power, we can simply send only one column to the _normalizeSimpleTimeSeries
    :param timeSeries:
    :param minnTS:
    :param maxTS:
    :return:
    """
    # First find out the col in which the TOD is present
    TODCol = _findTimeOfDayCol(timeSeries)

    powerTimeSeries = timeSeries[:, len(timeSeries[0]) - 1]

    normalizedPower, minnTS, maxTS = _normalizeSimpleTimeSeries(powerTimeSeries, minnTS, maxTS)
    if TODCol is not None:
        # If TOD is present, replace it with normalized version
        timeTimeSeries = _normalize_tod_timeSeries(timeSeries[:, TODCol])
        if timeSeries[:, TODCol].shape == timeTimeSeries.shape:
            timeSeries[:, TODCol] = timeTimeSeries
        else:
            timeSeries = np.delete(timeSeries, TODCol, 1)
            timeSeries = np.column_stack((timeTimeSeries, timeSeries))

    timeSeries[:, len(timeSeries[0]) - 1] = normalizedPower
    return timeSeries, minnTS, maxTS


def _normalizeSimpleTimeSeries(InputTimeSeries, minnTS=None, maxTS=None):
    """
    Normalize a time series that has only one col i.e the value of the time series (w/o other features)
    :param InputTimeSeries:
    :param minnTS:
    :param maxTS:
    :return: Normalized time series ndarray of size same as input TimeSeries
    """
    logTS = np.log10(InputTimeSeries)
    if maxTS is None:
        ts_max = logTS.max()
    else:
        ts_max = maxTS
    if minnTS is None:
        ts_min = logTS.min()
    else:
        ts_min = minnTS
    normalized = (logTS - ts_min) / (ts_max - ts_min)
    return normalized, ts_min, ts_max


def _normalize_tod_timeSeries(todTimeSeries):
    """

    :param todTimeSeries:
    :return: np.ndarray
    """
    return todTimeSeries / 23


def lagTimeSeries(timeseries, value):
    """
    To lag a time series, we basically need to append the first N values of the new time series, with the first value
    of the original time series, and clip the end, i.e so the new series is of the same length as the old one
    :param timeseries: The original TS
    :param value: the value you want to lag with
    :return: a lagged version of the original TS
    """
    assert len(timeseries) > 0
    # We convert the time series as an nd array of size LEN, 1
    laggedTimeSeries = np.reshape(np.copy(timeseries), (len(timeseries), 1))
    # We append the first element of the original TS "lag" times
    for i in range(0, value):
        laggedTimeSeries = np.insert(laggedTimeSeries, 0, timeseries[0])
    # Reshape the array, just to clean up things
    laggedTimeSeries = np.reshape(laggedTimeSeries, (len(laggedTimeSeries), 1))

    # Now we clip the end of the array so that the Lagged series has same number of points
    laggedTimeSeries = laggedTimeSeries[:len(timeseries)]
    return laggedTimeSeries


def pushTimeSeries(timeSeries, value):
    """
    To push a time series, we basically need to append the first N values of the new time series, with the first value
    of the original time series, and clip the end, i.e so the new series is of the same length as the old one
    :param timeSeries: The original TS
    :param value: the value you want to lag with
    :return: a lagged version of the original TS
    """
    assert len(timeSeries) > 0
    # We convert the time series as an nd array of size LEN, 1
    pushedTimeSeries = np.reshape(np.copy(timeSeries), (len(timeSeries), 1))
    # We append the first element of the original TS "lag" times
    for i in range(0, value):
        pushedTimeSeries = np.insert(pushedTimeSeries, len(pushedTimeSeries), timeSeries[len(timeSeries) - 1])
    # Reshape the array, just to clean up things
    pushedTimeSeries = np.reshape(pushedTimeSeries, (len(pushedTimeSeries), 1))

    # Now we clip the start of the array so that the pushed series has same number of points
    pushedTimeSeries = pushedTimeSeries[value:]
    return pushedTimeSeries


def deNormalization(normalizedTimeSeries, min, max):
    return np.asarray(map(antiLog, (normalizedTimeSeries * (max - min)) + min))


def calculateSSE(actual, forcasted):
    sse = 0
    assert len(actual) == len(forcasted)
    for i in range(0, len(actual)):
        sse += (actual[i] - forcasted[i]) ** 2
    return sse


def calculateRMSE(actual, forcasted):
    assert len(actual) == len(forcasted)
    mse = calculateSSE(actual, forcasted) / len(actual)
    return np.sqrt(mse)


def calculateCorrelation(actual, forcasted):
    """
    We calculate the correlation of all the predicted values with that of all the actual values at t
    So if we predict n values in the future, and we make m such predictions
    we find the correlation of all Actual m at prediction 0 to all Predicted m at prediction 0
    and the same for prediction 0...n
    :return: correlation coefficient
    """
    assert (len(actual) == len(forcasted))
    assert (len(actual[0]) == len(forcasted[0]))
    correlation = []
    for i in range(0, len(actual[0])):
        meanA = np.mean(actual[:, i])
        meanF = np.mean(forcasted[:, i])
        sdA = np.std(actual[:, i])
        sdF = np.std(forcasted[:, i])


def calculateMaxMinPerError(actual, forcasted):
    assert len(actual) == len(forcasted)
    pe = []
    for i in range(0, len(actual)):
        p = np.abs(forcasted[i] - actual[i]) / actual[i]
        p *= 100
        pe.append(p)
    return np.amax(pe, axis=0), np.amin(pe, axis=0)


def calculateSMAPE(actual, forrcasted):
    """
    There is a third version of SMAPE, which allows to measure the direction of the bias in the data by generating a positive and a negative error on line item level. Furthermore it is better protected against outliers and the bias effect mentioned in the previous paragraph than the two other formulas. The formula is:

    SMAPE = \frac{\sum_{t=1}^n \left|F_t-A_t\right|}{\sum_{t=1}^n (A_t+F_t)}

    :param actual:
    :param forcasted:
    :return:
    """
    assert (len(actual) == len(forcasted))
    smape = 0.0
    for i in range(0, len(actual)):
        smape += (abs(forcasted[i] - actual[i]) / (abs(actual[i]) + abs(forcasted[i])))
    return (smape / len(actual)) * 100


def calculateMAPE(actual, forcasted):
    assert len(actual) == len(forcasted)
    pe = 0
    for i in range(0, len(actual)):
        pe += np.abs(forcasted[i] - actual[i]) / actual[i]
    return (pe / len(actual)) * 100


def importData(fileName, delimiter=','):
    """ Impot data from a csv"""
    with open(fileName, 'r') as dest_f:
        data_iter = csv.reader(dest_f,
                               delimiter=delimiter,
                               quotechar='"')

        dataO = [float(data[5]) for data in data_iter]
    data_array = np.asarray(dataO).reshape(len(dataO), 1)
    return data_array


def importExtendedData(fileName, delimiter=','):
    """ Import an extended data set  from a csv
        The data set is supposed to have along with the time series also
        Holiday, dayOfWeekType, and timeOfDay
        The is to be arranged as
        [[time(hr(0-23)],[holiday(1/0)],[mon_fri(1/0)],[tue_thu(1/0)], [weekend(1/0)],[power]]
    """
    with open(fileName, 'r') as dest_f:
        data_iter = csv.reader(dest_f,
                               delimiter=delimiter,
                               quotechar='"')
        data = []

        def isHoliday(x):
            return 0 if x == 'FALSE' else 1

        def MO_AND_FR(x):
            return 1 if x == 'MO_AND_FR' else 0

        def TUE_TO_THU(x):
            return 1 if x == 'TUE_TO_THU' else 0

        def WEEKEND(x):
            return 1 if x == 'WEEKEND' else 0

        for row in data_iter:
            data.append([
                isHoliday(row[4]), MO_AND_FR(row[3]), TUE_TO_THU(row[3]),
                WEEKEND(row[3]), float(row[5])])
    assert len(data) > 0
    data_array = np.asarray(data).reshape(len(data), len(data[0]))
    return data_array


def printPerformance(dataName, actual, predicted, deNormFunc):
    mse = calculateRMSE(deNormFunc(actual), deNormFunc(predicted))
    print('MSE for %s is %s' % (dataName, mse))
    mape = calculateMAPE(deNormFunc(actual), deNormFunc(predicted))
    print('MAPE for %s is %s' % (dataName, mape))
    print('MSE for %s (Normalized) is %s' % (dataName, calculateRMSE(list(predicted), predicted)))
    return mape, mse


def benchmark(lags, inputDimensions, hidden, outputs, trainError, validationError, numParam, rmse, mape, smape,
              trainTime,
              iterations,
              flatFunction=nonlinearEvaluationFunction,
              fileName='../performance/neuralNetBenchmark1.csv'):
    """
    Log the benchmark to a file (so that it can be plotted for comparisons)
    :param algo:
    :param inputDimensions:
    :param lags:
    :param hidden:
    :param outputs:
    :param rmse:
    :param mape:
    :param fileName:
    :return:
    """
    # If the output is more than one, then we will have multiple mse, and mape's
    header = 'input lags; input Dimensions; hidden; outputs; net parameters; trainError; validationError;' + \
             ' RMSE; MAPE; SMAPE; train time; iterations \n'
    if not os.path.isfile(fileName):
        with open(fileName, 'w') as o:
            # Write the header
            o.write(header)
    else:
        num_lines = sum(1 for line in open(fileName))
        if num_lines == 0:
            with open(fileName, 'w') as o:
                # Write the header
                o.write(header)

    with open(fileName, 'a') as w:
        w.write('%s;%s;%s;%s;%s;%.3f;%.3f;%.3f;%.3f;%.3f;%.3f;%s\n' %
                (lags, inputDimensions, hidden, outputs, numParam,
                 trainError, validationError,
                 flatFunction(rmse), flatFunction(mape), flatFunction(smape), trainTime, iterations))
