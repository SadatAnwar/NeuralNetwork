import climate
import theanets

from utils import *

climate.enable_default_logging()


def prepareTrainingDatasets(originalTimeSeries, numberOfLags):
    """
    Takes input as a TimeSeries which is a numpArray and number of lags
    Outputs a ndarray that can be used as the input for the training
    The target to be used is the original TS
    The OP should just the a single col array with Target values
    The Input should be multiple cols with lagged inputs, plus the features


    """
    # A list that will contain all the timeSeries
    listOfLaggedTimeSeries = []
    powerSeries = originalTimeSeries[:, len(originalTimeSeries[0]) - 1]

    # We insert all the other time series here
    for i in range(1, numberOfLags + 1):
        listOfLaggedTimeSeries.append(lagTimeSeries(powerSeries, i))

    # No we need to bind the columns as array (we skip the original one as it is the target)
    inputs = np.append(np.column_stack(tuple(listOfLaggedTimeSeries)),
                       tuple(originalTimeSeries[:, :len(originalTimeSeries[0]) - 1]), axis=1)
    inputs = inputs.reshape(len(inputs), len(inputs[0]))
    targets = powerSeries.reshape(len(originalTimeSeries), 1)
    return inputs, targets


# Impot the data
timeSeries = np.row_stack((
    importExtendedData('../../DataManupulation/files/month_may_weather.csv'),  # May
    importExtendedData('../../DataManupulation/files/month_jun_weather.csv'),  # Jun
    importExtendedData('../../DataManupulation/files/month_jul_weather.csv')  # July
))
# Normalize it
TS, minOfOld, maxOfOld = normalizeTimeSeries(timeSeries)
# Define network specifics
epoch = 18000
lags = 24
extraFeatures = len(TS[0]) - 1
hidden = 30

net = theanets.Regressor(layers=[(lags + extraFeatures), (hidden, 'tanh'), 1], loss='mse')
# Prepare the inputs and training targets
inputs, targets = prepareTrainingDatasets(TS, lags)
validationSet = np.row_stack((
    importExtendedData('../../DataManupulation/files/month_aug_weather.csv')
))
# Normalize it
validationTS, minOfOld, maxOfOld = normalizeTimeSeries(validationSet, minOfOld, maxOfOld)

# Prepare the inputs and training targets
validData, validTarget = prepareTrainingDatasets(TS, lags)
net.train([inputs, targets], algo='sgd',
          batch_size=50,
          patience=10,
          valid=[validData, validTarget],
          max_updates=epoch,
          learning_rate=0.01,
          min_improvement=0.001,
          momentum=0.8)
# save the network status
net.save('../../DataManupulation/files/trained/aug.gz')
# Try out the series
test = inputs[:, :]

# print(series)
