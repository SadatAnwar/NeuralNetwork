import climate
from utils import *

climate.enable_default_logging()


def prepareTrainingDatasets(originalTimeSeries, numberOfLags):
    """
    Takes input as a TimeSeries which is a numpArray and number of lags
    Outputs a ndarray that can be used as the input for the training
    The target to be used is the original TS

    """
    # A list that will contain all the timeSeries
    listOfTimeSeries = [originalTimeSeries]

    # First we insert the Original time series (with lag 0)

    # We insert all the other time series here
    for i in range(1, numberOfLags + 1):
        listOfTimeSeries.append(lagTimeSeries(originalTimeSeries, i))

    # No we need to bind the columns as array (we skip the original one as it is the target)
    inputs = np.column_stack(tuple(listOfTimeSeries[1:]))
    targets = originalTimeSeries
    return inputs, targets

