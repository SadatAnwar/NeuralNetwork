import numpy as np
import os
from Logger import Log


class TrainingDataSeries(Log):
    """
    The TrainingDataSeries, is a time series with all features, basically have time on the one axis and all other elements
    on the other. Objects of the time series class should have methods to make training and validation data from the
    data available inside the TS class
    
    """
    _inputData = None
    _targetCol = None
    _dateCol = None
    _timeCol = None
    _tempCol = None
    _solRadCol = None
    _numberCols = None
    _data = None

    def __init__(self, inputData, targetCol=None, dateCol=None, timeCol=None, tempCol=None, solRadCol=None, **kwargs):
        self._inputData = inputData
        self._targetCol = targetCol
        self._dateCol = dateCol
        self._timeCol = timeCol
        if self._validateInput():
            self._getInputAsArray()
        # Verify the load was successful
        if self._data is None:
            raise Exception('Error reading data')

    def _validateInput(self):
        if isinstance(self._inputData, str):
            if os.path.isfile(self._inputData):
                return True
            else:
                raise Exception('Input file does not exist')
        elif not hasattr(self._inputData, '__iter__'):
            # If the input is not string file name and not an array of sorts throw exception
            raise Exception('Invalid inputData, input is of type: (%s) expected (%s) or (%s)' % (
                type(self._inputData).__name__, np.ndarray.__name__, 'list of list'))
        else:
            return True

    def _getInputAsArray(self):
        if isinstance(self._inputData, np.ndarray):
            self._data = self._inputData.copy()
            return
        if isinstance(self._inputData, str):
            # TODO: Implement a csv reader
            pass
