from collections import Sequence

import numpy as np
import datetime
from Logger import Logger
from TimePoint import TimePoint

logger = Logger(__name__)


class TimeSeries(Sequence):
    _listTimePoints = np.ndarray
    _dateTimeHash = {}
    sum_values = 0
    deviation = 0
    _isMeanCalculated = False
    _isDeviationCalculated = False

    def __init__(self, timePoints=None, timePointManipulatorFunc=None):
        self._maskFunctions = []
        self.sum_values = 0
        self._timePointManipulatorFunc = timePointManipulatorFunc
        self._listTimePoints = None
        self._isMeanCalculated = False
        self._isDeviationCalculated = False
        self._dateTimeHash = {}
        if timePoints is not None:
            # Make sure the timePoints are an ndarray
            if not isinstance(timePoints, np.ndarray):
                timePoints = np.asarray(timePoints)
            # Remove any NaNs
            self._listTimePoints = timePoints[~np.isnan(np.row_stack(tp.getValue() for tp in timePoints)).any(axis=1)]
            self._hashDateTime()

    def _hashDateTime(self):
        """
        Hash the datetime values with the index of the dateTime to make the lookups a lot faster
        :return:
        """
        i = 0
        for tp in self._listTimePoints:
            self._dateTimeHash[tp.getDateTime()] = i
            i += 1

    def applyMaskOnValues(self, func):
        """
        This function will apply a function mask to the values
        Its a quick way for applying a numeric mask, the functions will only be applied to the values when they are
        requested
        :return:
        """
        self._maskFunctions.append(func)
        pass

    def compoundTimeSeries(self, otherTimeSeries):
        raise NotImplementedError()

    def __getitem__(self, item):
        if isinstance(item, int):
            # Get the tp
            if item < len(self._listTimePoints):
                tp = self._listTimePoints[item]
            else:
                return np.nan
        elif isinstance(item, datetime.datetime):
            if self._dateTimeHash.get(item, None) is not None:
                tp = self._listTimePoints[self._dateTimeHash[item]]
            else:
                return np.nan
        # Get the value for the TP
        val = tp[1]
        # apply the mask function to the value
        for mask in self._maskFunctions:
            val = mask(val)
        # return a time point instance with value = val and datetime 0 tp.getDatetime()
        return TimePoint(tp[0], val)

    def __len__(self):
        return len(self._listTimePoints)

    def __iter__(self):
        for tp in self._listTimePoints:
            val = tp[1]
            for mask in self._maskFunctions:
                val = mask(val)
            yield TimePoint(tp[0], val)

    def values(self):
        for tp in self._listTimePoints:
            val = tp[1]
            for mask in self._maskFunctions:
                val = mask(val)
            yield val

    def getValuesBetween(self, startIndex, endIndex):
        """
        Returns the values in the requested period, both included
        :param startIndex:
        :param endIndex:
        :return:
        """
        assert (type(startIndex) == type(endIndex))
        if isinstance(startIndex, datetime.datetime):
            startIndex = self._dateTimeHash.get(startIndex, None)
            endIndex = self._dateTimeHash.get(endIndex, None)
        if startIndex is None or endIndex is None:
            return np.asarray([np.nan])
        if startIndex < endIndex:
            if endIndex + 1 > len(self._listTimePoints):
                return np.asarray([np.nan])
            vals = self._listTimePoints[startIndex:endIndex + 1, 1:]
        else:
            if startIndex + 1 > len(self._listTimePoints)-1:
                return np.asarray([np.nan])
            vals = self._listTimePoints[endIndex:startIndex + 1, 1:]
        for mask in self._maskFunctions:
            getVals = lambda x : mask(x[0])
            vals = np.apply_along_axis(getVals, axis=1, arr=vals)
        return vals

    def __delitem__(self, ii):
        del self._listTimePoints[ii]

    def __setitem__(self, ii, val):
        self._listTimePoints[ii] = val
        return self._listTimePoints[ii]

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return """<TimeSeries %s>""" % self._listTimePoints

    def insert(self, ii, val):
        raise NotImplemented
        # if isinstance(ii, datetime.datetime):
        #    if ii in self._dateTimeHash.keys():
        #        self._listTimePoints[self._dateTimeHash[ii]] = val
        #    else:
        #        self._listTimePoints.append(val)
        #        self._dateTimeHash[ii] = len(self._listTimePoints)

    def append(self, timePoint):
        if np.isnan(timePoint.getValue()).any():
            return
        if self._dateTimeHash.get(timePoint.getDateTime(), None) is not None:
            raise DuplicateTimePointException("The dateTime already exists, cannot have duplicates")
        else:
            if self._listTimePoints is None:
                self._listTimePoints = np.asarray([[timePoint.getDateTime(), timePoint.getValue()]])
            else:
                self._listTimePoints = np.vstack([self._listTimePoints,
                                                  np.asarray([[timePoint.getDateTime(), timePoint.getValue()]])])
            # To get sum, we must get the value as it is after any filter functions
            self.sum_values += timePoint.getValue()
            self._dateTimeHash[timePoint.getDateTime()] = len(self._listTimePoints) - 1

    def getIndex(self, dateTimePoint):
        if dateTimePoint in self._dateTimeHash.keys():
            return self._dateTimeHash[dateTimePoint]
        else:
            return None


class InvalidTimeSeriesException(Exception):
    pass


class DuplicateTimePointException(Exception):
    pass
