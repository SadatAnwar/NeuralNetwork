import os
import re

import climate
import numpy as np

import utils
from Logger import Logger
from TimePoint import TimePoint
from TimeSeries import TimeSeries

logger = Logger(__name__)


class RawData:
    def __init__(self, rawInput):
        self._rawInput = rawInput
        self._colNameMap = None
        self._headerPresent = None
        self._timeSeries = TimeSeries()

    def readAllValuesCSV(self, header=True, skip_cols=None, targetCol=None):
        """

        :param header:
        :param skip_cols: tuple
        :param targetCol:
        :return:
        """
        self._headerPresent = header
        if not os.path.isfile(self._rawInput):
            logger.error('Input file %s does not exist' % self._rawInput)
            raise IOError('Input file not present')
        with open(self._rawInput) as inputFile:
            climate.logging.info('Reading data from file %s' % self._rawInput)
            lineNumber = 0
            for line in inputFile:
                if '"' in line:
                    line = self._removeQuotes(line)
                columns = line.split(',')
                if lineNumber == 0 and self._headerPresent:
                    # Check the header
                    # Just look for alphabets
                    matches = []
                    for c in columns:
                        matches += re.findall('[A-z]+', c)
                    if len(matches) >= len(columns):
                        self._extractColNames(line)
                        lineNumber += 1
                        continue
                dateTime = columns[0]
                # normally the last val is the target
                target = columns[-1]
                if targetCol is not None:
                    # if a col is specified as target, get it as the target
                    target = columns[targetCol]
                values = []
                for i in range(1, len(columns)):
                    if i == targetCol:
                        continue
                    if skip_cols is not None and i in skip_cols:
                        continue
                    values.append(columns[i])
                values.append(target)
                # now we add traget to this now list of values
                columns.append(target)
                self._timeSeries.append(TimePoint(dateTime, values))
                lineNumber += 1
        climate.logging.info('%s lines read into TimeSeries' % lineNumber)
        return self._timeSeries

    def _removeQuotes(self, line):
        lineParts = line.split('"')
        newLine = ''
        for i in range(0, len(lineParts)):
            if i % 2 == 1:
                lineParts[i] = lineParts[i].replace(',', '.')
            newLine = newLine + lineParts[i]
        return newLine

    def _extractColNames(self, line):
        """
        Extract the names of the col from the header line.
        The input is the header line. If its not specified, the code automatically tries to find it
        :param line:
        :return:
        """
        colNames = line.split(',')
        colNameIndexMap = {}
        for i in range(0, len(colNames)):
            colNameIndexMap[colNames[i]] = i
        self._numCols = len(colNames)
        self._colNameMap = colNameIndexMap.copy()

    def _getTimeSeries(self, listTimeSeries):
        TimeSeriesList = []
        for timeSeries in listTimeSeries:
            TimeSeriesList.append(TimeSeries(timeSeries))
        return TimeSeriesList
