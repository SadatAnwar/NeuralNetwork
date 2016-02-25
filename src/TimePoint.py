import datetime
import numpy as np
import dateutil.parser

from Logger import Logger

logger = Logger(__name__)


class TimePoint:
    _dateTime = None
    _value = None
    minute = None
    hour = None
    weekDay = None
    month = None
    year = None
    date = None

    def __init__(self, dateTime, value):
        """
        Construct a TimePoint object from a dateTime string and a value string
        :param dateTime: String of ISO8601 offset datetime
        :param value: Value at the given dateTime can be a list of values, if the values is a list, the last value
                        i.e. value at value[len(value)-1] is the target value of a training TimePoint
        """
        if isinstance(dateTime, str):
            self._dateTime = dateutil.parser.parse(dateTime)
        elif isinstance(dateTime, datetime.datetime):
            self._dateTime = dateTime
        if isinstance(value, list) or isinstance(value, tuple):
            self._value = np.asarray([self._getNumericValue(x) for x in value])
        elif isinstance(value, np.ndarray):
            self._value = value

        else:
            self._value = np.asarray([self._getNumericValue(value)])
        self.minute = self._dateTime.minute
        self.hour = self._dateTime.hour
        self.weekDay = self._dateTime.weekday()
        self.month = self._dateTime.month
        self.year = self._dateTime.year
        self.date = self._dateTime.date()

    def getValue(self, pos=None):
        if pos is None:
            return np.asarray(self._value)
        else:
            return self._value[pos]

    def getDateTime(self):
        return self._dateTime

    def getTimePointFeatures(self):
        """
        Extract features from a datetime object present in a timePoint
        :return: array containing binary features for dateTime objects
        """

        def quarterToFeature():
            quarter = np.asarray([[0] * 4])
            if self.month in [12, 1, 2]:
                quarter[:, 0] = 1
            elif self.month in [3, 4, 5]:
                quarter[:, 1] = 1
            elif self.month in [6, 7, 8]:
                quarter[:, 2] = 1
            else:
                quarter[:, 3] = 1
            return quarter

        # Mon=0 tue=1 wed=2 thu=3 sun=6
        def dayToFeature(day):
            feature = np.asarray([[0] * 3])
            if day == 0 or day == 4:
                # Day is Mon or Fri
                feature[:, 0] = 1
            elif 0 < day < 4:
                # Day is Tue, Wed, Thu
                feature[:, 1] = 1
            else:
                # Weekend
                feature[:, 2] = 1
            return feature

        # Can split time of day as night and 4 halves
        def timeToFeature(time):
            feature = np.asarray([[0] * 17])
            if time >= 22 or time <= 5:
                feature[:, 0] = 1
            else:
                feature[:, time - 5] = 1
            return feature

        return np.concatenate((timeToFeature(self.hour).flatten(),
                               dayToFeature(self.weekDay).flatten(),
                               quarterToFeature().flatten()))

    def _getNumericValue(self, val):
        if isinstance(val, str):
            val = val.strip()
        try:
            return float(val)
        except:
            if val.lower() in ['true', 't', 'y', 'yes']:
                return 1
            elif val.lower() in ['false', 'f', 'no', 'n']:
                return 0
            else:
                return np.nan

    def applyFuncOnValues(self, func):
        """
        Apply func to the values of timePoint
        :param func: function that takes only one parameter i.e the ndarray of values
        :return:
        """
        self._value = func(self._value)

    def __hash__(self):
        return self.date.__hash__()

    def __cmp__(self, other):
        if isinstance(other, TimePoint):
            return self._dateTime.__cmp__(other._dateTime)

    def __eq__(self, other):
        if isinstance(other, TimePoint):
            return other._dateTime == self._dateTime
        else:
            return False

    def __gt__(self, other):
        if isinstance(other, TimePoint):
            return other._dateTime < self._dateTime
        else:
            return False

    def __lt__(self, other):
        if isinstance(other, TimePoint):
            return other._dateTime > self._dateTime
        else:
            return False

    def __ge__(self, other):
        if isinstance(other, TimePoint):
            return other._dateTime <= self._dateTime
        else:
            return False

    def __le__(self, other):
        if isinstance(other, TimePoint):
            return other._dateTime >= self._dateTime
        else:
            return False

    def isnan(self):
        return np.isnan(self._value).any()
