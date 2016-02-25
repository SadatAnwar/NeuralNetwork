from Logger import Logger
import unittest

import datetime
import dateutil.parser
from TimePoint import TimePoint

# create logger
logger = Logger(__name__)


class test_TimePoint(unittest.TestCase):
    def test_create_valid_timePoint(self):
        tp1 = TimePoint('2014-01-01T01:00', '3061.8725')
        tp2 = TimePoint('2014-01-01T02:00+1:00', '4000')
        self.assertEqual(3061.8725, tp1._value)
        self.assertEqual(datetime.datetime(2014, 1, 1, 01, 00, 00), tp1._dateTime)
        self.assertEqual(dateutil.parser.parse('2014-01-01T02:00+1:00'), tp2._dateTime)

    def test_compare_timePoint(self):
        tp1 = TimePoint('2014-01-01T01:00', '3061.8725')
        tp1a = TimePoint('2014-01-01T01:00', '4000.')
        tp2 = TimePoint('2014-01-01T02:00', '4000')
        self.assertGreater(tp2, tp1)
        self.assertLess(tp1, tp2)
        self.assertEqual(tp1, tp1a)

    def test_timePoint_dict(self):
        tp1 = TimePoint('2014-01-01T01:00', '3061.8725')
        tp2 = TimePoint('2014-01-01T02:00', '4000')
        tpDict = {tp1: 1, tp2: 2}
        self.assertEqual(tpDict[tp1], 1)
        self.assertEqual(tpDict[tp2], 2)

    def test_timePoint_params(self):
        tp1 = TimePoint('2014-01-01T01:00', '3061.8725')
        self.assertEqual(tp1.hour, 1)
        self.assertEqual(tp1.minute, 0)
        self.assertEqual(tp1.day, 1)
        self.assertEqual(tp1.month, 1)
        self.assertEqual(tp1.year, 2014)


