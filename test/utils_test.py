import unittest

import numpy as np

import utils as utils


class utilsTest(unittest.TestCase):
    def test_antiLog_int(self):
        utils.getTimeSeriesFromServer()

    def test_antiLog_ndarray(self):
        a = np.array([[1 ** 0.01, 1 ** 0.01], [1 ** 0.001, 1 ** 0.001]])
        expectedResult = [
            [utils.antiLog(1 ** 0.01), utils.antiLog(1 ** 0.01)],
            [utils.antiLog(1 ** 0.001), utils.antiLog(1 ** 0.001)]
        ]
        self.assertEqual(utils.antiLog(a), expectedResult)

    def test_antiLog_list(self):
        a = [1 ** 0.0001, 1 ** 0.001, 1 ** 0.01]
        expectedResult = [utils.antiLog(1 ** 0.0001), utils.antiLog(1 ** 0.001), utils.antiLog(1 ** 0.01)]
        self.assertEqual(utils.antiLog(a), expectedResult)
