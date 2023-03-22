import numpy as np
from ml_from_scratch.linear_regression import LinearRegression

__author__ = "Kumar"
__copyright__ = "Kumar"
__license__ = "MIT"


def test_cost():
    x = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
    y = np.array([250, 300, 480, 430, 630, 730])
    w = 209
    b = 2.4
    cost = LinearRegression().cost(x, y, w, b)
    assert np.isclose(cost, 1736, atol=1)
