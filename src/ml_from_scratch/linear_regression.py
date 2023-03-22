# Python class and a method called cost

import numpy as np


class LinearRegression:
    def cost(self, x, y, w, b):
        m = len(x)
        total_cost = 0
        for i in range(m):
            fwb = (w * x[i] + b)
            total_cost += (y[i] - fwb) ** 2
        total_cost = total_cost / (2 * m)
        return total_cost
