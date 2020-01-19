import numpy as np
from .base import BaseScoreType


class MaxAbsoluteError(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='max absolute error', precision=3):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        return np.max(np.abs(y_true - y_pred))
