from typing import Callable
from aaad.generation.operators import Arithmetic, Inequality, Operator
import torch
import numpy as np
import pandas as pd

from aaad.data_objects.data_types import DataObject

class Reducer:
    def __init__(self, operator:Operator):
        self.operator = operator

    def is_differentiable(self):
        return self.operator.is_differentiable()
    
    def evaluate(self, data:pd.DataFrame):
        raise NotImplementedError()

class Sum(Reducer):
    def __init__(self, operator:Arithmetic):
        super().__init__(operator)

    def evaluate(self, data: pd.DataFrame):
        return self.operator.evaluate(data).sum(axis=0)

class Mean(Reducer):
    def __init__(self, operator:Arithmetic):
        super().__init__(operator)

    def evaluate(self, data: pd.DataFrame):
        return self.operator.evaluate(data).mean(axis=0)

class Std(Reducer):
    def __init__(self, operator:Arithmetic):
        super().__init__(operator)

    def evaluate(self, data: pd.DataFrame):
        return self.operator.evaluate(data).std(axis=0)

class Var(Reducer):
    def __init__(self, operator:Arithmetic):
        super().__init__(operator)

    def evaluate(self, data: pd.DataFrame):
        return self.operator.evaluate(data).var(axis=0)

class All(Reducer):
    def __init__(self, operator:Inequality, use_torch:bool=False):
        super().__init__(operator)
        self.use_torch = use_torch

    def is_differentiable(self):
        return False

    def evaluate(self, data: pd.DataFrame):
        return self.operator.evaluate(data).all(axis=0)

class Any(Reducer):
    def __init__(self, operator:Inequality, use_torch:bool=False):
        super().__init__(operator)
        self.use_torch = use_torch

    def is_differentiable(self):
        return False

    def evaluate(self, data: pd.DataFrame):
        return self.operator.evaluate(data).any(axis=0)
