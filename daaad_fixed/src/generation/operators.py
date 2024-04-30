from typing import Callable, List, Union
import torch
import numpy as np
import pandas as pd

from aaad.data_objects.data_types import DataObject

class Operator:
    def __init__(self, *args):
        self.args = args

    def is_differentiable(self):
        for arg in self.args:
            if isinstance(arg, Operator) and not arg.is_differentiable():
                return False
        return True
    
    def evaluate(self, data:pd.DataFrame):
        raise NotImplementedError()

class Arithmetic(Operator):
    def __init__(self, value):
        super().__init__(value)

class Boolean(Operator):
    def __init__(self, value):
        super().__init__(value)

class Inequality(Operator):
    def __init__(self, value):
        super().__init__(value)


class Constant(Operator):
    def __init__(self, value):
        super().__init__(value)

    def evaluate(self, data:pd.DataFrame):
        return self.args[0]


class Add(Arithmetic):
    def __init__(self, *args):
        super().__init__(args)

    def evaluate(self, data:pd.DataFrame):
        result = 0
        for arg in self.args:
            if isinstance(arg, DataObject):
                result = result + data[arg.name]
            elif isinstance(arg, Operator):
                result = result + arg.evaluate(data)
        return result

class Multiply(Arithmetic):
    def __init__(self, *args):
        super().__init__(args)

    def evaluate(self, data:pd.DataFrame):
        result = 1
        for arg in self.args:
            if isinstance(arg, DataObject):
                result = result * data[arg.name]
            elif isinstance(arg, Operator):
                result = result * arg.evaluate(data)
        return result

class Subtract(Arithmetic):
    def __init__(self, arg1:Operator, arg2:Operator):
        super().__init__(arg1, arg2)

    def evaluate(self, data:pd.DataFrame):
        value1 = data[self.args[0].name] if isinstance(self.args[0], DataObject) else self.args[0].evaluate(data)
        value2 = data[self.args[1].name] if isinstance(self.args[1], DataObject) else self.args[1].evaluate(data)
        return value1 - value2

class Divide(Arithmetic):
    def __init__(self, arg1:Operator, arg2:Operator):
        super().__init__(arg1, arg2)
    
    def evaluate(self, data:pd.DataFrame):
        value1 = data[self.args[0].name] if isinstance(self.args[0], DataObject) else self.args[0].evaluate(data)
        value2 = data[self.args[1].name] if isinstance(self.args[1], DataObject) else self.args[1].evaluate(data)
        return value1 / value2
        

class LessThan(Inequality):
    def __init__(self, arg1:Operator, arg2:Operator):
        super().__init__(arg1, arg2)

    def evaluate(self, data:pd.DataFrame):
        value1 = data[self.args[0].name] if isinstance(self.args[0], DataObject) else self.args[0].evaluate(data)
        value2 = data[self.args[1].name] if isinstance(self.args[1], DataObject) else self.args[1].evaluate(data)
        return value1 < value2

class LessOrEqual(Inequality):
    def __init__(self, arg1:Operator, arg2:Operator,):
        super().__init__(arg1, arg2)

    def evaluate(self, data:pd.DataFrame):
        value1 = data[self.args[0].name] if isinstance(self.args[0], DataObject) else self.args[0].evaluate(data)
        value2 = data[self.args[1].name] if isinstance(self.args[1], DataObject) else self.args[1].evaluate(data)
        return value1 <= value2

class GreaterThan(Inequality):
    def __init__(self, arg1:Operator, arg2:Operator,):
        super().__init__(arg1, arg2)

    def evaluate(self, data:pd.DataFrame):
        value1 = data[self.args[0].name] if isinstance(self.args[0], DataObject) else self.args[0].evaluate(data)
        value2 = data[self.args[1].name] if isinstance(self.args[1], DataObject) else self.args[1].evaluate(data)
        return value1 > value2

class GreaterOrEqual(Inequality):
    def __init__(self, arg1:Operator, arg2:Operator,):
        super().__init__(arg1, arg2)

    def evaluate(self, data:pd.DataFrame):
        value1 = data[self.args[0].name] if isinstance(self.args[0], DataObject) else self.args[0].evaluate(data)
        value2 = data[self.args[1].name] if isinstance(self.args[1], DataObject) else self.args[1].evaluate(data)
        return value1 >= value2

class Log(Arithmetic):
    def __init__(self, arg1:Operator, use_torch:bool=False):
        super().__init__(arg1)
        self.use_torch = use_torch

    def evaluate(self, data:pd.DataFrame):
        value = data[self.args[0].name] if isinstance(self.args[0], DataObject) else self.args[0].evaluate(data)
        if self.use_torch:
            return torch.math.log(value)
        else:
            return np.log(value)

class Exp(Arithmetic):
    def __init__(self, arg1:Operator, use_torch:bool=False):
        super().__init__(arg1)
        self.use_torch = use_torch

    def evaluate(self, data:pd.DataFrame):
        value = data[self.args[0].name] if isinstance(self.args[0], DataObject) else self.args[0].evaluate(data)
        if self.use_torch:
            return torch.math.exp(value)
        else:
            return np.exp(value)

class Pow(Arithmetic):
    def __init__(self, base:Operator, exponent:Operator):
        super().__init__(base, exponent)

    def evaluate(self, data:pd.DataFrame):
        base = data[self.args[0].name] if isinstance(self.args[0], DataObject) else self.args[0].evaluate(data)
        exponent = data[self.args[1].name] if isinstance(self.args[1], DataObject) else self.args[1].evaluate(data)
        return base**exponent

class Not(Boolean):
    def __init__(self, arg:Union[Inequality, Boolean]):
        super().__init__(arg)

    def evaluate(self, data:pd.DataFrame):
        value = data[self.args[0].name] if isinstance(self.args[0], DataObject) else self.args[0].evaluate(data)
        return ~value

class And(Boolean):
    def __init__(self, *args:List[Union[Inequality, Boolean]]):
        super().__init__(args)

    def evaluate(self, data:pd.DataFrame):
        value = data[self.args[0].name] if isinstance(self.args[0], DataObject) else self.args[0].evaluate(data)
        for arg in self.args[1:]:
            value = value & data[arg.name] if isinstance(arg, DataObject) else arg.evaluate(data)
        return value

class Or(Boolean):
    def __init__(self, *args:List[Union[Inequality, Boolean]]):
        super().__init__(args)

    def evaluate(self, data:pd.DataFrame):
        value = data[self.args[0].name] if isinstance(self.args[0], DataObject) else self.args[0].evaluate(data)
        for arg in self.args[1:]:
            value = value | data[arg.name] if isinstance(arg, DataObject) else arg.evaluate(data)
        return value

class XOr(Boolean):
    def __init__(self, *args:List[Union[Inequality, Boolean]]):
        super().__init__(args)

    def evaluate(self, data:pd.DataFrame):
        value = data[self.args[0].name] if isinstance(self.args[0], DataObject) else self.args[0].evaluate(data)
        for arg in self.args[1:]:
            value = value ^ data[arg.name] if isinstance(arg, DataObject) else arg.evaluate(data)
        return value



