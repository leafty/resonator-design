import torch
import numpy as np

from dataset.data_set import DataSet
from dataset.data_sampler import QuantileSampler, Sampler

class MethodDataGenerator:
    def __init__(self, mode:str, dataset:DataSet, feature_ranges:dict={}, **kwargs):
        self.mode = mode
        self.dataset = dataset
        self.feature_ranges = {k: v for k, v in feature_ranges.items()}
        self.fixed_parameters = {}
        self.tunable_parameters = {}

        if self.mode == 'x':
            for k, f in self.dataset.x.items():
                if k in self.feature_ranges:
                    if isinstance(self.feature_ranges[k], Sampler):
                        self.tunable_parameters[k] = self.feature_ranges[k]
                    else:
                        self.fixed_parameters[k] = self.feature_ranges[k]
                else:
                    self.tunable_parameters[k] = QuantileSampler(f.data)
        else:
            for k, f in self.dataset.y.items():
                if k in self.feature_ranges:
                    if isinstance(self.feature_ranges[k], Sampler):
                        self.tunable_parameters[k] = self.feature_ranges[k]
                    else:
                        self.fixed_parameters[k] = self.feature_ranges[k]
                else:
                    self.tunable_parameters[k] = QuantileSampler(f.data)

    def generate(*args, **kwargs) -> dict:
        raise NotImplementedError()

class RandomDataGenerator(MethodDataGenerator):
    def __init__(self, mode:str, dataset:DataSet, feature_ranges:dict={}, **kwargs):
        super(RandomDataGenerator, self).__init__(mode, dataset, feature_ranges, **kwargs)

    def generate(self, k:int, **kwargs):
        params = np.random.uniform(size=(k, len(self.tunable_parameters)))
        params = {**{n: p.inverse_transform(params[:, i:i+1]) for i, (n, p) in enumerate(self.tunable_parameters.items())}, 
                  **{n: np.array([v]*k).reshape((-1, 1)) for n, v in self.fixed_parameters.items()}}
        return params

        
class SobolDataGenerator(MethodDataGenerator):
    def __init__(self, mode:str, dataset:DataSet, feature_ranges:dict={}, stateful:bool=True, **kwargs):
        super(SobolDataGenerator, self).__init__(mode, dataset, feature_ranges, **kwargs)
        self.engine = torch.quasirandom.SobolEngine(dimension=len(self.tunable_parameters))

    def generate(self, k:int, **kwargs):
        params = self.engine.draw(k).numpy()
        params = {**{n: p.inverse_transform(params[:, i:i+1]) for i, (n, p) in enumerate(self.tunable_parameters.items())},
                  **{n: np.array([v]*k).reshape((-1, 1)) for n, v in self.fixed_parameters.items()}}
        return params


class DataGenerator:
    def __init__(self, method:str, **kwargs):
        self.method = method
        self.inner_generator = None

    def generate(self, **kwargs):
        return self.inner_generator.generate(**kwargs)

class XDataGenerator(DataGenerator):
    def __init__(self, method:str='rnd', **kwargs):
        super().__init__(method, **kwargs)
        if method in ['random', 'uniform', 'rnd']:
            self.inner_generator = RandomDataGenerator(mode='x', **kwargs)
        elif method in ['sobol', 'sobol sequence']:
            self.inner_generator = SobolDataGenerator(mode='x', **kwargs)

class YDataGenerator(DataGenerator):
    def __init__(self, method:str='rnd', **kwargs):
        super().__init__(method, **kwargs)
        if method in ['random', 'uniform', 'rnd']:
            self.inner_generator = RandomDataGenerator(mode='y', **kwargs)
        elif method in ['sobol', 'sobol sequence']:
            self.inner_generator = SobolDataGenerator(mode='y', **kwargs)

