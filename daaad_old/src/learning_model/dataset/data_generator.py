import torch
import numpy as np

from sklearn.preprocessing import QuantileTransformer

from src.learning_model.dataset.data_set import DataSet

MAX_QUANTILES = 1000

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

# class BayesOptDataGenerator(MethodDataGenerator):
#     def __init__(self, mode:str, model:CondAEModel, dataset:DataSet, feature_ranges:dict, stateful:bool=True, **kwargs):
#         super(BayesOptDataGenerator, self).__init__(mode, dataset, feature_ranges, **kwargs)
#         self.model = model
#         self.stateful = stateful

#         if self.mode == 'x':
#             self.BO = BayesianOptimization(self.inner_loop_x, {k: (0, 1) for k in self.tunable_parameters.keys()}, verbose=0)
#         else:
#             self.BO = BayesianOptimization(self.inner_loop_y, {k: (0, 1) for k in self.tunable_parameters.keys()}, verbose=0)
        
#         #self.__register_dataset(BO)

#     def __register_dataset(self, BO:BayesianOptimization):
#         ds = self.dataset.get_batch(-1, transform=False)
#         ds = {**ds[0], **ds[1]}
#         for i in range(len(ds[list(ds.keys())[0]])):
#             params = {}
#             for k, v in ds.items():
#                 if k in self.tunable_parameters:
#                     params[k] = self.tunable_parameters[k].transform(v[i])
#             BO.register({k: v.transform(ds[k][i]) for k, v in self.tunable_parameters.items()}, target=0)

#     def inner_loop_x(self, **kwargs) -> float:
#         inputs = {**{k: self.tunable_parameters[k].inverse_transform(v) for k, v in kwargs.items()}, **self.fixed_parameters}
#         inputs = {k: np.reshape(v, (1, -1)).astype(float) for k, v in inputs.items()}
#         inputs = self.dataset.transform(inputs)

#         y = self.model.encode(inputs)['y']
#         x = self.model.decode(y)['x']
#         loss = self.dataset.evaluate(inputs, x)[0]
#         return -loss

#     def inner_loop_y(self, **kwargs) -> float:
#         inputs = {**{k: self.tunable_parameters[k].inverse_transform(v) for k, v in kwargs.items()}, **self.fixed_parameters}
#         inputs = {k: np.reshape(v, (1, -1)).astype(float) for k, v in inputs.items()}
#         inputs = self.dataset.transform(inputs)

#         x = self.model.decode(inputs)['x']
#         y = self.model.encode(x)['y']
#         loss = self.dataset.evaluate(inputs, y)[1]
#         return -loss

#     def generate(self, k:int, n_iter:int, init_points:int, eps:float=None, **kwargs):
#         self.BO.maximize(n_iter=0, init_points=init_points, acq=kwargs.get('acq', 'ei'), xi=kwargs.get('xi', 1e-2), **kwargs)
#         targets, params = [], []
#         while len(params) < k:
#             self.BO.maximize(n_iter=n_iter, init_points=0, acq=kwargs.get('acq', 'ei'), xi=kwargs.get('xi', 1e-2), **kwargs)
#             results = self.BO.res
#             for i in range(len(results)):
#                 if np.abs(results[i]['target']) > 0 and (eps is None or np.abs(results[i]['target']) < eps):
#                     targets.append(results[i]['target'])
#                     params.append(results[i]['params'])
#         targets = np.array(targets)
#         best_k_ixs = np.argsort(targets)[:k]
#         params = {**{n: p.inverse_transform(np.array([d[n] for d in params]).reshape((-1, 1)))[best_k_ixs] for n, p in self.tunable_parameters.items()},
#                   **{n: np.array([v]*k).reshape((-1, 1)) for n, v in self.fixed_parameters.items()}}

#         if not self.stateful:
#             if self.mode == 'x':
#                 self.BO = BayesianOptimization(self.inner_loop_x, {k: (0, 1) for k in self.tunable_parameters.keys()}, verbose=0)
#             else:
#                 self.BO = BayesianOptimization(self.inner_loop_y, {k: (0, 1) for k in self.tunable_parameters.keys()}, verbose=0)

#         return {'targets': targets[best_k_ixs], 'params': params}



class DataGenerator:
    def __init__(self, method:str, **kwargs):
        self.method = method
        self.inner_generator = None

    def generate(self, **kwargs):
        return self.inner_generator.generate(**kwargs)


class XDataGenerator(DataGenerator):
    def __init__(self, method:str, **kwargs):
        super().__init__(method, **kwargs)
        if method in ['random', 'uniform', 'rnd']:
            self.inner_generator = RandomDataGenerator(mode='x', **kwargs)
        elif method in ['sobol', 'sobol sequence']:
            self.inner_generator = SobolDataGenerator(mode='x', **kwargs)

class YDataGenerator(DataGenerator):
    def __init__(self, method:str, **kwargs):
        super().__init__(method, **kwargs)
        if method in ['random', 'uniform', 'rnd']:
            self.inner_generator = RandomDataGenerator(mode='y', **kwargs)
        elif method in ['sobol', 'sobol sequence']:
            self.inner_generator = SobolDataGenerator(mode='y', **kwargs)



class Sampler:
    def inverse_transform(self, values:np.array):
        pass

    def transform(self, values:np.array):
        pass

class UniformSampler(Sampler):
    def __init__(self, data:np.array=None, min_val:float=None, max_val:float=None):
        super(UniformSampler, self).__init__()
        if data is None and min_val is None and max_val is None:
            raise ValueError('Either "data" or "min_val" and "max_val" must be provided')
        self.min_val = min_val if min_val is not None else data.min()
        self.max_val = max_val if max_val is not None else data.max()

    def inverse_transform(self, values:np.array):
        return values * (self.max_val - self.min_val) + self.min_val

    def transform(self, values:np.array):
        return (values - self.min_val) / (self.max_val - self.min_val)

class QuantileSampler(Sampler):
    def __init__(self, data:np.array, min_val:float=None, max_val:float=None):
        super(QuantileSampler, self).__init__()
        self.min_val = min_val if min_val is not None else data.min()
        self.max_val = max_val if max_val is not None else data.max()
        self.qt = QuantileTransformer(n_quantiles=min(MAX_QUANTILES, len(data)), output_distribution="uniform", random_state=42)
        self.qt.fit(data[(data >= self.min_val) & (data <= self.max_val)].reshape(-1, 1))

    def inverse_transform(self, values: np.array):
        return self.qt.inverse_transform(values.reshape(-1, 1)).reshape(values.shape)

    def transform(self, values: np.array):
        return self.qt.transform(values.reshape(-1, 1)).reshape(values.shape)