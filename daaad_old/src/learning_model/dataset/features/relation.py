import numpy as np
from types import LambdaType

from src.learning_model.dataset.features.real import DataReal

class DataRelation(DataReal):

    def __init__(self, data:np.array, related_featuers:list, lam:LambdaType, name:str, scaling_type:str='standard', **kwargs):
        super(DataRelation, self).__init__(data, name, scaling_type, **kwargs)
        self.related_features = related_featuers
        self.lam = lam

    def get_config(self):
        return {
            'lam': self.lam, 
            'related_features': self.related_features, 
            **super(DataRelation, self).get_config(),
        }

class DataDivisionRelation(DataRelation):
    def __init__(self, data, feature1, feature2, name):
        super(DataDivisionRelation, self).__init__(data, [feature1, feature2], lambda x, y: x / y, name, scaling_type=None)
        self.feature1 = feature1
        self.feature2 = feature2
        feature1.encoder.fit(np.vstack([feature1.data, feature2.data]))
        feature2.encoder.fit(np.vstack([feature1.data, feature2.data]))

    def transform(self, data):
        return super(DataDivisionRelation, self).transform(data) / self.feature1.encoder.scale_[0]

    def inverse_transform(self, data):
        return super(DataDivisionRelation, self).transform(data * self.feature1.encoder.scale_[0])