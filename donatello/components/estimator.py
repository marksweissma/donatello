from warnings import warn
from abc import ABCMeta, abstractmethod
from sklearn.model_selection import GridSearchCV

from donatello.utils.base import BaseTransformer
from donatello.utils.decorators import pandas_series, grid_search
from donatello.utils.helpers import now_string, nvl
from donatello.utils.transformers import Selector


class BaseEstimator(BaseTransformer):
    """
    Donatello's Base Estimation object. Leverages a transformer to prepare and transform
    design and an ML model to fit and predict. Supports options for grid searching for
    hyperparameter optimization

    :param donatello.utils.base.BaseTransformer transformer: object implementing fit, transform, fit_transform
    :param sklearn.base.BaseEstimator model: ML model implementing fit, predict[a-z]*
    :param str method: string name of prediction method
    :param dict paramGrid: specificiont of  HPs to grid search
    :param dict gridKwargs: options for grid search
    :param str timeFormat: option to specify timestamp format
    """
    gridType = GridSearchCV
    __meta__ = ABCMeta

    def __init__(self,
                 transformer=None,
                 model=None,
                 method=None,
                 paramGrid={},
                 gridKwargs={},
                 timeFormat="%Y_%m_%d_%H_%M"
                 ):

        self._initTime = now_string(timeFormat)

        self.transformer = nvl(transformer, Selector(reverse=True))
        self.model = model
        self.method = method if method else getattr(self, 'method', None)

        self.paramGrid = paramGrid
        self.gridKwargs = gridKwargs
        self.timeFormat = timeFormat

        self.declaration = self.get_params()

# Read only attributes and magic methods
    @property
    def name(self):
        name = "_".join([self.__class__.__name__,
                         self.model.__class__.__name__])
        return name

    @property
    def declaration(self):
        return self._declaration.copy()

    @declaration.setter
    def declaration(self, value):
        self._declaration = value

    def __repr__(self):
        rep = ['{model} created at {time}'.format(model=self.name,
                                                  time=self._initTime),
               super(BaseEstimator, self).__repr__()]
        return "\n --- \n **sklearn repr** \n --- \n".join(rep)

    @property
    def predict_method(self):
        """
        """
        return getattr(self, self.method)

# Estimator determined properties
    @property
    def fields(self):
        return getattr(self.transformer, '_fields', [])

    @property
    def features(self):
        return getattr(self.transformer, '_features', [])

# Fitting
    @grid_search
    def fit(self, X, y=None,
            gridSearch=False, priorGridSearch=False,
            paramGrid=None, gridKwargs=None, **kwargs):

        transformed = self.transformer.fit_transform(X, y, **kwargs)
        self.model.fit(transformed, y)

        return self

    @abstractmethod
    def score(self, X):
        warn('no score method implemented defaulting transform')
        return self.transformer.transform(X)

    def transform(self, X, **kwargs):
        return self.transformer.transform(X, **kwargs)

# Predicting
    def __getattr__(self, name):
        prediction_methods = ['predict', 'predict_proba',
                              'predict_log_proba', 'decision_function']
        if name in prediction_methods:
            attr = getattr(self.model, name)

            def wrapped(X, *args, **kwargs):
                X = self.transform(X, **kwargs)
                result = attr(X, *args, **kwargs)
                return result
            return wrapped
        else:
            return getattr(self.model, name)

    def get_feature_names(self):
        return getattr(self, 'features', [])


class Regressor(BaseEstimator):
    method = 'predict'

    @pandas_series
    def score(self, X, name=''):
        return self.predict_method(X)


class Classifier(BaseEstimator):
    method = 'predict_proba'

    @pandas_series
    def score(self, X, name=''):
        return self.predict_method(X)[:, 1]
