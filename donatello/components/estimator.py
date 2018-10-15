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
        """
        Name of object, defaults to class name + model name
        """
        name = "_".join([self.__class__.__name__,
                         self.model.__class__.__name__])
        return name

    @property
    def declaration(self):
        """
        Dictionary of kwargs given during instantiation
        """
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
        Unified prediction interface
        """
        return getattr(self, self.method)

# Estimator determined properties
    @property
    def fields(self):
        """
        Fields passed into transformer
        """
        return getattr(self.transformer, '_fields', [])

    @property
    def features(self):
        """
        Features coming from transformer
        """
        return getattr(self.transformer, '_features', [])

# Fitting

    def grid_search(self, X, y=None, gridSearch=True, paramGrid=None, gridKwargs=None):
        """
        """
        if gridSearch and paramGrid:
            from sklearn.model_selection import GridSearchCV

            self.grid_search(paramGrid, gridKwargs)
            self.gridSearch = GridSearchCV(estimator=self,
                                               param_grid=paramGrid,
                                               **gridKwargs)
            self.gridSearch.fit(X, y=y, gridSearch=False)
            self.set_params(**self.gridSearch.best_params_)

    def fit(self, X, y=None,
            gridSearch=True,
            paramGrid=None, gridKwargs=None, **kwargs):
        """
        Fit method with options for grid searching hyperparameters
        """
        paramGrid = nvl(paramGrid, self.paramGrid)
        gridKwargs = nvl(gridKwargs, self.gridKwargs)

        self.grid_search(X, y=y, gridSearch=gridSearch, paramGrid=paramGrid, gridKwargs=gridKwargs)

        transformed = self.transformer.fit_transform(X, y, **kwargs)
        self.model.fit(transformed, y)
        return self

    @abstractmethod
    def score(self, X):
        """
        Scoring function
        """
        warn('no score method implemented defaulting transform')
        return self.transformer.transform(X)

    def transform(self, X, **kwargs):
        """
        Apply fit transformer to X
        """
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
    """
    Estimator for regression, applies predict as class atribute for method
    """
    method = 'predict'

    @pandas_series
    def score(self, X, name=''):
        return self.predict_method(X)


class Classifier(BaseEstimator):
    """
    Estimator for classification, applies predict_proba as class atribute for method
    """
    method = 'predict_proba'

    @pandas_series
    def score(self, X, name=''):
        return self.predict_method(X)[:, 1]
