from sklearn.model_selection import GridSearchCV

from donatello.utils.base import BaseTransformer
from donatello.utils.decorators import pandas_series, fallback
from donatello.utils.helpers import now_string, nvl
from donatello.utils.transformers import Selector


class Estimator(BaseTransformer):
    """
    Donatello's Base Estimation object. Leverages a transformer to prepare and transform
    design and an ML model to fit and predict. Supports options for grid searching for
    hyperparameter optimization

    Args:
        transformer (donatello.utils.base.BaseTransformer): object implementing fit, transform, fit_transform
        model (sklearn.base.BaseEstimator): ML model implementing fit, predict[a-z]*
        method (str): string name of prediction method
        paramGrid (dict): specificiont of  HPs to grid search
        gridKwargs (dict): options for grid search
        timeFormat (str): option to specify timestamp format
    """

    # this is to provide interface and not call super
    def __init__(self,
                 transformer=None,
                 model=None,
                 mlType='regression',
                 typeDispatch={'regression': {'method': 'predict', 'score': 'score_all'},
                               'classification': {'method': 'predict_proba', 'score': 'score_first'}
                               },
                 paramGrid={},
                 gridKwargs={},
                 timeFormat="%Y_%m_%d_%H_%M"
                 ):

        self._initTime = now_string(timeFormat)

        self.transformer = nvl(transformer, Selector(reverse=True))
        self.model = model
        self._mlType = mlType
        self._typeDispatch = typeDispatch

        self.paramGrid = paramGrid
        self.gridKwargs = gridKwargs
        self.timeFormat = timeFormat

        self.declaration = self.get_params()

    @property
    def declaration(self):
        """
        Dictionary of kwargs given during instantiation
        """
        return self._declaration.copy()

    @declaration.setter
    def declaration(self, value):
        self._declaration = value

    @property
    def mlType(self):
        return self._mlType

    @property
    def method(self):
        return self.typeDispatch[self.mlType]['method']

    @property
    def typeDispatch(self):
        return self._typeDispatch

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
    def sklearn_grid_search(self, X=None, y=None,
                    paramGrid=None, gridKwargs=None
                    ):
        """
        """

        self.gridSearch = GridSearchCV(estimator=self,
                                       param_grid=paramGrid,
                                       **gridKwargs)
        self.gridSearch.fit(X=X, y=y, gridSearch=False)
        self.set_params(**self.gridSearch.best_params_)

    @fallback('paramGrid', 'gridKwargs')
    def grid_search(self, X=None, y=None, gridSearch=True,
                    paramGrid=None, gridKwargs=None):
        """
        """
        if paramGrid and gridSearch:
            self.sklearn_grid_search(X=X, y=y, paramGrid=paramGrid, gridKwargs=gridKwargs)

    def fit(self, X=None, y=None,
            gridSearch=True,
            paramGrid=None, gridKwargs=None, **kwargs):
        """
        Fit method with options for grid searching hyperparameters
        """
        self.grid_search(X=X, y=y, gridSearch=gridSearch, paramGrid=paramGrid, gridKwargs=gridKwargs)

        transformed = self.transformer.fit_transform(X=X, y=y, **kwargs)
        self.model.fit(transformed, y)
        return self

    @pandas_series
    def score(self, X, name=''):
        scores = getattr(self, self.typeDispatch[self.mlType]['score'])(X)
        return scores

    def score_all(self, X):
        """
        Scoring function
        """
        return self.predict_method(X=X)

    def score_first(self, X):
        """
        Scoring function
        """
        return self.predict_method(X=X)[:, 1]

    def transform(self, X=None, **kwargs):
        """
        Apply fit transformer to X
        """
        return self.transformer.transform(X=X, **kwargs)

    def __getattr__(self, name):
        prediction_methods = ['predict', 'predict_proba',
                              'predict_log_proba', 'decision_function']
        if name in prediction_methods:
            attr = getattr(self.model, name)

            def wrapped(X, *args, **kwargs):
                X = self.transform(X=X, **kwargs)
                result = attr(X=X, *args, **kwargs)
                return result
            return wrapped
        else:
            return getattr(self.model, name)

    def get_feature_names(self):
        return getattr(self, 'features', [])
