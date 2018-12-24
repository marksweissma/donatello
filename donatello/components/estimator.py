from sklearn.model_selection import GridSearchCV

from donatello.utils.base import BaseTransformer
from donatello.utils.decorators import pandas_series, fallback
from donatello.utils.helpers import now_string


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
                 model=None,
                 mlClay='regression',
                 typeDispatch={'regression': {'method': 'predict', 'score': 'score_all'},
                               'classification': {'method': 'predict_proba', 'score': 'score_first'}
                               },
                 paramGrid={},
                 gridKwargs={},
                 timeFormat="%Y_%m_%d_%H_%M"
                 ):

        self._initTime = now_string(timeFormat)

        self.model = model
        self._mlClay = mlClay
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
    def mlClay(self):
        return self._mlClay

    @property
    def method(self):
        return self.typeDispatch[self.mlClay]['method']

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
        Fields passed into model
        """
        return getattr(self.model, '_fields', [])

    @property
    def features(self):
        """
        Features coming from model
        """
        return getattr(self.model, '_features', [])

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
        self.model.fit(X=X, y=y, **kwargs)
        return self

    @pandas_series
    def score(self, X, name=''):
        scores = getattr(self, self.typeDispatch[self.mlClay]['score'])(X)
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

    def __getattr__(self, name):
        return getattr(self.model, name)

    def get_feature_names(self):
        return getattr(self, 'features', [])
