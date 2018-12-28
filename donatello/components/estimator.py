from sklearn.model_selection import GridSearchCV

from donatello.utils.base import Dobject, BaseTransformer
from donatello.utils.decorators import pandas_series, fallback
from donatello.utils.helpers import now_string


class Estimator(Dobject, BaseTransformer):
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

    def __init__(self,
                 model=None,
                 foldClay=None,
                 scoreClay='regression',
                 foldDispatch=None,
                 scoreDispatch={'regression': {'method': 'predict', 'score': 'score_all'},
                                'classification': {'method': 'predict_proba', 'score': 'score_first'},
                                'anomaly': {'method': 'decision_function', 'score': 'score_invert'}
                                },
                 paramGrid={},
                 gridKwargs={},
                 timeFormat="%Y_%m_%d_%H_%M"
                 ):

        self._initTime = now_string(timeFormat)

        self.model = model
        self._foldClay = foldClay
        self._scoreClay = scoreClay
        self.foldDispatch = foldDispatch
        self._scoreDispatch = scoreDispatch

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
    def method(self):
        return self.scoreDispatch[self.scoreClay]['method']

    @property
    def scoreDispatch(self):
        return self._scoreDispatch

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
    @fallback('paramGrid', 'gridKwargs')
    def grid_search(self, X=None, y=None, gridSearch=True,
                    paramGrid=None, gridKwargs=None):
        """
        Grid search over hyperparameter space
        """
        if paramGrid and gridSearch:
            self.gridSearch = GridSearchCV(estimator=self,
                                           param_grid=paramGrid,
                                           **gridKwargs)
            self.gridSearch.fit(X=X, y=y, gridSearch=False)
            self.set_params(**self.gridSearch.best_params_)

    def fit(self, X=None, y=None, gridSearch=True,
            paramGrid=None, gridKwargs=None, **kwargs):
        """
        Fit method with options for grid searching hyperparameters
        """
        self.grid_search(X=X, y=y, gridSearch=gridSearch,
                         paramGrid=paramGrid, gridKwargs=gridKwargs)
        self.model.fit(X=X, y=y, **kwargs)
        return self

    @pandas_series
    def score(self, X, name=''):
        scores = getattr(self, self.scoreDispatch[self.scoreClay]['score'])(X)
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

    def score_invert(self, X):
        """
        Scoring function
        """
        return -1 * self.predict_method(X=X)

    def __getattr__(self, name):
        return getattr(self.model, name)

    def get_feature_names(self):
        return getattr(self, 'features', [])
