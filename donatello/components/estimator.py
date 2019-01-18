from sklearn.model_selection import GridSearchCV

from donatello.utils.base import Dobject, BaseTransformer
from donatello.utils.decorators import pandas_series, fallback
from donatello.utils.helpers import now_string

def score_column(obj, column=1):
    def _scorer(obj

    return self.predict_method(X=X)[:, self.column]

scorer = 

class Estimator(Dobject, BaseTransformer):
    """
    Donatello's estimation objec to suppoert model training and prediction

    Args:
        model (sklearn.base.BaseEstimator): ML model implementing fit, predict[a-z]*
        method (str): string name of prediction method
        scorer (func | str): callable or string name of method for scoring
        paramGrid (dict): specificiont of  HPs to grid search
        gridKwargs (dict): options for grid search
        timeFormat (str): option to specify timestamp format
    """

    def __init__(self,
                 model=None,
                 method='predict',
                 scorer='no_op',
                 paramGrid={},
                 gridKwargs={},
                 timeFormat="%Y_%m_%d_%H_%M"
                 ):

        self._initTime = now_string(timeFormat)

        self.model = model
        self.method = method
        self.scorer = scorer
        self.column = column

        self.paramGrid = paramGrid
        self.gridKwargs = gridKwargs
        self.timeFormat = timeFormat

        self.declaration = self.get_params()

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

    # Move to dispatch
    @pandas_series
    def score(self, X, name=''):
        score = self.scorer if callable(self.scorer) else getattr(self, self.scorer)
        scores = score(X)
        return scores

    def no_op(self, X):
        """
        Scoring function
        """
        return self.predict_method(X=X)

    def score_column(self, X):
        """
        Scoring function
        """
        return self.predict_method(X=X)[:, self.column]

    def score_invert(self, X):
        """
        Scoring function
        """
        return -1 * self.predict_method(X=X)

    def __getattr__(self, name):
        return getattr(self.model, name)

    def get_feature_names(self):
        return getattr(self, 'features', [])
