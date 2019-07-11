from sklearn.model_selection import GridSearchCV

from donatello.components import data

from donatello.utils.base import Dobject, BaseTransformer
from donatello.utils.decorators import pandas_series, fallback
from donatello.utils.helpers import now_string, access


def score_second(model, X):
    """
    Scoring function
    """
    return model.predict_method(X=X)[:, 1]


def score_invert(model, X):
    """
    Scoring function
    """
    return -1 * model.predict_method(X=X)


SCORE_REGISTRY = {
    'score_second': score_second,
    'score_invert': score_invert
}


class Estimator(Dobject, BaseTransformer):
    """
    Donatello's estimation object to support model training and prediction

    Note:
        sklearn GridSearchCV does not support indexing throuhg collections. If
        executing skSearch, dataset.designData must be flat (not dict)

    Args:
        model (sklearn.base.BaseEstimator): ML model implementing fit, predict[a-z]*
        method (str): string name of prediction method
        scorer (func | str): callable or string name of method for scoring
        paramGrid (dict): specificiont of  HPs to grid search
        searchKwargs (dict): options for grid search
        timeFormat (str): option to specify timestamp format
    """

    def __init__(self,
                 model=None,
                 method='predict',
                 scorer=None,
                 server=None,
                 paramGrid={},
                 searchKwargs={},
                 timeFormat="%Y_%m_%d_%H_%M"
                 ):

        self._initTime = now_string(timeFormat)

        self.model = model
        self.method = method
        self.scorer = scorer if (not scorer or callable(scorer)) else SCORE_REGISTRY[scorer]
        self.server = server

        self.paramGrid = paramGrid
        self.searchKwargs = searchKwargs
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
        return getattr(self.model, 'fields', [])

    @property
    def features(self):
        """
        Features coming from model
        """
        return getattr(self.model, 'features', [])

# Fitting
    def grid_search(self, dataset=None, gridSearch=True,
                    paramGrid=None, searchKwargs=None):
        """
        Grid search over hyperparameter space
        """
        if paramGrid and gridSearch:
            print('grid searching')
            self.gridSearch = GridSearchCV(estimator=self,
                                           param_grid=paramGrid,
                                           **searchKwargs)

            groups = access(dataset.designData,
                            **dataset.groupDap) if dataset.groupDap else None

            self.gridSearch.fit(X=dataset.designData, y=dataset.targetData,
                                groups=groups, gridSearch=False)

            self.set_params(**self.gridSearch.best_params_)

    @data.package_dataset
    @fallback('paramGrid', 'searchKwargs')
    def fit(self, X=None, y=None, dataset=None, gridSearch=True,
            paramGrid=None, searchKwargs=None, **kwargs):
        """
        Fit method with options for grid searching hyperparameters
        """
        self._records = len(dataset)
        self.grid_search(dataset=dataset, gridSearch=gridSearch,
                         paramGrid=paramGrid, searchKwargs=searchKwargs, **kwargs)
        self.model.fit(X=dataset.designData, y=dataset.targetData, **kwargs)
        return self

    # move to
    # @pandas_dim
    @pandas_series
    @fallback('scorer')
    def score(self, X, name='', scorer=None):
        scores = scorer(self, X) if scorer else self.predict_method(X=X)
        return scores

    @fallback('server')
    def serve(self, X, server=None):
        serves = server(self, X) if server else X
        return serves

    @data.package_dataset
    def transform(self, X=None, y=None, dataset=None):
        return self.score(dataset.designData)

    def get_feature_names(self):
        return getattr(self, 'features', [])

    def __getattr__(self, attr):
        return getattr(self.model, attr) if attr != '_name' else self.__class__.__name__
