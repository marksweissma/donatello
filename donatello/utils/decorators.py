import pandas as pd
from functools import wraps
from donatello.utils.helpers import now_string, nvl


def init_time(func):
    """
    Add _initTime attribute to object, format prescribed by
    **strFormat** kwarg
    """
    @wraps(func)
    def wrapped(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        signature = kwargs.get('timeFormat', None)
        payload = {'strFormat': signature} if signature else {}
        self._initTime = now_string(**payload)
        return result
    return wrapped


def split_data(func):
    """
    Split data contents into train and test sets
    """
    @wraps(func)
    def wrapped(self, data=None, X=None, y=None, **fitParams):
        if data.hasContents and self.splitter:
            data.unpack_splits(self.splitter.fit_transform(data))
        else:
            data.designData = data.contents
        return func(self, data, **fitParams)
    return wrapped


def prepare_design(func):
    """
    Apply `combiner` to data to create final design (if applicable)
    """
    @wraps(func)
    def wrapped(self, data=None, X=None, y=None, **fitParams):
        if getattr(self, 'combiner', None):
            data = self.combiner.fit_transform(data)
        return func(self, data, **fitParams)
    return wrapped


def pandas_series(func):
    """
    Enfore output as :py:class:`pandas.Series`
    """
    @wraps(func)
    def wrapped(self, X, index='index', name='', **kwargs):
        yhat = func(self, X)
        index = X[index] if index in X else X.index
        name = nvl(name, self.name)
        return pd.Series(yhat, index=index, name=name)
    return wrapped


def pandas_df(func):
    """
    Enfore output as :py:class:`pandas.DataFrame`
    """
    @wraps(func)
    def wrapped(self, X, index='index', columns=[], **kwargs):
        _df = func(self, X)
        index = X[index] if index in X else X.index
        columns = nvl(columns, X.columns)
        return pd.DataFrame(_df, index=index, columns=columns)
    return wrapped


def grid_search(func):
    """
    Grid search of hyperparameter space

    Instantates search through `obj.GridType` constructor call

    Stores `gridSearch` attribute with object
    """
    @wraps(func)
    def wrapped(self, X, y=None,
                gridSearch=True, priorGridSearch=False,
                paramGrid=None, gridKwargs=None, **kwargs):
        """
        """
        paramGrid = nvl(paramGrid, self.paramGrid)
        gridKwargs = nvl(gridKwargs, self.gridKwargs)

        if gridSearch and paramGrid:
            self.gridSearch = self.gridType(estimator=self,
                                            param_grid=paramGrid,
                                            **gridKwargs)
            self.gridSearch.fit(X, y, gridSearch=False)
            self.set_params(**self.gridSearch.best_params_)

        result = func(self, X, y, **kwargs)
        return result
    return wrapped
