import pandas as pd
from functools import wraps
from sklearn.model_selection import KFold, StratifiedKFold
from donatello.utils.base import Dobject
from donatello.utils.helpers import nvl
from donatello.utils.decorators import init_time


class Data(Dobject):
    """
    """
    @init_time
    def __init__(self, raws=None, queries=None,
                 etl=pd.read_csv, copyRaws=True,
                 X=None, y=None,
                 splitType=None,
                 splitKwargs={'n_splits': 5,
                              'shuffle': True,
                              'random_state': 22},
                 ):

        self.copyRaws = copyRaws
        self.link(raws, X, y)
        self.queries = queries
        self.etl = etl
        self.splitter = splitType(**splitKwargs)

    @property
    def contents(self):
        value = getattr(self, '_contents', None)
        if value is None:
            if self.copyRaws:
                value = self.raws.copy()
            else:
                value = self.raws
        return value

    @contents.setter
    def contents(self, value):
        self._contents = value

    @property
    def hasContents(self):
        if isinstance(self.contents, dict):
            state = self.contents != {}
        else:
            state = self.contents is not None
        return state

    def link(self, raws=None, X=None, y=None):
        if raws is None and X is not None:
            raws = [X, y] if y is not None else [X]
            self.raws = pd.concat(raws, axis=1)
            self.designData = X
            self.targetData = y
        else:
            self.raws = raws

    def execute_queries(self, queries=None, etl=None):
        for name, query in queries.iteritems():
            etl = query['etl'] if 'etl' in query else nvl(etl, self.etl)
            # Don't pop to prevent mutation
            payload = {i: j for i, j in query.iteritems() if i != 'etl'}
            if len(queries) == 1:
                self.raws = etl(**payload)
            else:
                self.raws[name] = etl(**payload)

    def unpack_splits(self, splitResults):
        for attr, result in splitResults.iteritems():
            setattr(self, attr, result)

    def __iter__(self):
        for train, test in self.splitter.split(self.designData,
                                               self.targetData):
            results = [self.designData.iloc[train], self.designData.iloc[test]]
            if self.targetData is not None:
                results.extend([self.targetData.iloc[train],
                                self.targetData.iloc[test]])
            else:
                results.extend([None, None])  # Something better here
            yield results
        raise StopIteration

    @property
    def next(self):
        return self.__next__


class DataClassification(Data):
    def __init__(self, splitType=StratifiedKFold, **kwargs):
        payload = kwargs
        payload.update({'splitType': splitType})
        super(DataClassification, self).__init__(**payload)


class DataRegression(Data):
    def __init__(self, splitType=KFold, **kwargs):
        payload = kwargs
        payload.update({'splitType': splitType})
        super(DataRegression, self).__init__(**payload)


def package_data(func):
    @wraps(func)
    def wrapped(self, data=None, X=None, y=None, **fitParams):
        if data is None and X is None:
            data = self.data
        elif X is not None:
            mlType = getattr(self, '_mlType', None)
            if mlType == 'Classification':
                data = DataClassification(X=X, y=y)
            elif mlType == 'Regression':
                data = DataRegression(X=X, y=y)
            else:
                data = Data(X=X, y=y)

        if not data.hasContents and data.queries:
            data.execute_queries(data.queries)
        return func(self, data=data, **fitParams)
    return wrapped
