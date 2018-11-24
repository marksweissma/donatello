import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from donatello.utils.base import Dobject
from donatello.utils.helpers import nvl
from donatello.utils.decorators import decorator, init_time


typeDispatch = {'splitter': {None: KFold,
                             'classification': StratifiedKFold,
                             'regression': KFold
                             }
                }

class Data(Dobject):
    """
    Object for managing data and helping prevent leakage

    :param obj raws: raw data
    :param dict queries: queries to execute to fetch data if not directly passed
    :param func etl: default function to execute queries
    :param bool copyRaws: option to have contents return copy of raws to preserve fetch
    :param obj X: option to specify design directly
    :param obj y: option to specify target directly
    :param obj type splitType: type of splitter to leverage in iterator
    :param obj splitKwargs: kwargs for split type to instantiate with in constructor
    """
    @init_time
    def __init__(self, raws=None, queries=None,
                 etl=pd.read_csv, copyRaws=True,
                 X=None, y=None, mlType=None,
                 typeDispatch=typeDispatch,
                 splitKwargs={'n_splits': 5,
                              'shuffle': True,
                              'random_state': 22}
                 ):

        self.copyRaws = copyRaws
        self.link(raws, X, y)
        self.queries = queries
        self.etl = etl
        self.typeDispatch = typeDispatch
        self.splitter = typeDispatch.get('splitter').get(mlType)(**splitKwargs)

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

@decorator
def package_data(wrapped, instance, args, kwargs):
    """
    Package X (and y if supervised) in Data object via mlType
    """
    X = kwargs.pop('X', None)
    y = kwargs.pop('y', None)
    data = kwargs.pop('data', None)

    if data is None and X is None:
        data = instance.data
    elif X is not None:
        mlType = getattr(instance, '_mlType', None)
        data = Data(X=X, y=y, mlType=mlType)
    if not data.hasContents and data.queries:
        data.execute_queries(data.queries)

    result = wrapped(data=data, **kwargs)
    return result
