import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from donatello.utils.base import Dobject
from donatello.utils.decorators import decorator, init_time, fallback


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
    :param func querier: default function to execute queries
    :param bool copyRaws: option to have contents return copy of raws to preserve fetch
    :param obj X: option to specify design directly
    :param obj y: option to specify target directly
    :param obj type splitType: type of splitter to leverage in iterator
    :param obj splitKwargs: kwargs for split type to instantiate with in constructor
    """
    @init_time
    def __init__(self, raws=None, queries=None,
                 querier=pd.read_csv, copyRaws=True,
                 X=None, y=None, mlType=None,
                 typeDispatch=typeDispatch,
                 splitKwargs={'n_splits': 5,
                              'shuffle': True,
                              'random_state': 22},
                 groupKey=None,
                 ):

        self.copyRaws = copyRaws
        self._mlType = mlType
        self.link(raws, X, y)
        self.queries = queries
        self.querier = querier
        self.typeDispatch = typeDispatch
        self.splitter = typeDispatch.get('splitter').get(mlType)(**splitKwargs)
        self.groupKey = groupKey
        self.params = {'mlType': mlType, 'typeDispatch': typeDispatch,
                       'splitKwargs': splitKwargs, 'groupKey': groupKey}

    @property
    def contents(self):
        value = getattr(self, '_contents', None)
        if value is None:
            if self.copyRaws and self.raws is not None:
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

    @fallback('querier')
    def execute_queries(self, queries=None, querier=None):
        """
        Execute data extraction via cascading querying dependencies
        Attaches return to :py:attr:`Data.raws`, which can be
        accessed via py:attr:`Data.contents`

        :param dict queries: payload of queries
        :param func querier: option to specify executor at the execution\
                level rather than the query level
        """

        if not queries:
            self.raws = querier
        else:
            for name, query in queries.items():
                querier = query.get('querier', querier)

                payload = {i: j for i, j in query.items() if i != 'querier'}
                if len(queries) == 1:
                    self.raws = querier(**payload)
                else:
                    self.raws[name] = querier(**payload)

    def unpack_splits(self, splitResults):
        for attr, result in splitResults.items():
            setattr(self, attr, result)

    def package_split_kwargs(self):
        kwargs = {'groups': self.designData[self.groupKey].values} if self.groupKey else {}
        return kwargs

    def __iter__(self):
        kwargs = self.package_split_kwargs()
        for train, test in self.splitter.split(self.designData,
                                               self.targetData, **kwargs):
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

    if not data:
        if X is None and hasattr(instance, 'data'):
            data = instance.data
        elif X is not None and hasattr(instance, 'data'):
            data = Data(X=X, y=y, **instance.data.get_params())

        elif X is not None:
            mlType = getattr(instance, '_mlType', None)
            data = Data(X=X, y=y, mlType=mlType)

    if not data.hasContents and data.queries is not None:
        data.execute_queries(data.queries)

    result = wrapped(data=data, **kwargs)
    return result
