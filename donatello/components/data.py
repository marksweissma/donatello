import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from donatello.utils.base import Dobject
from donatello.utils.decorators import decorator, init_time, fallback


typeDispatch = {'splitter': {None: KFold,
                             'classification': StratifiedKFold,
                             'regression': KFold
                             }
                }


class Dataset(Dobject):
    """
    Object for managing data and helping prevent leakage

    Args:
        raws (obj): raw data
        queries (dict): queries to execute to fetch data if not directly passed
        querier (func): default function to execute queries
        copyRaws (bool): option to have data return copy of raws to preserve fetch
        X (obj): option to specify design directly
        y (obj): option to specify target directly
        type (obj) splitType: type of splitter to leverage in iterator
        splitDeclaration (obj): kwargs for split type to instantiate with in constructor
    """
    @init_time
    def __init__(self, raws=None, queries=None,
                 querier=pd.read_csv, copyRaws=False,
                 X=None, y=None, mlType=None,
                 typeDispatch=typeDispatch,
                 splitDeclaration={'n_splits': 5,
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
        self.splitter = typeDispatch.get('splitter').get(mlType)(**splitDeclaration)
        self.groupKey = groupKey
        self.params = {'mlType': mlType, 'typeDispatch': typeDispatch,
                       'splitDeclaration': splitDeclaration, 'groupKey': groupKey}

    @property
    def data(self):
        value = getattr(self, '_data', None)
        if value is None:
            if self.copyRaws and self.raws is not None:
                value = self.raws.copy()
            else:
                value = getattr(self, 'raws', None)
        return value

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def hasData(self):
        if isinstance(self.data, dict):
            state = self.data != {}
        else:
            state = self.data is not None
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
        accessed via py:attr:`Data.data`

            queries (dict): payload of queries
            querier (func): option to specify executor at the execution\
                level rather than the query level
        """

        if not queries:
            self.raws = querier()
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
def package_dataset(wrapped, instance, args, kwargs):
    """
    Package X (and y if supervised) in Data object via mlType
    """
    X = kwargs.pop('X', None)
    y = kwargs.pop('y', None)
    dataset = kwargs.pop('dataset', None)

    if not dataset:
        if X is None and hasattr(instance, 'dataset'):
            dataset = instance.dataset
        elif X is not None and hasattr(instance, 'data'):
            dataset = Dataset(X=X, y=y, **instance.dataset.get_params())

        elif X is not None:
            mlType = getattr(instance, '_mlType', None)
            dataset = Dataset(X=X, y=y, mlType=mlType)

    if not dataset.hasData and dataset.queries is not None:
        dataset.execute_queries(dataset.queries)

    result = wrapped(dataset=dataset, **kwargs)
    return result
