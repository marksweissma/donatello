import inspect
import pandas as pd
from donatello.utils.base import Dobject, find_value
from donatello.components.folder import Folder
from donatello.utils.decorators import decorator, init_time, fallback


class Dataset(Dobject):
    """
    Object for owning a dataset

    Args:
        raws (obj): raw data
        queries (dict): queries to execute to fetch data if not directly passed
        querier (func): default function to execute queries
        copyRaws (bool): option to have data return copy of raws to preserve fetch
        X (obj): option to specify design directly
        y (obj): option to specify target directly
        type (obj) foldType: type of folder to leverage in iterator
        foldDeclaration (obj): kwargs for split type to instantiate with in constructor
    """
    @init_time
    def __init__(self, raws=None, X=None, y=None,
                 queries=None, querier=pd.read_csv, copyRaws=False,
                 foldClay=None, foldDispatch=Folder,
                 scoreClay=None,
                 target=None, primaryKey=None,
                 dap=None
                 ):

        self.copyRaws = copyRaws
        self.queries = queries
        self.querier = querier

        self.link(raws, X, y)

        self._foldClay = foldClay
        self.foldDispatch = foldDispatch
        self.folder = foldDispatch(foldClay=foldClay, target=target, primaryKey=primaryKey, dap=dap)

        self.target = target
        self.primaryKey = primaryKey
        self.dap = dap

    @property
    def params(self):
        spec = inspect.getargspec(self.__init__)
        exclusions = set(['self', 'raws', 'X', 'y'])
        params = {param: getattr(self, param) for param in spec.args if param not in exclusions}
        return params

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

        Args:
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

    def unpack_folds(self, foldResults):
        if isinstance(foldResults, dict):
            [setattr(self, attr, result) for attr, result in foldResults.items()]
        else:
            attrs = ['designTrain', 'designTest', 'designData',
                     'targetTrain', 'targetTest', 'targetData']
            [setattr(self, attr, result) for attr, result in zip(attrs, foldResults)]

    def __iter__(self):
        for train, test in self.folder.split(self.designData, self.targetData):
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
    Frome keyword arguments - package X (and y if supervised) in Data object via type
    """
    dataset = find_value(wrapped, args, kwargs, 'dataset')

    if not dataset:
        X = find_value(wrapped, args, kwargs, 'X')
        y = find_value(wrapped, args, kwargs, 'y')

        if X is None and hasattr(instance, 'dataset'):
            dataset = instance.dataset
        elif X is not None and hasattr(instance, 'data'):
            dataset = Dataset(X=X, y=y, **instance.dataset.get_params())

        elif X is not None:
            scoreClay = getattr(instance, '_scoreClay', None)
            dataset = Dataset(X=X, y=y, scoreClay=scoreClay)

    if not dataset.hasData and dataset.queries is not None:
        dataset.execute_queries(dataset.queries)

    exclusions = set(['X', 'y', 'dataset'])
    result = wrapped(dataset=dataset, **{i: j for i, j in kwargs.items() if i not in exclusions})
    return result
