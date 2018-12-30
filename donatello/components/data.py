import inspect
import pandas as pd
from donatello.utils.base import Dobject, find_value, replace_value
from donatello.components.folder import Folder
from donatello.utils.decorators import decorator, init_time, fallback, to_kwargs


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

    # should use folder as temp return
    @property
    def designData(self):
        return getattr(self, '_designData', self.raws.drop(self.target, axis=1, errors='ignore'))

    @designData.setter
    def designData(self, value):
        self._designData = value

    @property
    def targetData(self):
        return getattr(self, '_targetData', self.raws[self.target] if self.target in self.raws else None)

    @targetData.setter
    def targetData(self, value):
        self._targetData = value

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

    def subset(self, subset='train'):
        if isinstance(subset, str) and subset:
            subset[0] = subset[0].upper()
            attrs = ['{}{}'.format(attr, subset) for attr in ['design', 'target']]
            X, y = tuple(getattr(self,  attr) for attr in attrs)
        elif subset > 1:
            pass
        elif subset <= 1:
            pass
        return type(self)(X=X, y=y, **self.params)

    @property
    def __next__(self):
        return self.next

    def next(self):
        for train, test in self.folder.split(self.designData, self.targetData):
            results = [self.designData.iloc[train], self.designData.iloc[test]]
            if self.targetData is not None:
                results.extend([self.targetData.iloc[train],
                                self.targetData.iloc[test]])
            else:
                results.extend([None, None])  # Something better here
            return results

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


@to_kwargs
def pull_dataset(wrapped, instance, args, kwargs):
    dataset = kwargs.pop('dataset', None)

    if not dataset:
        X = kwargs.pop('X', None)
        y = kwargs.pop('y', None)
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

    kwargs.update({'dataset': dataset})
    return kwargs


@decorator
def package_dataset(wrapped, instance, args, kwargs):
    """
    From arguments - package X (and y if supervised) in Data object via type
    """
    kwargs = pull_dataset(wrapped, instance, args, kwargs)
    result = wrapped(**kwargs)
    return result


@decorator
def enforce_dataset(wrapped, instance, args, kwargs):
    kwargs = pull_dataset(wrapped, instance, args, kwargs)
    dataset = kwargs['dataset']
    result = wrapped(**kwargs)

    if not isinstance(result, Dataset) and isinstance(result, tuple) and len(result) <= 2:
        result = Dataset(X=result[0], y=result[1] if len(result) > 1 else dataset.targetData,
                         **dataset.params)
    return result


def subset_dataset(subset):
    @decorator
    def wrapper(wrapped, instance, args, kwargs):
        dataset = find_value(wrapped, args, kwargs, 'dataset').subset(subset)
        replace_value(wrapped, args, kwargs, 'dataset', dataset)
        result = wrapped(*args, **kwargs)
        return result
