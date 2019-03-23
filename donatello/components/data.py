import inspect
import pandas as pd

from donatello.utils.base import Dobject
from donatello.components.fold import Fold
from donatello.utils.decorators import decorator, init_time, fallback
from donatello.utils.helpers import find_value, replace_value, nvl, access


def has_data(data):
    """
    check if data exists. compares dict to empty dict else to None

    Args:
        data (obj): dat ato validate
    Returns:
        bool: whether data is present
    """
    if isinstance(data, dict):
        state = data != {}
    else:
        state = data is not None

    return state


# !!!TODO
# deprecate this abomination
@decorator
def fit_fold(wrapped, instance, args, kwargs):
    result = wrapped(*args, **kwargs)
    try:
        instance.fold.fit(instance)
        attrs = ('designTrain', 'designTest',
                 'targetTrain', 'targetTest')
        folded = next(instance.fold.fold(instance))
        [setattr(instance, attr, value) for attr, value in zip(attrs, folded) if value is not None]
    except:
        pass
    return result


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
        type (obj) foldType: type of fold to leverage in iterator
        foldDeclaration (obj): kwargs for split type to instantiate with in constructor
    """
    @init_time
    def __init__(self, raws=None, X=None, y=None,
                 queries=None, querier=pd.read_csv, copyRaws=False,
                 foldClay=None, foldType=Fold,
                 scoreClay=None,
                 target=None, primaryKey=None,
                 dap=None, force=False
                 ):

        self.copyRaws = copyRaws
        self.queries = queries
        self.querier = querier

        self.foldClay = foldClay
        self.foldType = foldType

        self.target = target
        self.primaryKey = primaryKey
        self.force = force

        self.fold = foldType(foldClay=foldClay, target=target, primaryKey=primaryKey, dap=dap)

        if any([i is not None for i in [raws, X, y]]):
            self.link(raws, X, y)

    @property
    def dap(self):
        return self.fold.dap

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
        return has_data(self.data)

    def link(self, raws=None, X=None, y=None):
        if raws is None and (X is not None or y is not None):
            raws = [X] if X is not None else []
            raws.append(y) if y is not None else None
            self.raws = pd.concat(raws, axis=1)
            self.designData = X
            self.targetData = y
            name = getattr(y, 'name', getattr(self, 'target', None))
            self.target = name
        else:
            self.raws = raws

        if has_data(raws) or has_data(X):
            self._fit_fold()

    def _fit_fold(self):
        self.fold.fit(self)
        attrs = ('designTrain', 'designTest',
                 'targetTrain', 'targetTest')
        folded = next(self.fold.fold(self))
        [setattr(self, attr, value) for attr, value in zip(attrs, folded)]

    def _split(self):
        return next(self.fold.fold(self))  # :(

    @property
    def _has_design(self):
        if self.data is None:
            has = False
        elif isinstance(self.data, pd.Series):
            has = self.data.name != self.target
        else:
            try:
                has = bool([i for i in self.data if i != self.target])
            except TypeError:
                has = False
        return has

    @property
    def _has_target(self):
        if self.data is None:
            has = False
        elif isinstance(self.data, pd.Series):
            has = self.data.name == self.target
        else:
            try:
                has = self.target in self.data
            except TypeError:
                has = False
        return has

    @property
    def designData(self):
        if hasattr(self, '_designData'):
            output = self._designData
        elif self._has_design:
            train, test, _, __ = self._split()
            output = pd.concat([train, test])
        else:
            output = None
        return output

    @designData.setter
    def designData(self, value):
        self._designData = value

    @property
    def targetData(self):
        if hasattr(self, '_targetData'):
            output = self._targetData
        elif self._has_target:
            _, __, train, test = self._split()
            output = pd.concat([train, test])
        else:
            output = None
        return output

    @targetData.setter
    def targetData(self, value):
        self._targetData = value

    @fallback('queries', 'querier', 'force')
    @fit_fold
    def execute_queries(self, queries=None, querier=None, force=False):
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

    def subset(self, subset='train'):
        subset = subset.capitalize()
        attrs = ['{}{}'.format(attr, subset) for attr in ['design', 'target']]
        X, y = tuple(getattr(self,  attr) for attr in attrs)
        return self.with_params(X=X, y=y)

    def _take(self, train, test):
        results = [self.designData.iloc[train], self.designData.iloc[test]]
        if self.targetData is not None:
            results.extend([self.targetData.iloc[train],
                            self.targetData.iloc[test]])
        else:
            results.extend([None, None])  # Something better here
        return results

    def take(self):
        train, test = next(self.fold.split(self.designData, self.targetData))
        results = self._take(train, test)
        return results

    def with_params(self, raws=None, X=None, y=None, **kwargs):
        kwargs.update({i: j for i, j in self.params.items() if i not in kwargs})
        return type(self)(raws=raws, X=X, y=y, **kwargs)

    def __iter__(self):
        for xTrain, xTest, yTrain, yTest in self.fold.fold(self):
            yield xTrain, xTest, yTrain, yTest
        raise StopIteration


# not a decorator, function for helping data decorators
def pull(wrapped, instance, args, kwargs):
    dataset = find_value(wrapped, args, kwargs, 'dataset')

    if not dataset:
        X = find_value(wrapped, args, kwargs, 'X')
        y = find_value(wrapped, args, kwargs, 'y')
        args, kwargs = replace_value(wrapped, args, kwargs, 'X', None)
        args, kwargs = replace_value(wrapped, args, kwargs, 'y', None)

        if X is None and hasattr(instance, 'dataset'):
            dataset = instance.dataset
        elif X is not None and hasattr(instance, 'dataset')\
            and instance.dataset is not None:
            dataset = Dataset(X=X, y=y, **instance.dataset.param)

        elif X is not None:
            param = dataset.param if dataset else {}
            dataset = Dataset(X=X, y=y, **param)

    if not dataset.hasData and dataset.queries is not None:
        dataset.execute_queries(dataset.queries)
        dataset.fold.fit(dataset)

    args, kwargs = replace_value(wrapped, args, kwargs, 'dataset', dataset)
    return args, kwargs


@decorator
def package_dataset(wrapped, instance, args, kwargs):
    """
    From arguments - package X (and y if supervised) in Data object via type
    """
    args, kwargs = pull(wrapped, instance, args, kwargs)
    result = wrapped(*args, **kwargs)
    return result


@decorator
def enforce_dataset(wrapped, instance, args, kwargs):
    args, kwargs = pull(wrapped, instance, args, kwargs)
    dataset = kwargs['dataset']
    result = wrapped(*args, **kwargs)

    if isinstance(result, pd.np.ndarray):
        features = result.columns.tolist() if hasattr(result, 'columns')\
                else list(instance.get_feature_names()) if hasattr(instance, 'get_feature_names')\
                else instance.fields
        result = pd.DataFrame(result, columns=features, index=dataset.designData.index)

    if not isinstance(result, Dataset):
        if isinstance(result, tuple) and len(result) <= 2:
            result = Dataset(X=result[0], y=result[1] if len(result) > 1 else dataset.targetData,
                             **dataset.params)
        elif isinstance(result, (pd.Series, pd.DataFrame, pd.Panel)):
            result = Dataset(X=result, **dataset.params)
    return result


def subset_dataset(subset):
    @decorator
    def wrapper(wrapped, instance, args, kwargs):
        dataset = find_value(wrapped, args, kwargs, 'dataset').subset(subset)
        args, kwargs = replace_value(wrapped, args, kwargs, 'dataset', dataset)
        result = wrapped(*args, **kwargs)
        return result
    return wrapper


@decorator
def extract_fields(wrapped, instance, args, kwargs):
    result = wrapped(*args, **kwargs)
    instance.features = None
    instance.isFit = True

    dataset = find_value(wrapped, args, kwargs, 'dataset')
    X = find_value(wrapped, args, kwargs, 'X')
    df = dataset.designData if (dataset is not None) else X if (X is not None) else None

    if df is not None:
        instance.fields = list(nvl(*[access(df, [attr], errors='ignore', slicers=()) for attr in ['columns', 'keys']]))
        instance.fieldDtypes = access(df, ['dtypes'], method='to_dict', errors='ignore', slicers=())

    return result


@decorator
def extract_features(wrapped, instance, args, kwargs):
    result = wrapped(*args, **kwargs)
    df = result.designData if isinstance(result, Dataset) else result

    postFit = not instance.features
    if postFit:
        features = df.columns.tolist() if hasattr(df, 'columns')\
            else list(instance.get_feature_names()) if (hasattr(instance, 'get_feature_names')
                    and instance.get_feature_names()) else instance.fields

        instance.features = features

    else:
        features = instance.features
    df = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df, columns=features)

    if postFit:
        instance.featureDtypes = access(df, ['dtypes'], method='to_dict',
                                        slicers=(), errors='ignore')
    return result
