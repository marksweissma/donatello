import inspect
import pandas as pd

from sklearn.model_selection import (KFold,
                                     StratifiedKFold,
                                     GroupShuffleSplit,
                                     TimeSeriesSplit)


from donatello.utils.base import Dobject, RANDOM_SEED
from donatello.utils.decorators import decorator, init_time, fallback, coelesce
from donatello.utils.helpers import find_value, replace_value, nvl, access


_base = {'n_splits': 5,
         'shuffle': True,
         'random_state': RANDOM_SEED}


typeDispatch = {None: KFold,
                'stratify': StratifiedKFold,
                'group': GroupShuffleSplit,
                'time': TimeSeriesSplit
                }


kwargDispatch = {None: _base,
                 'stratify': _base,
                 'group': {'n_splits': 5, 'random_state': RANDOM_SEED},
                 'time': {'n_splits': 5}
                 }


class Fold(Dobject):
    """
    Object to split data into training and testing/validation groups.
    Packages dataframes and dictionaries of dataframes

    Args:
        target (str): name of target field if supervised
        primaryKey (str): if dictionary of dataframes, key of dictionary\
            containing primrary df
        str (splitOver): option to split over unique values instead \
            of random or startification
    """
    @init_time
    @coelesce(dap={})
    def __init__(self, target=None, primaryKey=None,
                 scoreClay=None, foldClay=None,
                 splitDispatch=typeDispatch, kwargDispatch=kwargDispatch,
                 dap=None
                 ):

        self.target = target
        self.primaryKey = primaryKey
        self.scoreClay = scoreClay
        self.foldClay = foldClay
        self.folder = typeDispatch.get(foldClay)(**kwargDispatch.get(foldClay))
        self.dap = dap

    @fallback('target', 'primaryKey', 'dap')
    def fit(self, dataset=None, target=None, primaryKey=None, dap=None, **kwargs):
        """
        fit fold  => finds and store values for each set

        Args:
            dataset (donatello.components.dataset): dataset to fit on
            target (str): str name of target field to separate
            primaryKey (str): key for primary field (if dataset.data \
                is dict not df)

        Returns:
            object: self
        """
        df = dataset.data if not primaryKey else dataset.data[primaryKey]

        groups = access(df, **dap['groups']) if 'groups' in self.dap else None
        condition = target and (target in df)
        self.indices = list(self.split(df.drop(target, axis=1) if condition else df,
                                       df[target] if condition else None,
                                       groups=groups, **kwargs)
                            )

        values = groups if groups is not None else df.index.to_series()

        self.ids = [(values.iloc[trainIndices].values, values.iloc[testIndices].values)
                    for (trainIndices, testIndices) in self.indices]
        return self

    @staticmethod
    def _build_masks(df, key, target=None, train=None, test=None):
        if key is None:
            return [True] * df.shape[0], [True] * df.shape[0]
        elif key == 'index':
            ids = df.index
        else:
            ids = df[key]
        trainMask = ids.isin(train)
        testMask = ids.isin(test)
        return trainMask, testMask

    @staticmethod
    def _split(df, trainMask, testMask, target=None):
        data = df.drop(target, axis=1) if (target and target in df) else df
        train = data.loc[trainMask]
        test = data.loc[testMask]
        return train, test

    def split(self, X, y=None, groups=None, **kwargs):
        kwargs.update({key: access(X, **value) for key, value in self.dap.items()}
                      if self.dap else {})
        # [kwargs.update({i: j}) for i, j in zip(['X', 'y', 'groups'], [X, y, groups]) if j is not None]
        [kwargs.update({i: j}) for i, j in zip(['X', 'y', 'groups'], [X, y, groups])]

        return self.folder.split(**kwargs)

    @fallback('target', 'primaryKey')
    def fold(self, dataset=None, target=None, primaryKey=None):
        """
        Split data data into design/target train/test/data

        Args:
            dataset (donatello.components.dataset): dataset to fit on
            target (str): str name of target field to separate

        Returns:
            dict: paylod of train/test/data <> design/target subsets
        """
        df = dataset.data[primaryKey] if primaryKey else dataset.data

        def _wrap_split(key, data, train, test):
            _trainMask, _testMask = self._build_masks(data, self.dataMap.get(key, None), train=train, test=test)
            _designTrain[key], _designTest[key] = self._split(data, _trainMask, testMask)
            return _designTrain[key], _designTest[key]

        self.fit(dataset) if not hasattr(self, 'ids') else None

        for train, test in self.ids:
            over = self.dap['groups']['attrPath'][0] if 'groups' in self.dap else 'index'
            trainMask, testMask = self._build_masks(df, over, target, train=train, test=test)

            designTrain, designTest = self._split(df, trainMask, testMask, target)

            _designTrain = {self.primaryKey: designTrain}
            _designTest = {self.primaryKey: designTest}

            if isinstance(dataset.data, dict):
                for key, data in self.data.items():
                    if key != self.primaryKey:
                        _designTrain[key], _designTest[key] = _wrap_split(key, data, train, test)

            designTrain = _designTrain if None not in _designTrain else _designTrain[None]
            designTest = _designTest if None not in _designTest else _designTest[None]

            if target and target in df:
                targetTrain, targetTest = self._split(df[target], trainMask, testMask)
            else:
                targetTrain, targetTest = None, None

            results = [designTrain, designTest, targetTrain, targetTest]

            yield results
        raise StopIteration


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
    Maintain and provide access to underlying information and metadata

    Args:
        raw (obj): raw data
        X (obj): option to specify design directly
        y (obj): option to specify target directly
        queries (dict): queries to execute to fetch data if not directly passed
        querier (func): default function to execute queries
        copyRaw (bool): option to have data return copy of raw to preserve fetch
        type (obj) foldType: type of fold to leverage in iterator
        foldDeclaration (obj): kwargs for split type to instantiate with in constructor
    """
    @init_time
    def __init__(self, raw=None, X=None, y=None,
                 queries=None, querier=pd.read_csv, copyRaw=False,
                 foldClay=None, foldType=Fold,
                 scoreClay=None,
                 target=None, primaryKey=None,
                 dap=None, force=False
                 ):

        self.copyRaw = copyRaw
        self.queries = queries
        self.querier = querier

        self.foldClay = foldClay
        self.foldType = foldType

        self.target = target
        self.primaryKey = primaryKey
        self.force = force

        self.fold = foldType(foldClay=foldClay, target=target, primaryKey=primaryKey, dap=dap)

        if any([i is not None for i in [raw, X, y]]):
            self.link(raw, X, y)

    @property
    def dap(self):
        return self.fold.dap

    @property
    def params(self):
        spec = inspect.getargspec(self.__init__)
        exclusions = set(['self', 'raw', 'X', 'y', 'data'])
        params = {param: getattr(self, param) for param in spec.args if param not in exclusions}
        return params

    @property
    def data(self):
        value = getattr(self, '_data', None)
        if value is None:
            if self.copyRaw and self.raw is not None:
                value = self.raw.copy()
            else:
                value = getattr(self, 'raw', None)
        return value

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def hasData(self):
        return has_data(self.data)

    def link(self, raw=None, X=None, y=None):
        if raw is None and (X is not None or y is not None):
            raw = [X] if X is not None else []
            raw.append(y) if y is not None else None
            self.raw = pd.concat(raw, axis=1)
            self.designData = X
            self.targetData = y
            name = getattr(y, 'name', getattr(self, 'target', None))
            self.target = name
        else:
            self.raw = raw

        if has_data(raw) or has_data(X):
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
        Attaches return to :py:attr:`Data.raw`, which can be
        accessed via py:attr:`Data.data`

        Args:
            queries (dict): payload of queries
            querier (func): option to specify executor at the execution\
                level rather than the query level
        """

        if not queries:
            self.raw = querier()
        else:
            for name, query in queries.items():
                querier = query.get('querier', querier)

                payload = {i: j for i, j in query.items() if i != 'querier'}
                if len(queries) == 1:
                    self.raw = querier(**payload)
                else:
                    self.raw[name] = querier(**payload)

    def subset(self, subset='train'):
        """
        Create a new `donatello.components.data.Dataset`
        with a (sub)set of the dataset's data. Either
        by referencing by name (train, test) or passing
        a dap for :py:func:`donatello.utils.helpers.access`

        Args:
            subset (str|dict): attribute to select

        Returns:
            Dataset: with same params as current dataset
        """
        if isinstance(subset, basestring):
            subset = subset.capitalize()
            attrs = ['{}{}'.format(attr, subset) for attr in ['design', 'target']]
            X, y = tuple(getattr(self,  attr) for attr in attrs)
        else:
            X = access(self.designData, **subset)
            y = access(self.targetData, **subset)
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

    def with_params(self, raw=None, X=None, y=None, **kwargs):
        kwargs.update({i: j for i, j in self.params.items() if i not in kwargs})
        return type(self)(raw=raw, X=X, y=y, **kwargs)

    def __iter__(self):
        for xTrain, xTest, yTrain, yTest in self.fold.fold(self):
            yield xTrain, xTest, yTrain, yTest
        raise StopIteration

    def __len__(self):
        return len(self.data if self.data is not None else [])


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
    Package Dataset, and remove X, y from call.
    Iff passed X [y] packgae into dataset,
    if instance has an associated dataset and dataset not passed
    use instance.dataset params
    """
    args, kwargs = pull(wrapped, instance, args, kwargs)
    result = wrapped(*args, **kwargs)
    return result


@decorator
def enforce_dataset(wrapped, instance, args, kwargs):
    """"
    Enforce return is dataset, if X, [y] package into dataset
    """
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
    """
    decorator to susbet dataset passed to function
    """
    @decorator
    def wrapper(wrapped, instance, args, kwargs):
        dataset = find_value(wrapped, args, kwargs, 'dataset').subset(subset)
        args, kwargs = replace_value(wrapped, args, kwargs, 'dataset', dataset)
        result = wrapped(*args, **kwargs)
        return result
    return wrapper


@decorator
def extract_fields(wrapped, instance, args, kwargs):
    """
    Record the column names and/or keys of the incoming dataset
    before a function call
    """
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
    """
    Record the column names and/or keys of the incoming dataset
    after a function call if not already attached to instance
    """
    result = wrapped(*args, **kwargs)
    df = result.designData if isinstance(result, Dataset) else result

    postFit = not getattr(instance, 'features', None)
    if postFit:
        features = df.columns.tolist() if hasattr(df, 'columns')\
            else list(instance.get_feature_names()) if (hasattr(instance, 'get_feature_names')
                    and instance.get_feature_names()) else instance.fields

        instance.features = features

    else:
        features = instance.features
    try:
        df = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df, columns=features)
    except:
        import pdb; pdb.set_trace()

    if postFit:
        instance.featureDtypes = access(df, ['dtypes'], method='to_dict',
                                        slicers=(), errors='ignore')
    return result
