import inspect
import pandas as pd

from sklearn.model_selection import (KFold,
                                     StratifiedKFold,
                                     GroupShuffleSplit,
                                     TimeSeriesSplit)


from donatello.utils.base import Dobject, RANDOM_SEED
from donatello.utils.decorators import decorator, init_time, fallback, coelesce
from donatello.utils.helpers import find_value, replace_value, access


_BASE = {'n_splits': 5,
         'shuffle': True,
         'random_state': RANDOM_SEED}


TYPE_DISPATCH = {None: KFold,
                 'stratify': StratifiedKFold,
                 'group': GroupShuffleSplit,
                 'time': TimeSeriesSplit
                 }


KWARG_DISPATCH = {None: _BASE,
                  'stratify': _BASE,
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
        clay (str): option to set folding options through dispatch
        splitDispatch(dict): Folding Types keyed by clays
        kwargDispatch(dict): keyword arguments conjugate to folding types
        groupDap (str): option add runtime kwargs to folding and split (i.e. groups)
    """
    @init_time
    @coelesce(groupDap={}, dataMap={})
    def __init__(self, target=None, primaryKey=None, clay=None,
                 splitDispatch=TYPE_DISPATCH, kwargDispatch=KWARG_DISPATCH,
                 groupDap=None, dataMap=None
                 ):

        self.target = target
        self.primaryKey = primaryKey
        self.clay = clay
        self.folder = splitDispatch.get(clay)(**kwargDispatch.get(clay))
        self.groupDap = groupDap
        self.dataMap = dataMap

    @fallback('target', 'primaryKey', 'groupDap')
    def fit(self, dataset=None, target=None, primaryKey=None, groupDap=None, **kwargs):
        """
        fit fold  => finds and store values for each set

        Args:
            dataset (donatello.components.dataset): dataset to fit on
            target (str): str name of target field to separate
            primaryKey (str): key for primary field (if dataset.data \
                is dict not df)
            groupDap (dict): payload to specify groups through access

        Returns:
            object: self
        """
        df = dataset.designData if not primaryKey else dataset.designData[primaryKey]

        groups = access(df, **groupDap) if groupDap else None

        self.indices = list(self.split(df,
                                       dataset.targetData,
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

    def split(self, X, y=None, groups=None):
        return self.folder.split(X=X, y=y, groups=groups)

    def _wrap_split(self, key, data, train, test):
        _trainMask, _testMask = self._build_masks(
            data, self.dataMap.get(key, None), train=train, test=test)

        train, test = self._split(data, _trainMask, _testMask)

        return train, test

    def _package_fold(self, dataset, df, train, test, target, groupDap):
        # move build_masks to access
        over = groupDap['attrPath'][0] if groupDap else 'index'
        trainMask, testMask = self._build_masks(df, over, target, train=train, test=test)

        designTrain, designTest = self._split(df, trainMask, testMask, target)

        if isinstance(dataset.data, dict):
            designTrain = {self.primaryKey: designTrain}
            designTest = {self.primaryKey: designTest}
            for name, data in dataset.data.items():
                if name != self.primaryKey:
                    key = self.dataMap.get(name, None)
                    designTrain[name], designTest[name] = self._wrap_split(key, data, train, test)

        if target and target in df:
            targetTrain, targetTest = self._split(df[target], trainMask, testMask)
        else:
            targetTrain, targetTest = None, None

        results = (designTrain, designTest, targetTrain, targetTest)
        return results

    @fallback('target', 'primaryKey', 'groupDap')
    def fold(self, dataset=None, target=None, primaryKey=None, groupDap=None):
        """
        Split data data into design/target train/test/data

        Args:
            dataset (donatello.components.dataset): dataset to fit on
            target (str): str name of target field to separate

        Returns:
            dict: paylod of train/test/data <> design/target subsets
        """
        self.fit(dataset)

        df = dataset.data[primaryKey] if primaryKey else dataset.data
        for train, test in self.ids:
            results = self._package_fold(dataset, df, train, test, target, groupDap)
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


class Dataset(Dobject):
    """
    Maintain and provide access to underlying information and metadata

    Args:
        raw (obj): raw data
        X (obj): option to specify design directly
        y (obj): option to specify target directly
        copyRaw (bool): option to have data return copy of raw to preserve fetch
        clay (str): option to set fodling options through dispatch
        foldType(type): type of fold object
        groupDap (dict): groupDap for Folder
    """
    @init_time
    @coelesce(dataMap={})
    def __init__(self, raw=None, X=None, y=None,
                 copyRaw=False,
                 clay=None, foldType=Fold,
                 target=None, primaryKey=None,
                 groupDap=None, dataMap=None
                 ):

        self.copyRaw = copyRaw
        self.raw = raw

        self.clay = clay
        self.foldType = foldType

        self.target = target if target else getattr(y, 'name', None)
        self.primaryKey = primaryKey
        self.dataMap = dataMap

        self.fold = foldType(clay=clay, target=self.target, primaryKey=primaryKey, groupDap=groupDap,
                             dataMap=dataMap)

        if any([i is not None for i in [raw, X, y]]):
            self.link(raw, X, y)

    @property
    def groupDap(self):
        return self.fold.groupDap

    @property
    def params(self):
        """
        Params of object excluding those that hold the data/information to operate on
        (raw, data, X, y)
        """
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
                value = self.raw
        return value

    @property
    def hasData(self):
        return has_data(self.data)

    def _link_df(self, X, y):
        raw = [X] if X is not None else []
        raw.append(y) if y is not None else None
        self.raw = pd.concat(raw, axis=1)

    def _link_dfs(self, X, y):
        XDf = X[self.primaryKey]
        primary = [XDf]
        primary.append(y) if y is not None else None
        primary = pd.concat(primary, axis=1)
        X[self.primaryKey] = primary
        self.raw = X

    def link(self, raw=None, X=None, y=None):
        if raw is None and (X is not None or y is not None):
            if X is not None:
                self._designData = X
            if y is not None:
                self._targetData = y

            name = getattr(y, 'name', getattr(self, 'target', None))
            self.target = name

            self._link_dfs(X, y) if isinstance(X, dict) else self._link_df(X, y)

        else:
            self.raw = raw

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
                _df = self.data[self.primaryKey] if isinstance(self.data, dict) else self.data
                has = self.target in _df
            except TypeError:
                has = False
        return has

    @property
    def designData(self):
        # if hasattr(self, '_designData'):
            # output = self._designData
        # elif self._has_design:
        try:
            if isinstance(self.data, dict):
                primary = self.data[self.primaryKey].drop(self.target, axis=1, errors='ignore')
                output = {self.primaryKey: primary}
                [output.update({key: value}) for key, value in self.data.items() if key != self.primaryKey]
            else:
                if self.target:
                    output = self.data.drop(self.target, axis=1, errors='ignore')
                else:
                    output = self.data

        # else:
        except:
            output = None
            import ipdb; ipdb.set_trace()
        return output

    @designData.setter
    def designData(self, value):
        self._designData = value
        # use reassign
        self.link(X=value, y=self.targetData)

    @property
    def targetData(self):
        # if hasattr(self, '_targetData'):
            # output = self._targetData
        # elif self._has_target:
        try:
            if isinstance(self.data, dict):
                output = self.data[self.primaryKey][self.target]
            else:
                output = self.data[self.target]
        # else:
        except:
            output = None
        return output

    @targetData.setter
    def targetData(self, value):
        self._targetData = value
        self.link(X=self.designData, y=value)

    @property
    def designTrain(self):
        return self.take()[0]

    @property
    def targetTrain(self):
        return self.take()[2]

    @property
    def designTest(self):
        return self.take()[1]

    @property
    def targetTest(self):
        return self.take()[3]

    def subset(self, subset='train'):
        """
        Create a new `donatello.components.data.Dataset`
        with a (sub)set of the dataset's data. Either
        by referencing by name (train, test) or passing
        a payload for :py:func:`donatello.utils.helpers.access`

        Args:
            subset (str|dict): attribute to select

        Returns:
            Dataset: with same params as current dataset
        """
        if isinstance(subset, basestring):
            subset = subset.capitalize()
            attrs = ['{}{}'.format(attr, subset) for attr in ['design', 'target']]
            X, y = tuple(getattr(self, attr) for attr in attrs)
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

    def with_params(self, raw=None, X=None, y=None, safe=True, **kwargs):
        kwargs = kwargs.copy()
        kwargs.update({i: j for i, j in self.params.items() if i not in kwargs})
        # if safe and (isinstance(raw, dict) or isinstance(X, dict)):
            # kwargs = {i: j for i, j in kwargs.items() if i not in ['primaryKey', 'dataMap']}

        return type(self)(raw=raw, X=X, y=y, **kwargs)

    @property
    def shape(self):
        return self.data.shape if hasattr(self.data, 'shape') else len(self.data)\
            if hasattr(self.data, '__len__') else None

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
        if X is not None:
            args, kwargs = replace_value(wrapped, args, kwargs, 'X', None)
        if y is not None:
            args, kwargs = replace_value(wrapped, args, kwargs, 'y', None)

        if X is None and hasattr(instance, 'dataset'):
            dataset = instance.dataset
        elif X is not None and hasattr(instance, 'dataset')\
                and instance.dataset is not None:
            dataset = Dataset(X=X, y=y, **instance.dataset.params)

        elif X is not None:
            param = dataset.param if dataset else {}
            dataset = Dataset(X=X, y=y, **param)

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
        instance.fields = access(df,  errors='ignore', cb=list)
        # Need dfs to find schema
        instance.fieldDtypes = access(df, ['dtypes'], method='to_dict', errors='ignore', slicers=())

    return result


@decorator
def extract_features(wrapped, instance, args, kwargs):
    """
    Record the column names and/or keys of the outgoing dataset
    after a function call if not already attached to instance
    """
    dataset = find_value(wrapped, args, kwargs, 'dataset')
    X = find_value(wrapped, args, kwargs, 'X')
    initial = dataset.designData if (dataset is not None) else X if (X is not None) else None
    index = initial.index if hasattr(initial, 'index') else None

    result = wrapped(*args, **kwargs)
    df = result.designData if isinstance(result, Dataset) else result

    postFit = not getattr(instance, 'features', None)
    if postFit and df is not None:
        features = df.columns.tolist() if hasattr(df, 'columns') else list(instance.get_feature_names()) if (
            hasattr(instance, 'get_feature_names') and instance.get_feature_names()) else instance.fields

        instance.features = features

    else:
        features = getattr(instance, 'features', [])
    df = df.reindex(features) if isinstance(
        df, pd.DataFrame) else pd.DataFrame(
        df, columns=features, index=index)

    if postFit:
        instance.featureDtypes = access(df, ['dtypes'], method='to_dict',
                                        slicers=(), errors='ignore')
    return result
