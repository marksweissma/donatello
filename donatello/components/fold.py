from sklearn.model_selection import (KFold,
                                     StratifiedKFold,
                                     GroupShuffleSplit,
                                     TimeSeriesSplit)

from donatello.utils.base import Dobject
from donatello.utils.helpers import access
from donatello.utils.decorators import fallback, init_time, coelesce


_base = {'n_splits': 5,
         'shuffle': True,
         'random_state': 22}


typeDispatch = {None: KFold,
                'stratify': StratifiedKFold,
                'group': GroupShuffleSplit,
                'time': TimeSeriesSplit
                }


kwargDispatch = {None: _base,
                 'stratify': _base,
                 'group': {'n_splits': 5, 'random_state': 22},
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
        self.dap = dap if dap else {}
        self.scoreClay = scoreClay
        self.foldClay = foldClay
        self.folder = typeDispatch.get(foldClay)(**kwargDispatch.get(foldClay))

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
