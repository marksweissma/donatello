from sklearn.model_selection import (KFold,
                                     StratifiedKFold,
                                     GroupKFold)

from donatello.utils.helpers import access
from donatello.utils.decorators import fallback, init_time

typeDispatch = {None: KFold,
                'classification': StratifiedKFold,
                'regression': KFold,
                'group': GroupKFold
                }

foldKwargs = {'n_splits': 5,
               'shuffle': True,
               'random_state': 22}


class Splitter(object):
    """
    Object to split data into training and testing/validation groups.
    Packages dataframes and dictionaries of dataframes

    :param str target: name of target field if supervised
    :param str primaryKey: if dictionary of dataframes, key of dictionary\
            containing primrary df
    :param splitOver str: option to split over unique values instead \
            of random or startification
    """
    @init_time
    def __init__(self,
                 target=None,
                 primaryKey=None,
                 splitOver=None,
                 foldKwargs=foldKwargs,
                 typeDispatch=typeDispatch,
                 runTimeAccess=None,
                 mlType=None
                 ):

        self.target = target
        self.primaryKey = primaryKey
        self.splitOver = splitOver
        self.foldKwargs = foldKwargs
        self.typeDispatch = typeDispatch
        self.runTimeAccess = runTimeAccess if runTimeAccess else {}
        self.folder = typeDispatch.get(mlType)(**foldKwargs)
        self.mlType = mlType

    @fallback('target', 'primaryKey')
    def fit(self, data=None, target=None, primaryKey=None, **fitParams):
        """
        fit splitter => finds and store values for each set

        :param donatello.components.data data: data to fit on
        :param str target: str name of target field to separate
        :param str primaryKey: key for primary field (if data.contents \
                is dict (not df)
        :returns: fit transformer
        """
        df = data.contents if not primaryKey else data.contents[primaryKey]

        kwargs = {key: access(df, **value) for key, value in self.runTimeAccess} if self.runTimeAccess else {}

        self.indices = [(trainValues, testValues) for trainValues, testValues
                        in self.folder.split(df.index, df[target], **kwargs)]

        values = df[self.splitOver] if self.splitOver else df.index.to_series()

        self.ids = [(values.iloc[trainValues].values, values.iloc[testValues].values)
                    for (trainValues, testValues) in self.indices]
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

    @fallback('target', 'primaryKey')
    def split(self, data=None, target=None, primaryKey=None, **fitParams):
        """
        Split data contents into design/target train/test/data

        :param donatello.components.data data: data to fit on
        :param str target: str name of target field to separate
        :returns: paylod of train/test/data <> design/target subsets
        :rtype: dict
        """
        df = data.contents[primaryKey] if primaryKey else data.contents
        designData = df.drop(target, axis=1) if target else df

        def _wrap_split(key, content, train, test):
            _designData[key] = content
            _trainMask, _testMask = self._build_masks(content, self.contentMap.get(key, None), train=train, test=test)
            _designTrain[key], _designTest[key] = self._split(content, _trainMask, testMask)
            return _designTrain[key], _designTest[key]

        for train, test in self.ids:
            trainMask, testMask = self._build_masks(df, self.splitOver if self.splitOver else 'index',
                                                    target, train=train, test=test)

            designTrain, designTest = self._split(df, trainMask, testMask, target)

            _designData = {self.primaryKey: designData}
            _designTrain = {self.primaryKey: designTrain}
            _designTest = {self.primaryKey: designTest}

            if isinstance(data.contents, dict):
                for key, content in self.contents.iteritems():
                    if key != self.primaryKey:
                        _designTrain[key], _designTest[key] = _wrap_split(key, content, train, test)

            designData = _designData if None not in _designData else _designData[None]
            designTrain = _designTrain if None not in _designTrain else _designTrain[None]
            designTest = _designTest if None not in _designTest else _designTest[None]

            results = {'designData': designData,
                       'designTrain': designTrain,
                       'designTest': designTest}

            if target:
                targetData = df[target]
                targetTrain, targetTest = self._split(targetData, trainMask, testMask)
            else:
                targetData, targetTrain, targetTest = None, None, None

            results.update({'targetData': targetData,
                            'targetTrain': targetTrain,
                            'targetTest': targetTest
                            })

            yield results
        raise StopIteration

    @fallback('target', 'primaryKey')
    def fit_split(self, data=None, target=None, primaryKey=None, **fitParams):
        self.fit(data=data,target=target, primaryKey=primaryKey, **fitParams)
        return self.split(data=data,target=target, primaryKey=primaryKey, **fitParams)
