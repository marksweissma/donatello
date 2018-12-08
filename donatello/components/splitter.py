from sklearn.model_selection import train_test_split
from donatello.utils.base import BaseTransformer
from donatello.utils.helpers import now_string, nvl
from donatello.utils.decorators import fallback
from sklearn.model_selection import (KFold,
                                     StratifiedKFold,
                                     GroupKFold)

typeDispatch = {None: KFold,
                'classification': StratifiedKFold,
                'regression': KFold,
                'group': GroupKFold
                }

splitKwargs = {'n_splits': 5,
               'shuffle': True,
               'random_state': 22}


class Splitter(BaseTransformer):
    """
    Object to split data into training and testing/validation groups.
    Packages dataframes and dictionaries of dataframes

    :param str target: name of target field if supervised
    :param str contentKey: if dictionary of dataframes, key of dictionary\
            containing primrary df
    :param splitOver str: option to split over unique values instead \
            of random or startification
    :param bool stratifyTarget: option to startify over the target
    :param list attrs: attributes to package for return
    :param dict testKwargs: kwargs for :py:func:`sklearn.model_selection.train_test_split`
    """
    def __init__(self,
                 target=None,
                 contentKey=None,
                 splitKwargs=splitKwargs,
                 typeDispatch=typeDispatch,
                 runTimeAccess=None,
                 attrs=['Train', 'Test', 'Data'],
                 mlType=None,
                 timeFormat="%Y_%m_%d_%H_%M",
                 ):

        self._initTime = now_string(timeFormat)
        self.target = target
        self.contentKey = contentKey
        self.splitKwargs = splitKwargs
        self.typeDispatch = typeDispatch
        self.runTimeAccess = runTimeAccess if runTimeAccess
        self.splitter = typeDispatch.get(mlType)(**splitKwargs)

        self.attrs = attrs
        self.mlType = mlType

    @fallback('target')
    def fit(self, data=None, target=None, contentKey=None, **fitParams):
        """
        fit splitter => finds and store values for each set

        :param donatello.components.data data: data to fit on
        :param str target: str name of target field to separate
        :param str contentKey: key for primary field (if data.contents \
                is dict (not df)
        :returns: fit transformer
        """
        df = data.contents if not self.contentKey else data.contents[self.contentKey]
        values = df[self.splitOver] if self.splitOver else df.index
        kwargs = {key: access(df, **value) for key, value in self.runTimeAccess} if self.runTimeAccess else df.index

        self.ids = [(trainValues, testValues) for trainValues, testValues
                    in self.splitter.split(values, **kwargs)]
        return self

    def _build_masks(self, df, key, target=None, train=None, test=None):
        if key is None:
            return [True] * df.shape[0], [True] * df.shape[0]
        elif key == 'index':
            ids = df.index
        else:
            ids = df[key]
        trainMask = ids.isin(train)
        testMask = ids.isin(test)
        return trainMask, testMask

    def _split(self, df, trainMask, testMask, target=None):
        data = df.drop(target, axis=1) if (target and target in df) else df
        train = data.loc[trainMask]
        test = data.loc[testMask]
        return train, test

    @property
    def split(self):
        return self.transform

    @fallback('target')
    def transform(self, data=None, target=None, **fitParams):
        """
        Split data contents into design/target train/test/data

        :param donatello.components.data data: data to fit on
        :param str target: str name of target field to separate
        :returns: paylod of train/test/data <> design/target subsets
        :rtype: dict
        """
        df = data.contents[self.contentKey] if self.contentKey else data.contents
        designData = df.drop(target, axis=1) if target else df

        for train, test in self.ids:
            trainMask, testMask = self._build_masks(df, self.splitOver if self.splitOver else 'index',
                                                    target, train=train, test=test)

            designTrain, designTest = self._split(df, trainMask, testMask, target)

            _designData = {self.contentKey: designData}
            _designTrain = {self.contentKey: designTrain}
            _designTest = {self.contentKey: designTest}
            if isinstance(data.contents, dict):
                for key, content in self.contents.iteritems():
                    if key != self.contentKey:
                        _designData[key] = content

                        _trainMask, _testMask = self._build_masks(content, self.contentMap.get(key, None), train=train, test=test)
                        _designTrain[key], _designTest[key] = self._split(content, _trainMask, testMask)

            designData = _designData if None not in _designData else _designData[None]
            designTrain = _designTrain if None not in _designTrain else _designTrain[None]
            designTest = _designTest if None not in _designTest else _designTest[None]

            if target:
                targetData = df[target]
                targetTrain, targetTest = self._split(targetData, trainMask, testMask)
            else:
                targetData, targetTrain, targetTest = None, None, None

            results = {'designData': designData,
                       'designTrain': designTrain,
                       'designTest': designTest,
                       'targetData': targetData,
                       'targetTrain': targetTrain,
                       'targetTest': targetTest
                       }

            yield results
        raise StopIteration
