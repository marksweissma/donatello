from sklearn.model_selection import train_test_split
from donatello.utils.base import BaseTransformer
from donatello.utils.helpers import now_string, nvl


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
    :param int _maxStratification: flag to auto stratify
    """
    def __init__(self,
                 target=None,
                 contentKey=None,
                 splitOver=None,
                 stratifyTarget=True,
                 testKwargs={'random_state': 42, 'test_size': .25},
                 attrs=['Train', 'Test', 'Data'],
                 _maxStratification=50,
                 timeFormat="%Y_%m_%d_%H_%M",
                 ):

        self._initTime = now_string(timeFormat)
        self.target = target
        self.contentKey = contentKey
        self.splitOver = splitOver
        self.stratifyTarget = stratifyTarget
        self.testKwargs = testKwargs
        self.attrs = attrs
        self._maxStratification = _maxStratification

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
        target = nvl(target, self.target)


        condition = self.stratifyTarget and len(df[target].unique()) <  self._maxStratification
        self.testKwargs.update({'stratify': df[target]}) if condition else None

        values = df[self.splitOver].unique() if self.splitOver else df.index
        self.trainIds, self.testIds = train_test_split(values, **self.testKwargs)
        return self

    def _split(self, df, key, target=None):
        if key is None:
            return df, df
        elif key == 'index':
            ids = df.index
        else:
            ids = df[key]
        trainMask = ids.isin(self.trainIds)
        testMask = ids.isin(self.testIds)

        _designData = df.drop(target, axis=1) if target else df
        designTrain = _designData.loc[trainMask]
        designTest = _designData.loc[testMask]

        return designTrain, designTest

    # !!!TODO refactor this
    def transform(self, data=None, target=None, **fitParams):
        """
        Split data contents into design/target train/test/data

        :param donatello.components.data data: data to fit on
        :param str target: str name of target field to separate
        :returns: paylod of train/test/data <> design/target subsets
        :rtype: dict
        """
        df = data.contents[self.contentKey] if self.contentKey else data.contents
        target = nvl(target, self.target)
        designData = df.drop(target, axis=1) if target else df
        designTrain, designTest = self._split(df, self.splitOver if self.splitOver else 'index', target)

        _designData = {self.contentKey: designData}
        _designTrain = {self.contentKey: designTrain}
        _designTest = {self.contentKey: designTest}
        if isinstance(data.contents, dict):
            for key, content in self.contents.iteritems():
                if key != self.contentKey:
                    _designData[key] = content
                    _designTrain[key], _designTest[key] = self._split(content, self.contentMap.get(key, None))

        designData = _designData if None not in _designData else _designData[None]
        designTrain = _designTrain if None not in _designTrain else _designTrain[None]
        designTest = _designTest if None not in _designTest else _designTest[None]

        if target:
            targetData = df[target]

            targetTrain, targetTest = self._split(targetData,
                                                  self.splitOver if self.splitOver else 'index',
                                                  None)
        else:
            targetData, targetTrain, targetTest = None, None, None

        results = {'designData': designData,
                   'designTrain': designTrain,
                   'designTest': designTest,
                   'targetData': targetData,
                   'targetTrain': targetTrain,
                   'targetTest': targetTest
                   }

        return results
