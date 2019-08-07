from abc import ABCMeta
from sklearn.base import BaseEstimator, TransformerMixin
from inflection import underscore
from donatello.utils.helpers import nvl


RANDOM_SEED = 22


class Dobject(object):
    """
    Base object for Donatello - templates in name and __repr___
    """
    __meta__ = ABCMeta

    @property
    def name(self):
        name = getattr(self, '_name', underscore(self.__class__.__name__))
        return name

    @name.setter
    def name(self, value):
        self._name = nvl(value, underscore(self.__class__.__name__))

    @property
    def clay(self):
        """
        Define type of splitting

            #. None -> KFold
            #. stratify
            #. group
        """
        return getattr(self, '_clay', None)

    @clay.setter
    def clay(self, value):
        self._clay = value

    @property
    def initTime(self):
        return getattr(self, '_initTime', '[no init time]')

    @initTime.setter
    def initTime(self, value):
        self._initTime = value

    def __repr__(self):
        name = self.name
        rep = "_".join([name, self.initTime])
        return rep


class PandasAttrs(Dobject):
    """
    Mixin for improving scikit-learn <> pandas interaction
    """
    @property
    def fields(self):
        """
        Incoming column names
        """
        return getattr(self, '_fields', [])

    @fields.setter
    def fields(self, value):
        self._fields = value

    @property
    def features(self):
        """
        Outgoing column names
        """
        return getattr(self, '_features', [])

    @features.setter
    def features(self, value):
        self._features = value

    @property
    def featureDtypes(self):
        return getattr(self, '_featureDtypes', {})

    @featureDtypes.setter
    def featureDtypes(self, value):
        """
        Transformed data types
        """
        self._featureDtypes = value

    def get_fields(self):
        return self.fields

    def get_feature_names(self):
        return self.features


class BaseTransformer(BaseEstimator, TransformerMixin):
    """
    Base scikit-learn style transformer
    """

    def fit(self, X=None, y=None, **kwargs):
        return self

    def transform(self, X=None, y=None, **kwargs):
        output = (X, y) if y is not None else X
        return output

    @property
    def name(self):
        """
        Name of object, defaults to class name + model name
        """
        _name = self.__class__.__name__
        name = [_name, self.model.__class__.__name__] if hasattr(self, 'model') else [_name]
        time = getattr(self, '_initTime', '[no init time]').replace(' ', '_')
        return "_".join(name + [time])

    def __repr__(self):
        time = getattr(self, '_initTime', '[no init time]')
        rep = ['{model} created at {time}'.format(model=self.name,
                                                  time=time),
               super(BaseTransformer, self).__repr__()]
        return "\n --- \n **sklearn repr** \n --- \n".join(rep)


class BaseDatasetTransformer(BaseTransformer):
    """
    Base scikit-learn style transformer
    """

    def fit(self, X=None, y=None, dataset=None, **kwargs):
        return self

    def transform(self, X=None, y=None, dataset=None, **kwargs):
        return dataset

    def fit_transform(self, X=None, y=None, dataset=None, **kwargs):
        self.fit(X=X, y=y, dataset=dataset, **kwargs)
        return self.transform(X=X, y=y, dataset=dataset, **kwargs)
