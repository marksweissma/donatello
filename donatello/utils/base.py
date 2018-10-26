from warnings import warn
from abc import ABCMeta, abstractmethod, abstractproperty
from sklearn.base import BaseEstimator, TransformerMixin


class Dobject(object):
    """
    Base object for Donatello - templates in name and __repr___
    """
    __meta__ = ABCMeta

    @abstractproperty
    def name(self):
        return self.__class__.__name__

    def __repr__(self):
        time = getattr(self, '_initTime', '[no_init_time]')
        return '{name} created at {time}'.format(name=self.name, time=time)


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
    def transformedDtypes(self):
        return getattr(self, '_transformedDtypes', {})

    @transformedDtypes.setter
    def transformedDtypes(self, value):
        """
        Transformed data types
        """
        self._transformedDtypes = value

    def get_fields(self):
        return self.fields

    def get_feature_names(self):
        return self.features


class PandasWrapper(PandasAttrs):
    """
    Object for class factory to bind pandas and scikit-learn
    """
    @property
    def fields(self):
        """
        Incoming column names
        """
        return self.transformerWrapped.fields

    @fields.setter
    def fields(self, value):
        self.transformerWrapped.fields = value

    @property
    def features(self):
        """
        Outgoing column names
        """
        return self.transformerWrapped.features

    @features.setter
    def features(self, value):
        self.transformerWrapped.features = value

    @property
    def transformedDtypes(self):
        return self.transformerWrapped.dtypes

    @transformedDtypes.setter
    def transformedDtypes(self, value):
        self.transformerWrapped.transformedDtypes = value


class BaseTransformer(BaseEstimator, TransformerMixin):
    """
    Base scikit-learn style transformer
    """
    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, y=None, **kwargs):
        return X
