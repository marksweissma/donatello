from abc import ABCMeta
from sklearn.base import BaseEstimator, TransformerMixin


class Dobject(object):
    """
    Base object for Donatello - templates in name and __repr___
    """
    __meta__ = ABCMeta

    @property
    def name(self):
        name = getattr(self, '_name',  self.__class__.__name__)
        return name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def mlType(self):
        """
        Define type of learning
            #. regression
            #. classificaiton
            #. clustering
       """
        return getattr(self, '_mlType', None)

    @mlType.setter
    def mlType(self, value):
        self._mlType = value

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


class BaseTransformer(BaseEstimator, TransformerMixin):
    """
    Base scikit-learn style transformer
    """
    def fit(self, X=None, y=None, **kwargs):
        return self

    def transform(self, X=None, y=None, **kwargs):
        return X, y

    @property
    def name(self):
        """
        Name of object, defaults to class name + model name
        """
        _name = self.__class__.__name__
        name = [_name, self.model.__class__.__name__] if hasattr(self, 'model') else [_name]
        return "_".join(name)

    def __repr__(self):
        time = getattr(self, '_initTime', '[no init time]')
        rep = ['{model} created at {time}'.format(model=self.name,
                                                  time=time),
               super(BaseTransformer, self).__repr__()]
        return "\n --- \n **sklearn repr** \n --- \n".join(rep)
