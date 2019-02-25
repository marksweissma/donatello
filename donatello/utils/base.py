from abc import ABCMeta
from sklearn.base import BaseEstimator, TransformerMixin
from donatello.utils.decorators import coelesce
from donatello.utils.helpers import nvl


class Dobject(object):
    """
    Base object for Donatello - templates in name and __repr___
    """
    __meta__ = ABCMeta

    @coelesce(kargs={})
    def _update_to(self, kargs, *names):
        kargs.update({name: getattr(self, name) for name in names})
        return kargs

    @property
    def name(self):
        name = getattr(self, '_name',  self.__class__.__name__)
        return name

    @name.setter
    def name(self, value):
        self._name = nvl(value,  self.__class__.__name__)

    @property
    def foldClay(self):
        """
        Define type of splitting

            #. None -> KFold
            #. stratify
            #. group
        """
        return getattr(self, '_foldClay', None)

    @foldClay.setter
    def foldClay(self, value):
        self._foldClay = value

    @property
    def scoreClay(self):
        """
        Define type of learning

            #. None -> regression
            #. classificaiton
            #. anomaly
       """
        return getattr(self, '_scoreClay', None)

    @scoreClay.setter
    def scoreClay(self, value):
        self._scoreClay = value

    @property
    def foldDispatch(self):
        """
        """
        return getattr(self, '_foldDispatch', None)

    @foldDispatch.setter
    def foldDispatch(self, value):
        self._foldDispatch = value

    @property
    def scoreDispatch(self):
        """
        """
        return getattr(self, '_scoreDispatch', None)

    @scoreDispatch.setter
    def scoreDispatch(self, value):
        self._scoreDispatch = value

    def __repr__(self):
        name = self.name
        time = getattr(self, '_initTime', '[no init time]')
        rep = "_".join([name, time])
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
        return dataset
