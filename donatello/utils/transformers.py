import re
import pandas as pd

from inspect import getmembers

from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.base import TransformerMixin
from sklearn.pipeline import (Pipeline, FeatureUnion,
                              _fit_transform_one, _transform_one)
from sklearn.externals.joblib import Parallel, delayed

from donatello.utils.base import PandasAttrs, BaseTransformer


def _base_methods():
    methods = set([])
    for _type in [BaseTransformer, Pipeline, FeatureUnion]:
        methods = methods.union(set([i[0] for i in getmembers(_type)]))
    return methods


base_methods = _base_methods()


def extract_fields(func):
    def wrapped(self, *args, **kwargs):
        self.fields = args[0].columns.tolist()
        self.features = None
        result = func(self, *args, **kwargs)
        self.isFit = True
        return result
    return wrapped


def enforce_features(func):
    def wrapped(self, *args, **kwargs):
        result = func(self, *args, **kwargs)

        postFit = not self.features
        if postFit:
            features = result.columns.tolist() if hasattr(result, 'columns')\
                    else self.fields
            self.features = features

        result = result if isinstance(result, pd.DataFrame)\
            else pd.DataFrame(result, columns=self.features,
                              index=kwargs.get('index', args[0].index))

        result = result.reindex(columns=self.features)

        if postFit:
            # DFS to collect data types for feature unions to rm patch
            self.transformedDtypes = result.dtypes.to_dict()

        return result
    return wrapped


class PandasMixin(PandasAttrs):
    """
    Scikit-learn transformer with pandas bindings
    to enforce fields and features
    """
    @extract_fields
    def fit(self, *args, **kwargs):
        return super(PandasMixin, self).fit(*args, **kwargs)

    @enforce_features
    def transform(self, *args, **kwargs):
        return super(PandasMixin, self).transform(*args, **kwargs)

    def fit_transform(self, *args, **kwargs):
        self.fit(*args, **kwargs)
        return self.transform(*args, **kwargs)


class PandasTransformer(PandasMixin, BaseTransformer):
    pass

class Imputer(PandasMixin, Imputer):
    pass


class StandardScaler(PandasMixin, StandardScaler):
    pass


class Pipeline(PandasMixin, Pipeline):
    pass


class FeatureUnion(PandasMixin, FeatureUnion):
    """
    Ripped from sklearn 19.1 to use pandas concat over numpy hstack
    in transform to maintain datatypes
    """
    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit all transformers, transform the data and concatenate results.

        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.

        y : array-like, shape (n_samples, ...), optional
            Targets for supervised learning.

        Returns
        -------
        X_t : array-like or sparse matrix, shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.
        """
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(trans, weight, X, y,
                                        **fit_params)
            for name, trans, weight in self._iter())

        if not result:
            # All transformers are None
            return pd.np.zeros((X.shape[0], 0))
        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        Xs = pd.concat(Xs, axis=1)
        return Xs

    def transform(self, X):
        """
        Transform X separately by each transformer, concatenate results.

        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.

        Returns
        -------
        X_t : array-like or sparse matrix, shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.
        """
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, weight, X)
            for name, trans, weight in self._iter())
        if not Xs:
            # All transformers are None
            return pd.np.zeros((X.shape[0], 0))
        else:
            Xs = pd.concat(Xs, axis=1)
        return Xs


class Selector(PandasTransformer):
    """
    Select subset of columns from keylike-valuelike store

    :param obj selectValue: values used for selection
    :param str selectMethod: type of selection
            #. None / '' -> direct key /value look up (i.e. column names to\
                    slice with)
            # 'data_type' -> uses :py:meth:`pandas.DataFrame.select_dtypes`\
                    to select by data type.
    :param bool reverse: option to select all except those fields isolated\
            by selectValue and selectMethod
    """
    def __init__(self, selectValue=(), selectMethod=None, reverse=False):
        self.selectMethod = selectMethod
        self.selectValue = selectValue
        self.reverse = reverse

    def data_type(self, X, inclusionExclusionKwargs):
        inclusions = X.select_dtypes(**inclusionExclusionKwargs
                                     ).columns.tolist()
        return inclusions

    def regex(self, X, patterns):
        inclusions = [i for i in X if any([re.match(j, i) for j in patterns])]
        return inclusions

    @extract_fields
    def fit(self, X, y=None):
        if self.selectMethod:
            inclusions = getattr(self, self.selectMethod)(X, self.selectValue)
        else:
            inclusions = self.selectValue

        if self.reverse:
            exclusions = set(inclusions)
            inclusions = [i for i in X if i not in exclusions]

        self.inclusions = inclusions

        return self

    @enforce_features
    def transform(self, X, y=None):
        return X.reindex(columns=self.inclusions)


class CategoricalTransformer(PandasTransformer):
    """
    One hot encoder for enumerated string typed fields.

    :param bool dropFirst: option to drop first value from each column
    """
    def __init__(self, dropFirst=False):
        self.dropFirst = dropFirst

    def fit(self, X, y=None):
        self.categories = {field: [] for field in self.fields}
        for field in self.categories:
            self.categories[field] = X.loc[X[field].notnull()][field]\
                    .unique().tolist()
        return self

    def transform(self, X, y=None):
        for field in set(self.fields).intersection(X):
            X[field] = pd.Series(X[field], dtype='category').cat.\
                    set_categories(self.categories[field])
        X = pd.get_dummies(X, columns=self.fields,
                           drop_first=self.dropFirst)

        return X


class AttributeTransformer(PandasTransformer):
    """
    Transformer leveraing attribute methods of the design object
    """
    def __init__(self, attribute=None, args=(), kwargs={}):
        self.attribute = attribute
        self.args = args
        self.kwargs = kwargs

    def transform(self, X, **fitParams):
        X = getattr(X, self.attribute)(*self.args, **self.kwargs)
        return X


class CallbackTransformer(PandasTransformer):
    """
    Transformer to apply call back on design object
    """
    def __init__(self, callback=None, args=(), kwargs={}):
        self.callback
        self.args = args
        self.kwargs = kwargs

    def transform(self, X, **fitParams):
        X = self.callback(X, *self.args, **self.kwargs)
        return X


_selectNumbers = Selector(selectMethod='data_type',
                          selectValue={'include': [pd.np.number]})
_selectObjects = Selector(selectMethod='data_type',
                          selectValue={'include': [object]})

_selectNotAmounts = Selector(selectMethod='regex',
                             selectValue='_amount', reverse=True)
_selectAmounts = Selector(selectMethod='regex', selectValue='_amount')

_zeroFill = AttributeTransformer('fillna', (0,))


def load_simple_numeric(select=_selectNumbers,
                        fill=_zeroFill, scaler=StandardScaler()):
    steps = [('select_numeric', select)]
    steps.append(('fill_zero', fill)) if fill else None
    steps.append(('scale', scaler)) if scaler else None

    transformer = Pipeline(steps=steps)
    return transformer


def load_simple_categories(dropFirst=False):
    steps = [('select_objects', _selectObjects),
             ('dummify', CategoricalTransformer(dropFirst=dropFirst))
             ]
    transformer = Pipeline(steps=steps)
    return transformer


def load_num_str_split(numeric=load_simple_numeric(),
                       strings=load_simple_categories()):
    features = FeatureUnion([('numeric', numeric), ('strings', strings)])
    return features


def load_basic_transformer(drops=(), features=load_num_str_split()):
    steps = [('drops', Selector(selectValue=drops, reverse=True)),
             ('features', features)
             ]
    transformer = Pipeline(steps=steps)
    return transformer
