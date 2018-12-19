import re
import pandas as pd

from inspect import getmembers

from sklearn.preprocessing import OneHotEncoder  # ,Imputer,  StandardScaler
from sklearn import clone

from donatello.utils.base import PandasAttrs, BaseTransformer
from donatello.utils.decorators import init_time, package_data, name


def _base_methods():
    methods = set([])
    for _type in [BaseTransformer]:
        methods = methods.union(set([i[0] for i in getmembers(_type)]))
    return methods


base_methods = _base_methods()


def extract_fields(func):
    def wrapped(self, *args, **kwargs):
        try:
            self.fields = kwargs.get('X', pd.DataFrame()).columns.tolist()
        except:
            try:
                self.fields = args[0].columns.tolist()
            except:
                self.fields = args[1].columns.tolist()

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
                    else list(self.get_feature_names()) if hasattr(self, 'get_feature_names')\
                    else self.fields
            self.features = features
        try:
            index = kwargs.get('index', kwargs.get('X').index)
        except:
            try:
                index = args[0].index
            except:
                self.fields = args[1].index

        result = result if isinstance(result, pd.DataFrame)\
            else pd.DataFrame(result, columns=self.features,
                              index=index)

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

    @extract_fields
    @enforce_features
    def fit_transform(self, *args, **kwargs):
        return super(PandasMixin, self).fit_transform(*args, **kwargs)


class PandasTransformer(PandasMixin, BaseTransformer):
    pass


class OneHotEncoder(PandasMixin, OneHotEncoder):
    pass


class Selector(PandasTransformer):
    """
    Select subset of columns from keylike-valuelike store

    Args:
        selectValue (obj): values used for selection
        selectMethod (str): type of selection
            #. None / '' -> direct key /value look up (i.e. column names to\
                    slice with)
            # 'data_type' -> uses :py:meth:`pandas.DataFrame.select_dtypes`\
                    to select by data type.
        reverse (bool): option to select all except those fields isolated\
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
    def fit(self, X=None, y=None):
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
    def transform(self, X=None, y=None):
        return X.reindex(columns=self.inclusions)


class AccessTransformer(PandasTransformer):
    """
    """
    def __init__(self, attribute=None, args=(), kwargs={}):
        self.attribute = attribute
        self.args = args
        self.kwargs = kwargs

    def transform(self, X=None, **fitParams):
        X = getattr(X, self.attribute)(*self.args, **self.kwargs)
        return X


class Node(object):
    @init_time
    @name
    def __init__(self, name='transformer', transformers=None, aggregator=None):
        self.name = name
        self.transformers = transformers
        self.aggregator = aggregator
        self.information = None

    @property
    def information_available(self):
        return self.information is not None

    def reset(self):
        self.information = None
        self.transformers = [clone(transformer) for transformer in self.transformers]

    @package_data
    def fit(self, dataset=None, X=None, y=None, **kwargs):
        [transformer.fit(X=dataset.designData, y=dataset.targetData, **kwargs) for transformer in self.transformers]
        return self

    def transform(self, X=None, y=None, **kwargs):
        if not self.information_available:
            information = pd.concat([transformer.transform(X=X, y=y) for
                                     transformer in self.transformer],
                                    axis=1)
            self.information = information

        return self.information


class TransformationDAG(object):
    @name
    def __init__(self, graph=None, attr='executor'):
        self.graph = graph
        self.attr = attr

    def clean(self):
        [self.graph.nodes[node][self.attr].reset() for node in self.graph]

    def flush(self, node, data):
        self.clean_graph()
        parents = tuple(self.graph.succesors(node))
        if not parents:
            information = self.graph.nodes[node][self.attr].transform(data)
        elif all([self.graph.nodex[parent][self.attr].information_available for parent in parents]):
            fields = [self.graph.nodes[parent][self.attr].information for parent in parents]
            information = self.graph.nodes[node][self.attr].fit_transform(fields)
        else:
            information = self.graph.nodes[node][self.attr].fit_transform([self.flush(parent, data)
                                                                           for parent in parents])

        return information
