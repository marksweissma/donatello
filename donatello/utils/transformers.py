import re
import pandas as pd

from inspect import getmembers

from sklearn.preprocessing import OneHotEncoder  # ,Imputer,  StandardScaler
from sklearn import clone

import networkx as nx

from donatello.utils.base import Dobject, PandasAttrs, BaseTransformer
from donatello.utils.decorators import init_time, package_data, name, fallback
from donatell.utils.helpers import access


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

    def fit(self, X=None, y=None, **fitParams):
        super(Selector, self).fit(X=X, y=y, **fitParams)

        if self.selectMethod:
            inclusions = getattr(self, self.selectMethod)(X, self.selectValue)
        else:
            inclusions = self.selectValue

        if self.reverse:
            exclusions = set(inclusions)
            inclusions = [i for i in X if i not in exclusions]

        self.inclusions = inclusions

        return self

    def transform(self, X=None, y=None):
        X = X.reindex(columns=self.inclusions)
        return super(Selector, self).transform(X=X, y=y)


class AccessTransformer(PandasTransformer):
    """
    """
    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def transform(self, X=None, y=None):
        X = access(X, **self._kwargs)
        return super(Selector, self).transform(X=X, y=y)


class Node(Dobject):
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

    @package_data
    def transform(self, X=None, y=None, **kwargs):
        if not self.information_available:
            information = pd.concat([transformer.transform(X=X, y=y) for
                                     transformer in self.transformer],
                                    axis=1)
            self.information = information

        return self.information


class ModelDAG(nx.DiGraph):
    @init_time
    @name
    def __init__(self, executor='executor', *args, **kwargs):
        super(ModelDAG, self).__init__(*args, **kwargs)
        self.executor = executor

    def nodes_exec(self, node):
        return self.nodes[node][self.executor]

    def clean(self):
        [self.nodes_exec(node).reset() for node in self]

    @property
    def root(self):
        root = [node for node in self.nodes if self.out_degree(node) == 0 and self.in_degree(node) == 1][0]
        return root

    @fallback('root')
    def fit(self, data, root=None):
        self.clean_graph()
        head = self.root

        parents = tuple(self.succesors(head))
        if len(parents) > 1:
            raise ValueError('Terminal transformation is not unified')

        data = self.apply(data, parents[0], 'fit_transform') if parents else data
        self.nodes_exec(head).fit(data)
        return self

    # replace concat with combine
    def apply(self, node, data, method):
        parents = tuple(self.succesors(node))
        if parents and all([self.nodes_exec(parent).information_available for parent in parents]):
            data = pd.concat([self.nodes_exec(parent).information for parent in parents], axis=1)
        elif parents:

            output = [self.apply(parent,
                                 access(self.get_edge_data(parent, node)[self.executor], method)(data),
                                 method
                                 )
                      for parent in parents]
            data = self.node_exec(node).combine(output)

        information = access(self.nodes_exec(node), method=method, methodArgs=(data))

        return information

    def add_node_transformer(self, node):
        self.add_node(node.name, **{self.executor: node})

    def add_edge_selector(self, node_from, node_to, *args, **kwargs):
        selector = Selector(*args, **kwargs)
        self.add_edge(node_from.name, node_to.name, **{self.executor: selector})
