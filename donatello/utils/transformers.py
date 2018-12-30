import re
import pandas as pd


def warn(arg):
    print(arg)


import inspect

from sklearn.preprocessing import OneHotEncoder, Imputer, StandardScaler
from sklearn.base import TransformerMixin
from sklearn import clone

import networkx as nx

from donatello.utils.base import Dobject, PandasAttrs, BaseTransformer, find_value
from donatello.utils.decorators import init_time, fallback
from donatello.utils.helpers import access, nvl
from donatello.components import data


def _base_methods():
    methods = set([])
    for _type in [BaseTransformer]:
        methods = methods.union(set([i[0] for i in inspect.getmembers(_type)]))
    return methods


base_methods = _base_methods()


def extract_fields(func):
    def wrapped(self, *args, **kwargs):
        X = find_value(func, args, kwargs, accessKey='X')
        self.fields = nvl(*[access(X, [attr], errors='ignore') for attr in ['columns', 'keys']])
        self.fieldDtypes = access(X, ['dtypes'], method='to_dict', errors='ignore')

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

        if not isinstance(result, (pd.DataFrame, pd.Series)):
            try:
                index = kwargs.get('index', kwargs.get('X').index)
            except:
                try:
                    index = args[0].index
                except:
                    self.fields = args[1].index

            result = pd.DataFrame(result, columns=self.features, index=index)

        if postFit:
            self.featureDtypes = result.dtypes.to_dict()
        else:
            result = result.reindex(columns=self.features)

        return result
    return wrapped


class PandasMixin(TransformerMixin, PandasAttrs):
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


class PandasTransformer(BaseTransformer):
    @extract_fields
    def fit(self, *args, **kwargs):
        return super(PandasTransformer, self).fit(*args, **kwargs)

    @enforce_features
    def transform(self, *args, **kwargs):
        return super(PandasTransformer, self).transform(*args, **kwargs)


class OneHotEncoder(PandasMixin, OneHotEncoder):
    pass


class Imputer(PandasMixin, Imputer):
    pass


class StandardScaler(PandasMixin, StandardScaler):
    pass


class KeySelector(PandasTransformer):
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
        super(KeySelector, self).fit(X=X, y=y, **fitParams)

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
        return super(KeySelector, self).transform(X=X, y=y)


class AccessTransformer(PandasTransformer):
    """
    """
    def __init__(self, dap):
        self.dap = dap

    def transform(self, X=None, y=None):
        X = access(X, **self.dap)
        return super(AccessTransformer, self).transform(X=X, y=y)


def concat(datasets, params=None):
    """

    Args:
        datasets (list): list of datasets to combine
        params (dict): params for dataset object, if None infered from first dataset

    Returns:
        data.Dataset: combined dataset
    """
    if datasets:
        Xs = [dataset.designData for dataset in datasets if dataset.designData is not None]
        ys = [dataset.targetData for dataset in datasets if dataset.targetData is not None]

        X = pd.concat(Xs, axis=1) if Xs else None
        y = ys[0] if len(ys) == 1 else None if len(ys) < 1 else pd.concat(ys, axis=1)
        params = params if params is not None else datasets[0].params
        dataset = data.Dataset(X=X, y=y, **params)
    else:
        dataset = None
    return dataset


class TransformNode(Dobject, BaseTransformer):
    def __init__(self, name='transformer', transformers=None, aggregator=None,
                 combine=concat, fitOnly=False):

        self.name = name
        self.transformers = transformers if isinstance(transformers, list) else list(transformers)
        self.aggregator = aggregator
        self.fitOnly = fitOnly
        self.combine = combine

        self.information = None
        self.isFit = False

    @property
    def information_available(self):
        return self.information is not None

    def reset(self):
        self.information = None
        self.isFit = False
        self.transformers = [clone(transformer) for transformer in self.transformers]

    @data.package_dataset
    def fit(self, dataset=None, X=None, y=None, **kwargs):
        [transformer.fit(dataset, **kwargs) for transformer in self.transformers]
        self.isFit = True
        return self

    @data.package_dataset
    def transform(self, dataset=None, X=None, y=None, **kwargs):
        if self.fitOnly:
            output = dataset
        else:
            if not self.information_available:
                information = pd.concat([transformer.transform(X=dataset.designData,
                                                               y=dataset.targetData) for
                                         transformer in self.transformers],
                                        axis=1)
                self.information = information

            output = self.information

        if not isinstance(output, data.Dataset) and isinstance(output, tuple) and len(output) <= 2:
            output = data.Dataset(X=output[0], y=output[1] if len(output) > 1 else dataset.targetData,
                                  **dataset.params)
        else:
            warn('unregisted data return, expecting downstream node contracts to be upheld by users')
        return output


class Garden(nx.DiGraph, Dobject):
    @init_time
    def __init__(self, executor='executor', selectType=KeySelector, *args, **kwargs):
        super(Garden, self).__init__(*args, **kwargs)
        self.executor = executor
        self.selectType = selectType

    def node_exec(self, node):
        return self.nodes[node][self.executor]

    def edge_exec(self, node_to, node_from):
        return self.get_edge_data(node_to, node_from)[self.executor]

    def clean(self):
        [self.node_exec(node).reset() for node in self]

    @property
    def terminal(self):
        terminal = [node for node in self.nodes if self.out_degree(node) == 0]
        terminal = terminal[0] if len(terminal) == 1 else terminal
        return terminal

    @fallback('terminal')
    def transform(self, data, terminal=None):
        head = terminal

        parents = tuple(self.predecessors(head))

        data = self.apply(parents[0], data, 'transform') if parents else data
        transformed = self.node_exec(head).transform(data)
        return transformed

    @fallback('terminal')
    def fit_transform(self, data, terminal=None):
        head = terminal
        parents = tuple(self.predecessors(head))
        print len(parents)
        if len(parents) > 1:
            raise ValueError('Terminal transformation is not unified')

        data = self.apply(parents[0], data, 'fit_transform') if parents else data
        transformed = self.node_exec(head).fit_transform(data)
        return transformed

    @fallback('terminal')
    def fit(self, data, terminal=None):
        self.clean()
        head = terminal

        parents = tuple(self.predecessors(head))
        data = [self.apply(parent, data, 'fit_transform') for parent in parents] if parents else [data]

        # need for to execute edge selection here

        # return subselected data here

        self.node_exec(head).fit(self.node_exec(head).combine(data))
        return self

    def apply(self, node, data, method):
        parents = tuple(self.predecessors(node))

        if parents and all([self.node_exec(parent).information_available for parent in parents]):
            output = [self.node_exec(parent).information for parent in parents]

        elif parents:
            output = [self.apply(parent,
                                 access(self.edge_exec(parent, node), method)(data),
                                 method
                                 ) for
                      parent in parents]

        data = self.node_exec(node).combine(output)

        information = access(self.node_exec(node), method=method, methodArgs=(data,))

        return information

    def add_node_transformer(self, node):
        self.add_node(node.name, **{self.executor: node})

    @fallback('selectType')
    def add_edge_selector(self, node_from, node_to, selectType=None, *args, **kwargs):
        self.add_node_transformer(node_from) if not isinstance(node_from, str) else None
        self.add_node_transformer(node_to) if not isinstance(node_to, str) else None

        selector = selectType(*args, **kwargs)

        self.add_edge(node_from.name, node_to.name, **{self.executor: selector})
