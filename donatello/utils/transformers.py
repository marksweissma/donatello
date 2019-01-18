import re
import pandas as pd
import networkx as nx

from wrapt import decorator

from sklearn.preprocessing import OneHotEncoder, Imputer, StandardScaler
from sklearn.base import TransformerMixin
from sklearn import clone


from donatello.utils.base import Dobject, PandasAttrs, BaseTransformer
from donatello.utils.decorators import init_time, fallback
from donatello.utils.helpers import access, nvl, find_value
from donatello.components import data


@decorator
def extract_fields(wrapped, instance, args, kwargs):
    result = wrapped(*args, **kwargs)
    dataset = find_value(wrapped, args, kwargs, 'dataset')

    instance.fields = list(nvl(*[access(dataset, ['designData', attr], errors='ignore', slicers=())
                            for attr in ['columns', 'keys']]))
    instance.fieldDtypes = access(dataset, ['designData', 'dtypes'],
                                  method='to_dict', errors='ignore', slicers=())

    instance.features = None
    instance.isFit = True
    return result


@decorator
def extract_features(wrapped, instance, args, kwargs):
    result = wrapped(*args, **kwargs)

    postFit = not instance.features
    if postFit:
        df = result.designData
        features = df.columns.tolist() if hasattr(df, 'columns')\
                    else list(df.get_feature_names()) if hasattr(instance, 'get_feature_names')\
                    else instance.fields
        instance.features = features
        instance.featureDtypes = access(result, ['designData', 'dtypes'], method='to_dict',
                                        slicers=(), errors='ignore')

    return result


class PandasMixin(TransformerMixin, PandasAttrs):
    """
    Scikit-learn transformer with pandas bindings
    to enforce fields and features
    """
    @data.package_dataset
    @extract_fields
    def fit(self, *args, **kwargs):
        return super(PandasMixin, self).fit(*args, **kwargs)

    @data.enforce_dataset
    @extract_features
    def transform(self, *args, **kwargs):
        return super(PandasMixin, self).transform(*args, **kwargs)


class PandasTransformer(BaseTransformer):
    @data.package_dataset
    @extract_fields
    def fit(self, X=None, y=None, dataset=None, *args, **kwargs):
        return self

    @data.enforce_dataset
    @extract_features
    def transform(self, X=None, dataset=None, *args, **kwargs):
        return dataset

    def fit_transform(self, X=None, y=None, dataset=None, *args, **kwargs):
        self.fit(X=X, y=y, dataset=dataset, *args, **kwargs)
        return self.transform(X=X, dataset=dataset, *args, **kwargs)


class TargetConductor(BaseTransformer):
    def __init__(self, passTarget):
        self.passTarget = passTarget

    @data.enforce_dataset
    def transform(self, dataset=None, **kwargs):
        return dataset.targetData if self.passTarget else None


class DesignConductor(PandasTransformer):
    """
    Select subset of columns from keylike-valuelike store

    Args:
        selectValue (obj): values used for selection
        selectMethod (str): type of selection

            #. None / '' -> direct key /value look up (i.e. column names to slice with)
            # 'data_type' -> uses :py:meth:`pandas.DataFrame.select_dtypes` to select by data type.

        reverse (bool): option to select all except those fields isolated\
            by selectValue and selectMethod
    """
    def __init__(self, selectValue=(), selectMethod=None, reverse=False):
        self.selectValue = selectValue
        self.selectMethod = selectMethod
        self.reverse = reverse

    def data_type(self, X, inclusionExclusionKwargs):
        X = X.select_dtypes(**inclusionExclusionKwargs).columns.tolist()
        return X

    def regex(self, X, patterns):
        inclusions = [i for i in X if any([re.match(j, i) for j in patterns])]
        return inclusions

    @data.package_dataset
    def fit(self, dataset=None, **fitParams):

        if self.selectMethod:
            inclusions = getattr(self, self.selectMethod)(dataset.designData, self.selectValue)
        else:
            inclusions = self.selectValue

        if self.reverse:
            exclusions = set(inclusions)
            inclusions = [i for i in dataset.designData if i not in exclusions]

        self.inclusions = inclusions

        return self

    @data.enforce_dataset
    def transform(self, dataset=None, **kwargs):
        dataset.designData = dataset.designData.reindex(columns=self.inclusions)
        return super(DesignConductor, self).transform(dataset)


class DatasetConductor(BaseTransformer):
    def __init__(self,
                 selectValue=(), selectMethod=None, reverse=False,
                 passTarget=False):
        self.selectValue = selectValue
        self.selectMethod = selectMethod
        self.reverse = reverse
        self.passTarget = passTarget

    def data_type(self, X, inclusionExclusionKwargs):
        X = X.select_dtypes(**inclusionExclusionKwargs).columns.tolist()
        return X

    def regex(self, X, patterns):
        inclusions = [i for i in X if any([re.match(j, i) for j in patterns])]
        return inclusions

    def fit_design(self, dataset, **fitParams):
        if self.selectMethod:
            inclusions = getattr(self, self.selectMethod)(dataset.designData, self.selectValue)
        else:
            inclusions = self.selectValue

        if self.reverse:
            exclusions = set(inclusions)
            inclusions = [i for i in dataset.designData if i not in exclusions]

        self.inclusions = inclusions

    def transform_target(self, dataset):
        return dataset.targetData if self.passTarget else None

    def transform_design(self, dataset):
        design = dataset.designData.reindex(columns=self.inclusions)
        return design

    @extract_fields
    def fit(self, dataset=None, **fitParams):
        self.fit_design(dataset, **fitParams)
        return self

    @extract_features
    def transform(self, dataset=None, **fitParams):
        design = self.transform_design(dataset)
        target = self.transform_target(dataset)
        dataset = dataset.with_params(X=design, y=target)
        return dataset


class AccessTransformer(BaseTransformer):
    """
    """
    def __init__(self, dap):
        self.dap = dap

    @data.package_dataset
    @data.enforce_dataset
    @extract_features
    def transform(self, dataset=None, X=None, y=None):
        dataset.designData = access(dataset.designData, **self.dap)
        return super(AccessTransformer, self).transform(X=dataset.designData, y=y)


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
        params = nvl(params, datasets[0].params)
        dataset = data.Dataset(X=X, y=y, **params)
    else:
        dataset = None
    return dataset


class TransformNode(Dobject, BaseTransformer):
    def __init__(self, transformer=None, aggregator=None,
                 combine=concat, fitOnly=False, name=None):

        self.name = name
        self.transformer = transformer
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
        self.transformer = clone(self.transformer)

    @data.package_dataset
    def fit(self, dataset=None, X=None, y=None, **kwargs):
        self.transformer.fit(X=dataset.designData, y=dataset.targetData, **kwargs)
        self.isFit = True
        return self

    @data.package_dataset
    def transform(self, dataset=None, X=None, y=None, **kwargs):
        if not self.information_available:
            self.information = self._transform(dataset=None, X=None, y=None, **kwargs)
        return self.information

    @data.enforce_dataset
    def _transform(self, dataset=None, X=None, y=None, **kwargs):
        if self.fitOnly:
            output = dataset
        else:
            output = self.transformer.transform(X=dataset.designData, y=dataset.targetData)

        return output


class ModelDAG(nx.DiGraph, Dobject):
    @init_time
    def __init__(self, executor='executor', conductionType=DatasetConductor, *args, **kwargs):
        super(ModelDAG, self).__init__(*args, **kwargs)
        self.executor = executor
        self.conductionType = conductionType

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

    @fallback(node='terminal')
    def transform(self, data, node=None):
        parents = tuple(self.predecessors(node))
        # iterate through nodes => terminal_list to list
        data = self.apply(parents[0], data, 'transform') if parents else data
        transformed = self.node_exec(node).transform(data)
        return transformed

    @fallback(node='terminal')
    def fit(self, data, node=None):
        self.clean()
        # iterate through nodes => terminal_list to list
        parents = tuple(self.predecessors(node))
        if parents:
            upstreams = [self.apply(parent, data, 'fit_transform') for parent in parents]
            output = [self.edge_exec(parent, node).fit_transform(upstream)
                      for parent, upstream in zip(parents, upstreams)]

            data = self.node_exec(node).combine(output)

        self.node_exec(node).fit(data)

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
        else:
            output = [data]

        data = self.node_exec(node).combine(output)

        information = access(self.node_exec(node), method=method, methodArgs=(data,))

        return information

    def add_node_transformer(self, node):
        self.add_node(node.name, **{self.executor: node})

    @fallback('conductionType')
    def add_edge_selector(self, node_from, node_to, conductionType=None, *args, **kwargs):
        self.add_node_transformer(node_from) if not isinstance(node_from, str) else None
        self.add_node_transformer(node_to) if not isinstance(node_to, str) else None

        conductor = conductionType(*args, **kwargs)
        node_from = node_from.name if not isinstance(node_from, str) else node_from
        node_to = node_to.name if not isinstance(node_to, str) else node_to

        self.add_edge(node_from, node_to, **{self.executor: conductor})


class OneHotEncoder(PandasMixin, OneHotEncoder):
    pass


class Imputer(PandasMixin, Imputer):
    pass


class StandardScaler(PandasMixin, StandardScaler):
    pass
