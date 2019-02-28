import re
import inspect
import pandas as pd
import networkx as nx

from sklearn.preprocessing import OneHotEncoder, Imputer, StandardScaler
from sklearn.base import TransformerMixin
from sklearn import clone


from donatello.utils.base import Dobject, PandasAttrs, BaseTransformer
from donatello.utils.decorators import fallback
from donatello.utils.helpers import access, nvl, now_string
from donatello.components import data


class PandasMixin(TransformerMixin, PandasAttrs):
    """
    Scikit-learn transformer with pandas bindings
    to enforce fields and features
    """
    @data.package_dataset
    @data.extract_fields
    def fit(self, X=None, y=None, dataset=None, *args, **kwargs):
        return super(PandasMixin, self).fit(X=dataset.designData, y=dataset.targetData, *args, **kwargs)

    @data.enforce_dataset
    @data.extract_features
    def transform(self, X=None, dataset=None, *args, **kwargs):
        return super(PandasMixin, self).transform(X=dataset.designData, *args, **kwargs)


class PandasTransformer(BaseTransformer):
    @data.package_dataset
    @data.extract_fields
    def fit(self, X=None, y=None, dataset=None, *args, **kwargs):
        return self

    @data.enforce_dataset
    @data.extract_features
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
                 passTarget=True, passDesign=True):
        self.selectValue = selectValue
        self.selectMethod = selectMethod
        self.reverse = reverse
        self.passTarget = passTarget
        self.passDesign = passDesign

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
        design = dataset.designData.reindex(columns=self.inclusions) if self.passDesign else None
        return design

    @data.extract_fields
    def fit(self, dataset=None, **fitParams):
        self.fit_design(dataset, **fitParams)
        return self

    @data.extract_features
    def transform(self, dataset=None, **fitParams):
        design = self.transform_design(dataset)
        target = self.transform_target(dataset)
        dataset = dataset.with_params(X=design, y=target)
        return dataset

    def fit_transform(self, dataset=None, *args, **kwargs):
        self.fit(dataset=dataset, *args, **kwargs)
        return self.transform(dataset=dataset, *args, **kwargs)


class AccessTransformer(BaseTransformer):
    """
    """
    def __init__(self, dap):
        self.dap = dap

    @data.package_dataset
    @data.enforce_dataset
    @data.extract_features
    def transform(self, X=None, y=None, dataset=None):
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
    """
    Node in model execution grap
    """
    def __init__(self, name=None, transformer=None, aggregator=None,
                 combine=concat, fitOnly=False, store=True):

        self.name = name
        self.transformer = transformer
        self.aggregator = aggregator
        self.fitOnly = fitOnly
        self.combine = combine
        self.store = store

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
    def fit(self, X=None, y=None, dataset=None, **kwargs):
        print('fitting {}'.format(self.name))
        spec = inspect.getargspec(self.transformer.fit)
        if 'dataset' in spec.args:
            payload = {'dataset': dataset}
        else:
            payload = {'X': dataset.designData}
            payload.update({'y': dataset.targetData}) if 'y' in spec.args else None
        payload.update(kwargs)
        self.transformer.fit(**payload)

        self.isFit = True
        return self

    @data.package_dataset
    def transform(self, X=None, y=None, dataset=None, **kwargs):
        print('transforming {}'.format(self.name))
        if not self.information_available and not self.isFit:
            information = self._transform(dataset=dataset, **kwargs)
            self.information = information if self.store else None
        elif not self.isFit:
            information = self.information
        else:
            information = self._transform(dataset=dataset, **kwargs)

        return information

    @data.enforce_dataset
    def _transform(self, X=None, y=None, dataset=None, **kwargs):
        print('_transforming {}'.format(self.name))
        if self.fitOnly:
            output = dataset
        else:
            output = self.transformer.transform(dataset=dataset)

        return output

    def fit_transform(self, X=None, y=None, dataset=None, *args, **kwargs):
        self.fit(X=X, y=y, dataset=dataset, *args, **kwargs)
        return self.transform(X=X, dataset=dataset, *args, **kwargs)

    def __getattr__(self, attr):
        return getattr(self.transformer, attr)


class ModelDAG(Dobject, nx.DiGraph, BaseTransformer):
    def __init__(self, executor='executor',
                 conductor=DatasetConductor(reverse=True, passTarget=True),
                 timeFormat="%Y_%m_%d_%H_%M",
                 _nodes=set([]), _edges={},
                 graphArgs=tuple(), graphKwargs={}):
        super(ModelDAG, self).__init__(*graphArgs, **graphKwargs)

        self.timeFormat = timeFormat
        self._initTime = now_string(timeFormat)

        self.graphArgs = graphArgs
        self.graphKwargs = graphKwargs
        self.executor = executor

        self.conductor = conductor

        self._nodes = _nodes
        self._edges = _edges

        [self.add_node_transformer(i) for i in _nodes]
        [self.add_edge_conductor(i, j, k) for (i, j), k in _edges.items()]

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

    @data.package_dataset
    @fallback(node='terminal')
    def fit(self, X=None, y=None, dataset=None, node=None):
        self.clean()
        # iterate through nodes => terminal_list to list
        parents = tuple(self.predecessors(node))
        if parents:
            upstreams = [self.apply(parent, dataset, 'fit_transform') for parent in parents]
            datas = [self.edge_exec(parent, node).fit_transform(upstream)
                     for parent, upstream in zip(parents, upstreams)]

            dataset = self.node_exec(node).combine(datas)

        self.node_exec(node).fit(dataset=dataset)

        self.isFit = True
        return self

    @data.package_dataset
    @fallback(node='terminal')
    def predict_proba(self, X=None, y=None, node=None, dataset=None):

        parents = tuple(self.predecessors(node))
        if parents:
            upstreams = [self.apply(parent, dataset, 'transform') for parent in parents]
            datas = [self.edge_exec(parent, node).fit_transform(upstream)
                     for parent, upstream in zip(parents, upstreams)]

            dataset = self.node_exec(node).combine(datas)

        probas = self.node_exec(node).predict_proba(dataset.designData)
        return probas

    @data.package_dataset
    @fallback(node='terminal')
    def transform(self, X=None, y=None, dataset=None, node=None):
        parents = tuple(self.predecessors(node))
        if parents:
            upstreams = [self.apply(parent, dataset, 'transform') for parent in parents]
            datas = [self.edge_exec(parent, node).fit_transform(upstream)
                     for parent, upstream in zip(parents, upstreams)]

            dataset = self.node_exec(node).combine(datas)

        transformed = self.node_exec(node).transform(dataset.designData)
        return transformed

    @data.package_dataset
    @fallback(node='terminal')
    def fit_transform(self, X=None, y=None, dataset=None, node=None):
        self.fit(data=dataset, node=node)
        return self.transform(dataset=dataset, node=node)

    def apply(self, node, data, method):
        parents = tuple(self.predecessors(node))

        if parents:
            output = [(self.node_exec(parent).information if self.node_exec(parent).information_available else
                      self.apply(parent,
                                 access(self.edge_exec(parent, node), [method])(data),
                                 method
                                 )) for
                      parent in parents]
        else:
            output = [data]

        dataset = self.node_exec(node).combine(output)

        spec = inspect.getargspec(getattr(self.node_exec(node), method))
        if 'dataset' in spec.args:
            payload = {'dataset': dataset}
        else:
            payload = {'X': dataset.designData}
            payload.update({'y': dataset.targetData}) if 'y' in spec.args else None
        information = access(self.node_exec(node), method=method, methodKwargs=payload)

        return information

    def add_node_transformer(self, node):
        self._nodes.add(node)
        self.add_node(node.name, **{self.executor: node})

    @fallback('conductor')
    def add_edge_conductor(self, node_from, node_to, conductor=None, **kwargs):
        conductor = clone(conductor)
        conductor.set_params(**kwargs)
        self.add_node_transformer(node_from) if not isinstance(node_from, str) else None
        self.add_node_transformer(node_to) if not isinstance(node_to, str) else None

        node_from = node_from.name if not isinstance(node_from, str) else node_from
        node_to = node_to.name if not isinstance(node_to, str) else node_to

        self.add_edge(node_from, node_to, **{self.executor: conductor})

        self._edges.update({(node_from, node_to): conductor})

    def __getattr__(self, attr):
        return getattr(self.node_exec(self.terminal), attr)


class OneHotEncoder(PandasMixin, OneHotEncoder):
    pass


class Imputer(PandasMixin, Imputer):
    pass


class StandardScaler(PandasMixin, StandardScaler):
    pass
