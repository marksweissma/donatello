import re
import inspect
import pandas as pd
import networkx as nx

from collections import defaultdict

from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn import clone

from donatello.utils.base import Dobject, PandasAttrs, BaseTransformer, BaseDatasetTransformer
from donatello.utils.decorators import fallback, init_time
from donatello.utils.helpers import access, nvl
from donatello.components import data


if hasattr(inspect, 'signature'):
    funcsigs = inspect
else:
    import funcsigs


class PandasMixin(TransformerMixin, PandasAttrs):
    """
    Scikit-learn transformer with pandas bindings
    to enforce fields and features
    """
    @data.package_dataset
    @data.extract_fields
    def fit(self, X=None, y=None, dataset=None, *args, **kwargs):
        return super(PandasMixin, self).fit(X=dataset.designData,
                                            y=dataset.targetData, *args, **kwargs)

    @data.enforce_dataset
    @data.extract_features
    def transform(self, X=None, dataset=None, *args, **kwargs):
        return super(PandasMixin, self).transform(X=dataset.designData, *args, **kwargs)

    def fit_transform(self, X=None, y=None, dataset=None, *args, **kwargs):
        self.fit(X=X, y=y, dataset=dataset, *args, **kwargs)
        return self.transform(X=X, dataset=dataset, *args, **kwargs)


class PandasTransformer(BaseTransformer):
    """
    The scikit-learn TransformerMixin renders on fit_transform, disregard
    """
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


class DatasetTransformer(BaseDatasetTransformer):
    """
    Base scikit-learn style transformer
    """
    @data.package_dataset
    @data.extract_fields
    def fit(self, X=None, y=None, dataset=None, **kwargs):
        self.fields = list(dataset.designData) if dataset.designData is not None else []
        return self

    @data.enforce_dataset
    @data.extract_features
    def transform(self, X=None, y=None, dataset=None, **kwargs):
        return dataset

    def fit_transform(self, X=None, y=None, dataset=None, **kwargs):
        self.fit(X=X, y=y, dataset=dataset, **kwargs)
        return self.transform(X=X, y=y, dataset=dataset, **kwargs)


class TargetFlow(BaseTransformer):
    def __init__(self, passTarget):
        self.passTarget = passTarget

    @data.enforce_dataset
    def transform(self, dataset=None, **kwargs):
        return dataset.targetData if self.passTarget else None


def select_data_type(X, kwargs):
    """
    Select columns from X through :py:meth:`pandas.DataFrame.select_dtypes`
    kwargs are keyword arguments for the method

    Args:
        X (pandas.DataFrame):  table to select from
        **kwargs (): selection arguments
    Returns:
        list: features to include
    """
    features = list(X.select_dtypes(**kwargs))
    return features


def select_regex(X, patterns):
    """
    Select columns from X through matching any :py:func:`re.search`
    pattern in patterns
    kwargs are keyword arguments for the method

    Args:
        X (pandas.DataFrame):  table to select from
        patterns (iterable): patterns to search for matches
    Returns:
        list: features to include
    """
    features = [column for column in X if
                any([re.search(pattern, column) for pattern in patterns])]
    return features


FLOW_REGISTRY = {'data_type': select_data_type, 'regex': select_regex}


class DesignFlow(PandasTransformer):
    """
    Select subset of columns from keylike-valuelike store

    Args:
        selectValue (obj): values used for selection
        selectMethod (str): type of selection

            #. None / '' -> direct key /value look up (i.e. column names to slice with)
            # 'data_type' -> uses :py:meth:`pandas.DataFrame.select_dtypes` to select by data type.

        invert (bool): option to select all except those fields isolated\
            by selectValue and selectMethod
    """

    def __init__(self, selectValue=(), selectMethod=None, invert=False):
        self.selectValue = selectValue
        self.selectMethod = selectMethod
        self.invert = invert

    @data.package_dataset
    def fit(self, dataset=None, **fitParams):

        if self.selectMethod:
            inclusions = FLOW_REGISTRY[self.selectMethod](dataset.designData, self.selectValue)
        else:
            inclusions = self.selectValue

        if self.invert:
            exclusions = set(inclusions)
            inclusions = [i for i in dataset.designData if i not in exclusions]

        self.inclusions = inclusions

        return self

    @data.enforce_dataset
    def transform(self, dataset=None, **kwargs):
        dataset.designData = dataset.designData.reindex(columns=self.inclusions)
        return super(DesignFlow, self).transform(dataset)


class DatasetFlow(BaseTransformer):
    """
    Select subset of columns from keylike-valuelike store

    Args:
        selectValue (obj): values used for selection
        selectMethod (str): type of selection

            #. None / '' -> direct key /value look up (i.e. column names to slice with)
            # 'data_type' -> uses :py:meth:`pandas.DataFrame.select_dtypes` to select by data type.
        invert (bool): option to select all except those fields isolated\
            by selectValue and selectMethod
        passTarget (bool): flag to raw dataset's targetData through
        passDesign (bool): flag to raw dataset's designData through
    """

    def __init__(self,
                 selectValue=(), selectMethod=None, invert=False,
                 passTarget=True, passDesign=True):
        self.selectValue = selectValue
        self.selectMethod = selectMethod
        self.invert = invert
        self.passTarget = passTarget
        self.passDesign = passDesign

    def fit_design(self, dataset, **fitParams):
        if self.selectMethod:
            inclusions = FLOW_REGISTRY[self.selectMethod](dataset.designData, self.selectValue)
        else:
            inclusions = self.selectValue

        if self.invert:
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


class Apply(DatasetTransformer):
    """
    Apply a function during transform step on a dataset

    Args:
        func (function): function to apply
        fitOnly (bool): only apply during tranform following fit
    """

    def __init__(self, func=None, fitOnly=False):
        self.func = func
        self.fitOnly = fitOnly

    @data.package_dataset
    @data.extract_features
    @data.enforce_dataset
    def transform(self, X=None, y=None, dataset=None):

        if self.fitOnly and getattr(self, 'features', None):
            print('{} - fitOnly transformer - passing through'.format(self.name))
            output = dataset
        else:
            output = self.func(dataset)

        return output


class Access(DatasetTransformer):
    """
    Unified transform only interface. Leverages donatello's
    data access protoal to apply transform. For more info
    seoo :py:func:`donatello.utils.helpers.access`

    Args:
        designDap (dict): keyword arguments to apply on designData attribute via access protocol
        targetDap (dict): keyword arguments to apply on targetData attribute via access protocol
        datasetDap (dict): keyword arguments to apply on dataset via access protocol
        fitOnly (bool): only apply during tranform following fit
    """

    def __init__(self, designDap=None, targetDap=None, datasetDap=None, fitOnly=False):
        self.designDap = designDap
        self.targetDap = targetDap
        self.datasetDap = datasetDap
        self.fitOnly = fitOnly

    @data.package_dataset
    @data.extract_features
    @data.enforce_dataset
    def transform(self, X=None, y=None, dataset=None):

        if self.fitOnly and getattr(self, 'features', None):
            print('fitOnly transformer')
            return dataset

        if self.designDap:
            dataset = access(dataset.designData, **self.designDatadap)
        elif self.targetDap:
            dataset = access(dataset.targetData, **self.targetDatadap)
        elif self.datasetDap:
            dataset = access(dataset, **self.datasetDap)
        return dataset


def concat(datasets, params=None, dataType=data.Dataset):
    """
    Simple helper to combine design and target data  from propogation through
    :py:class:`ModelDAG`

    Args:
        datasets (list): list of datasets to combine
        params (dict): params for dataset object, if None infered from first dataset

    Returns:
        data.Dataset: combined dataset
    """
    if datasets:
        if len(datasets) == 1:
            dataset = datasets[0]
        elif not any([isinstance(i.data, dict) for i in datasets]):
            Xs = [dataset.designData for dataset in datasets if dataset.designData is not None]
            X = pd.concat(Xs, axis=1, join='inner') if Xs else None

            ys = [dataset.targetData for dataset in datasets if dataset.targetData is not None]
            y = ys[0] if len(ys) == 1 else None if len(ys) < 1 else pd.concat(ys, axis=1, join='inner')
            params = nvl(params, getattr(datasets[0], 'params', {}))
            dataset = dataType(X=X, y=y, **params)
        else:
            raise ValueError('datasets contains dicts and has len > 1, auto concat \
                            is not deterministic, provide a deterministic concat')
    else:
        dataset = None
    return dataset


class Node(Dobject, BaseTransformer):
    """
    Node in model execution grap

    Args:
        name (str): identifier for node in graph
        transformer (obj): supporting fit, transform, and fit_transform calls
        combine (func): function to combine incoming datasets from upstream nodes
        store (bool): hold information on Node (improves in memory calcs if multiple calls)
        enforceTarget (bool): enforce target in transformed dataset - this\
                is mainly a patch for scikit-learn wrapped transformers which ignore\
                y in transform calls preventing it from passing through the transformer
    """

    def __init__(self, name, transformer=None, combine=concat, enforceTarget=False, fitOnly=False):

        self.name = name
        self.transformer = transformer
        self.combine = combine
        self.enforceTarget = enforceTarget
        self.fitOnly = fitOnly

    def reset(self):
        self.transformer = clone(self.transformer)

    @data.package_dataset
    @data.extract_fields
    def fit(self, X=None, y=None, dataset=None, **kwargs):
        sig = funcsigs.signature(self.transformer.fit)

        if 'dataset' in sig.parameters:
            payload = {'dataset': dataset}
        else:
            payload = {'X': dataset.designData}
            payload.update({'y': dataset.targetData}) if 'y' in sig.parameters else None
        payload.update({i: j for i, j in kwargs.items() if i != 'fitting'})
        self.transformer.fit(**payload)

        return self

    @data.package_dataset
    @data.extract_features
    def transform(self, X=None, y=None, dataset=None, fitting=False, **kwargs):
        if self.fitOnly and not fitting:
            return dataset
        information = self._transform(dataset=dataset, **kwargs)
        if isinstance(information,
                      data.Dataset) and self.enforceTarget and not information._has_target:
            information.targetData = dataset.targetData

        return information

    @data.enforce_dataset
    def _transform(self, X=None, y=None, dataset=None, **kwargs):
        output = self.transformer.transform(dataset=dataset)
        return output

    def fit_transform(self, X=None, y=None, dataset=None, fitting=False, *args, **kwargs):
        self.fit(X=X, y=y, dataset=dataset, *args, **kwargs)
        return self.transform(X=X, y=y, dataset=dataset, ftting=fitting, *args, **kwargs)

    def __getattr__(self, attr):
        return getattr(self.transformer, attr)


class ModelDAG(nx.DiGraph, Dobject, BaseTransformer):
    """
    In memory data transformation graph. Nodes are accessed via
    via any :py:class:`networkx.DiGraph` accessor of the node's name.
    The networkx graph is constucted via the names with the goal that
    all user operations are more intuitively accomplished by a string / name
    access pattern. The ModelDAG acts as a wrapper to enable that concept
    in the context of a scikit-learn style transformer

    Note:

        ``set_params`` method
        is patches to set node[executor].transformer attributes.
        Changing the ``C`` value of a graph with a Logistic Regression in node 'n2'
        is accomplished via graph.set_params(n2__C=value)


    Warning:

        private (``_``) node and edges arguments in constructor  are a hack
        to maintin the scikit-learn contracts. Supplying nodes in the graphArgs/Kwargs
        will cause cloning to fail or cloning to lose neccesary information. Please
        do not do this

    Args:
        _nodes (set): nodes objects to intialize with - recommended to use set([]) and buidl with helpers
        _edges (dict): edge ids mapped to transformers - recommended to use {} and build helpers
        executor (str): name of attribute in each node where transform object is stored
        flow (obj): default edge flow transformer
        timeFormat (str): str format for logging init time
    """

    @init_time
    def __init__(self, _nodes, _edges, executor='executor',
                 flow=DatasetFlow(invert=True, passTarget=True),
                 timeFormat="%Y_%m_%d_%H_%M", initTime=None,
                 graphArgs=tuple(), graphKwargs={}
                 ):

        super(ModelDAG, self).__init__(*graphArgs, **graphKwargs)

        self.graphArgs = graphArgs
        self.graphKwargs = graphKwargs
        self.executor = executor

        self.flow = flow

        self._nodes = _nodes
        self._edges = _edges

        [self.add_node_transformer(i) for i in _nodes]
        [self.add_edge_flow(i, j, k) for (i, j), k in _edges.items()]

    def set_params(self, **params):
        """Set the parameters of this estimator.

        **Ripped from sklearn 0.20.02** extended for setting transformer attributes
        of edges and nodes accessed via node names and node name `_` separated edges

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Returns
        -------
        self
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self

        valid_params = self.get_params(deep=True)
        for name, node in self.nodes.items():
            valid_params[name] = node
            valid_params.update({name + '__' + k: val for k, val in
                                 self.nodes[name][self.executor].transformer.get_params().items()})

        for (n1, n2), edge in self.edges.items():
            n1_n2 = "_".join([n1, n2])
            valid_params[n1_n2] = edge
            valid_params.update({n1_n2 + '__' + k: val for k, val in
                                 self.edges[(n1, n2)][self.executor].get_params().items()})

        nested_params = defaultdict(dict)  # grouped by prefix
        node_params = defaultdict(dict)  # grouped by prefix
        edge_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))
            if delim:
                if key in self.nodes:
                    node_params[key][sub_key] = value
                elif key in set(["_".join([n1, n2]) for n1, n2 in self.edges]):
                    edge_params[key][sub_key] = value
                else:
                    nested_params[key][sub_key] = value
            else:
                if key in self._nodes:
                    self.add_node_transformer(value)
                elif key in set(["_".join([n1, n2]) for n1, n2 in self.edges]):
                    n1, n2 = key.split('_')
                    self.add_edge_flow(n1, n2, value)
                else:
                    setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        for key, sub_params in node_params.items():
            self.node_exec(key).transformer.set_params(**sub_params)

        for key, sub_params in edge_params.items():
            self.edge_exec(*key.split('_')).set_params(**sub_params)

        return self

    def node_exec(self, node):
        """
        Return object accessed by node name

        Args:
            node (hashable): node accessor

        Returns:
            obj: (by default a Node)
        """
        return self.nodes[node][self.executor]

    def edge_exec(self, node_from, node_to):
        """
        Return object accessed by edge

        Args:
            node_from (hashable): node accessor for originating node
            node_to (hashable): node accessor for terminating node

        Returns:
            obj: by default graph's default flow
        """
        return self.get_edge_data(node_from, node_to)[self.executor]

    def clean(self):
        [self.node_exec(node).reset() for node in self]

    @property
    def terminal(self):
        """
        accessor for terminal node, or list of terminal nodes if not single terminating node
        """
        terminal = [node for node in self.nodes if self.out_degree(node) == 0]
        terminal = terminal[0] if len(terminal) == 1 else terminal
        if isinstance(terminal, list):
            raise Exception('''multi terminating graphs not yet support
                               graph has {}: {} terminal nodes'''.format(len(terminal), ", ".join(terminal))
                            )
        return terminal

    def process(self, dataset, node, method, **kwargs):
        """
        Evalute method through nodes that accept dataset inputs
        """
        parents = tuple(self.predecessors(node))
        if parents:
            upstreams = [self.apply(parent, dataset, method, **kwargs) for parent in parents]
            datas = [
                access(
                    self.edge_exec(
                        parent, node), method=method, methodKwargs=dict(
                        dataset=upstream)) for parent, upstream in zip(
                    parents, upstreams)]

            dataset = self.node_exec(node).combine(datas)
        return dataset

    @data.package_dataset
    @fallback(node='terminal')
    def fit(self, X=None, y=None, dataset=None, node=None, clean=True, **kwargs):
        """
        Fit graph to data
        """
        if clean:
            self.clean()

        kwargs.update({'fitting': True})
        dataset = self.process(dataset, node, 'fit_transform', **kwargs)

        self.features = list(dataset.designData)
        self.node_exec(node).fit(dataset=dataset, **kwargs)

        self.isFit = True
        return self

    @data.package_dataset
    @fallback(node='terminal')
    def predict(self, X=None, y=None, node=None, dataset=None, **kwargs):
        """
        Transform data and predict at termination of subcomponent
        """
        dataset = self.process(dataset, node, 'transform', **kwargs)
        predictions = self.node_exec(node).predict(dataset.designData)
        return predictions

    @data.package_dataset
    @fallback(node='terminal')
    def predict_proba(self, X=None, y=None, node=None, dataset=None, **kwargs):
        """
        Transform data and predict_proba at termination of subcomponent
        """
        dataset = self.process(dataset, node, 'transform', **kwargs)
        probas = self.node_exec(node).predict_proba(dataset.designData)
        return probas

    @data.package_dataset
    @fallback(node='terminal')
    def predict_log_proba(self, X=None, y=None, node=None, dataset=None, **kwargs):
        """
        Transform data and predict_proba at termination of subcomponent
        """
        dataset = self.process(dataset, node, 'transform', **kwargs)
        log_probas = self.node_exec(node).predict_log_proba(dataset.designData)
        return log_probas

    @data.package_dataset
    @fallback(node='terminal')
    def decision_function(self, X=None, y=None, node=None, dataset=None, **kwargs):
        """
        Transform data and decision_function at termination of subcomponent
        """
        dataset = self.process(dataset, node, 'transform', **kwargs)
        decisions = self.node_exec(node).decision_function(dataset.designData)
        return decisions

    @data.package_dataset
    @fallback(node='terminal')
    def transform(self, X=None, y=None, dataset=None, node=None, **kwargs):
        """
        Transform data given through node given a fit subcomponent
        """
        dataset = self.process(dataset, node, 'transform', **kwargs)
        transformed = self.node_exec(node).transform(dataset=dataset, **kwargs)
        return transformed

    @data.package_dataset
    @fallback(node='terminal')
    def fit_transform(self, X=None, y=None, dataset=None, node=None, clean=True, **kwargs):
        """
        Fit model to data and return the transform at the given node
        """
        if clean:
            self.clean()
        kwargs.update({'fitting': True})
        dataset = self.process(dataset, node, 'fit_transform', **kwargs)
        transformed = self.node_exec(node).fit_transform(dataset=dataset, **kwargs)
        return transformed

    def apply(self, node, data, method, **kwargs):
        """
        Apply a method through the graph terminating at a node

        Args:
            node (str): name of terminal node
            data (data.Dataset): data to process
            method (str): string name of method to call through nodes

        Returns:
            data.Dataset: transformed data
        """
        parents = tuple(self.predecessors(node))
        if parents:
            output = [access(self.edge_exec(parent, node), method=method,
                             methodKwargs=dict(dataset=self.apply(parent, data, method))
                             )
                      for parent in parents]
        else:
            output = [data]

        dataset = self.node_exec(node).combine(output)

        sig = funcsigs.signature(getattr(self.node_exec(node), method))
        if 'dataset' in sig.parameters:
            payload = {'dataset': dataset}
        else:
            payload = {'X': dataset.designData}
            payload.update({'y': dataset.targetData}) if 'y' in sig.parameters else None
            payload.update(kwargs)
        information = access(self.node_exec(node), method=method, methodKwargs=payload)

        return information

    def add_node_transformer(self, node):
        """
        Add a node object to the graph referenceable by it's name

        Args:
            node (obj): node to add to the graph
        """
        self._nodes.add(node)
        self.add_node(node.name, **{self.executor: node})

    @fallback('flow')
    def add_edge_flow(self, node_from, node_to, flow=None, **kwargs):
        """
        Add an edge to execution graph

        Args:
            node_from (str | obj): reference the source node by adding the node object\
                    or the name of the existing object in the graph
            node_to (str | obj): reference the sink node by adding the node object\
                    or the name of the existing object in the graph
            flow (obj): defaults to self's default - transformer for the edge
            **kwargs: params to set of the flow obj
        """
        flow = clone(flow)
        flow.set_params(**kwargs)

        if not isinstance(node_from, str):
            self.add_node_transformer(node_from)
            node_from = node_from.name
        if not isinstance(node_to, str):
            self.add_node_transformer(node_to)
            node_to = node_to.name

        self.add_edge(node_from, node_to, **{self.executor: flow})

        self._edges.update({(node_from, node_to): flow})

    def __getattr__(self, attr):
        return getattr(self.node_exec(self.terminal), attr)


class OneHotEncoder(PandasTransformer):
    def __init__(self, columns=None, dropOne=False, missing='missing'):
        self.columns = columns
        self.dropOne = dropOne
        self.missing = missing

    @data.package_dataset
    @data.extract_fields
    def fit(self, X=None, y=None, dataset=None, *args, **kwargs):
        df = dataset.designData[self.columns] if self.columns else dataset.designData
        df = df.fillna(self.missing)
        self.taxonomy = {i: df[i].unique() for i in df}
        return self

    @data.enforce_dataset
    @data.extract_features
    def transform(self, X=None, y=None, dataset=None, *args, **kwargs):
        design = [dataset.designData.drop(self.taxonomy.keys(), errors='ignore', axis=1)]

        _X = pd.concat([dataset.designData[column].astype('category').cat.set_categories(value)
                        for column, value in self.taxonomy.items()], axis=1)
        X = pd.get_dummies(_X, drop_first=self.dropOne)
        dataset = dataset.with_params(X=pd.concat(design + [X], axis=1), y=dataset.targetData)
        return dataset


class Exists(PandasTransformer):
    def __init__(self, columns=None, tolerance=0, suffix='exists',
                 minFields=None, maxFields=None, meanFields=None):
        self.columns = columns
        self.tolerance = tolerance
        self.suffix = suffix
        self.minFields = minFields
        self.maxFields = maxFields
        self.meanFields = meanFields

    @data.package_dataset
    @data.extract_fields
    def fit(self, X=None, y=None, dataset=None, *args, **kwargs):
        self.mins = {}
        self.maxs = {}
        self.means = {}

        df = dataset.designData[self.columns] if self.columns else dataset.designData
        self.existsColumns = [i for i in df if df[i].isnull().mean() >= self.tolerance]

        if self.minFields:
            for field in [i for i in self.minFields if i in df]:
                self.mins[field] = df[field].min()
        if self.maxFields:
            for field in [i for i in self.maxFields if i in df]:
                self.maxs[field] = df[field].max()
        if self.meanFields:
            for field in [i for i in self.meanFields if i in df]:
                self.means[field] = df[field].mean()
        return self

    @data.enforce_dataset
    @data.extract_features
    def transform(self, X=None, y=None, dataset=None, *args, **kwargs):
        df = dataset.designData
        for field in self.existsColumns:
            if field in df:
                df["_".join([field, self.suffix])] = df[field].notnull()
            else:
                df["_".join([field, self.suffix])] = False

        for field in [i for i in self.mins if i in df]:
            df[field] = df[field].fillna(self.mins[field]) if field in df else self.mins[field]

        for field in [i for i in self.maxs if i in df]:
            df[field] = df[field].fillna(self.maxs[field]) if field in df else self.maxs[field]

        for field in [i for i in self.means if i in df]:
            df[field] = df[field].fillna(self.means[field]) if field in df else self.means[field]

        dataset = dataset.with_params(X=df, y=dataset.targetData)
        return dataset


class Imputer(PandasMixin, Imputer):
    """
    Wrapped :py:class:`sklearn.preprocessing.Imputer`
    """
    pass


class StandardScaler(PandasMixin, StandardScaler):
    """
    Wrapped :py:class:`sklearn.preprocessing.StandardScaler`
    """
    pass


class Pipeline(PandasMixin, Pipeline):
    """
    Wrapped :py:class:`sklearn.pipeline.Pipeline`

    Will default to looking for attributes in last transformer
    """

    def __getattr__(self, attr): return getattr(self.steps[-1][1], attr)
