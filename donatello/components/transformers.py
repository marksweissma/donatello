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
    def fit(self, X=None, y=None, dataset=None, **kwargs):
        self.fields = list(dataset.designData) if dataset.designData is not None else []
        return self

    @data.enforce_dataset
    @data.extract_features
    def transform(self, X=None, y=None, dataset=None, **kwargs):
        self.features = list(dataset.designData) if dataset.designData is not None else []
        return dataset

    def fit_transform(self, X=None, y=None, dataset=None, **kwargs):
        self.fit(X=X, y=y, dataset=dataset, **kwargs)
        return self.transform(X=X, y=y, dataset=dataset, **kwargs)


class TargetConductor(BaseTransformer):
    def __init__(self, passTarget):
        self.passTarget = passTarget

    @data.enforce_dataset
    def transform(self, dataset=None, **kwargs):
        return dataset.targetData if self.passTarget else None


def select_data_type(X, **kwargs):
    """
    Select columns from X through :py:meth:`pandas.DataFrame.select_dtypes`
    kwargs are keyword arguments for the method

    Args:
        X (pandas.DataFrame):  table to select from
        **kwargs (): selection arguments
    Returns:
        list: features to include
    """
    features = X.select_dtypes(**kwargs).columns.tolist()
    return features


def select_regex(X, patterns):
    """
    Select columns from X through matching any :py:func:`re.match`
    pattern in patterns
    kwargs are keyword arguments for the method

    Args:
        X (pandas.DataFrame):  table to select from
        patterns (iterable): patterns to search for matches
    Returns:
        list: features to include
    """
    features = [i for i in X if any([re.match(j, i) for j in patterns])]
    return features


CONDUCTION_REGISTRY = {'data_type': select_data_type, 'regex': select_regex}


class DesignConductor(PandasTransformer):
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
            inclusions = CONDUCTION_REGISTRY[self.selectMethod](dataset.designData, self.selectValue)
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
        return super(DesignConductor, self).transform(dataset)


class DatasetConductor(BaseTransformer):
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
            inclusions = CONDUCTION_REGISTRY[self.selectMethod](dataset.designData, self.selectValue)
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


class ApplyTransformer(DatasetTransformer):
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

        if self.fitOnly and getattr(self, 'features', None) not in ([], None):
            output = dataset
        else:
            output = self.func(dataset)

        return output


class AccessTransformer(DatasetTransformer):
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

        if self.fitOnly and getattr(self, 'features', None) not in ([], None):
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
        Xs = [dataset.designData for dataset in datasets if dataset.designData is not None]
        ys = [dataset.targetData for dataset in datasets if dataset.targetData is not None]

        X = pd.concat(Xs, axis=1, join='inner') if Xs else None
        y = ys[0] if len(ys) == 1 else None if len(ys) < 1 else pd.concat(ys, axis=1, join='inner')
        params = nvl(params, getattr(datasets[0], 'params', None))
        dataset = dataType(X=X, y=y, **params)
    else:
        dataset = None
    return dataset


class _TransformNode(Dobject, BaseTransformer):
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
    def __init__(self, name, transformer=None, combine=concat, store=False, enforceTarget=False):

        self.name = name
        self.transformer = transformer
        self.combine = combine
        self.store = store

        self.information = None
        self.enforceTarget = enforceTarget

    @property
    def information_available(self):
        return self.information is not None

    def reset(self):
        self.information = None
        self.transformer = clone(self.transformer)

    @data.package_dataset
    @data.extract_fields
    def fit(self, X=None, y=None, dataset=None, **kwargs):
        spec = inspect.getargspec(self.transformer.fit)
        if 'dataset' in spec.args:
            payload = {'dataset': dataset}
        else:
            payload = {'X': dataset.designData}
            payload.update({'y': dataset.targetData}) if 'y' in spec.args else None
        payload.update(kwargs)
        self.transformer.fit(**payload)

        return self

    @data.package_dataset
    @data.extract_features
    def transform(self, X=None, y=None, dataset=None, **kwargs):
        try:
            if not self.information_available:
                information = self._transform(dataset=dataset, **kwargs)
                self.information = information if self.store else None
            else:
                information = self._transform(dataset=dataset, **kwargs)

            if isinstance(information, data.Dataset) and self.enforceTarget and not information._has_target:
                information.targetData = dataset.targetData
        except:
            import pdb; pdb.set_trace()

        return information

    @data.enforce_dataset
    def _transform(self, X=None, y=None, dataset=None, **kwargs):
        output = self.transformer.transform(dataset=dataset)
        return output

    def fit_transform(self, X=None, y=None, dataset=None, *args, **kwargs):
        self.fit(X=X, y=y, dataset=dataset, *args, **kwargs)
        return self.transform(X=X, dataset=dataset, *args, **kwargs)

    def __getattr__(self, attr):
        return getattr(self.transformer, attr)


class TransformNode(Dobject, BaseTransformer):
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
    def __init__(self, name, transformer=None, combine=concat, enforceTarget=False):

        self.name = name
        self.transformer = transformer
        self.combine = combine
        self.enforceTarget = enforceTarget

    def reset(self):
        self.transformer = clone(self.transformer)

    @data.package_dataset
    @data.extract_fields
    def fit(self, X=None, y=None, dataset=None, **kwargs):
        spec = inspect.getargspec(self.transformer.fit)
        if 'dataset' in spec.args:
            payload = {'dataset': dataset}
        else:
            payload = {'X': dataset.designData}
            payload.update({'y': dataset.targetData}) if 'y' in spec.args else None
        payload.update(kwargs)
        self.transformer.fit(**payload)

        return self

    @data.package_dataset
    @data.extract_features
    def transform(self, X=None, y=None, dataset=None, **kwargs):
        information = self._transform(dataset=dataset, **kwargs)
        if isinstance(information, data.Dataset) and self.enforceTarget and not information._has_target:
            information.targetData = dataset.targetData

        return information

    @data.enforce_dataset
    def _transform(self, X=None, y=None, dataset=None, **kwargs):
        output = self.transformer.transform(dataset=dataset)
        return output

    def fit_transform(self, X=None, y=None, dataset=None, *args, **kwargs):
        self.fit(X=X, y=y, dataset=dataset, *args, **kwargs)
        return self.transform(X=X, y=y, dataset=dataset, *args, **kwargs)

    def __getattr__(self, attr):
        return getattr(self.transformer, attr)


class ModelDAG(Dobject, nx.DiGraph, BaseTransformer):
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
        conductor (obj): default edge conduction transformer
        timeFormat (str): str format for logging init time
        _nodes (set): patch for handlin
    """
    def __init__(self, _nodes, _edges, executor='executor',
                 conductor=DatasetConductor(invert=True, passTarget=True),
                 timeFormat="%Y_%m_%d_%H_%M",
                 graphArgs=tuple(), graphKwargs={}
                 ):

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

    def set_params(self, **params):
        """Set the parameters of this estimator.

        **Ripped from sklearn 0.20.02** enables setting transformer attributers
        accessed via node names and `_` separated edges

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
                    self.add_edge_conductor(n1, n2, value)
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

        Arg:
            node (hashable): node accessor
        Returns:
            obj: (by default a TransformNode)
        """
        return self.nodes[node][self.executor]

    def edge_exec(self, node_from, node_to):
        """
        Return object accessed by edge

        Arg:
            node_from (hashable): node accessor for originating node
            node_to (hashable): node accessor for terminating node
        Returns:
            obj: by default graph's default conductor
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
        return terminal

    @data.package_dataset
    @fallback(node='terminal')
    def fit(self, X=None, y=None, dataset=None, node=None, clean=True):
        if clean:
            self.clean()
        parents = tuple(self.predecessors(node))
        if parents:
            upstreams = [self.apply(parent, dataset, 'fit_transform') for parent in parents]
            datas = [self.edge_exec(parent, node).fit_transform(upstream)
                     for parent, upstream in zip(parents, upstreams)]

            for parent, upstream in zip(parents, upstreams):
                print parent
                print upstream.data.head()
                print '*'*10
            dataset = self.node_exec(node).combine(datas)

        self.node_exec(node).fit(dataset=dataset)

        self.isFit = True
        return self

    @data.package_dataset
    @fallback(node='terminal')
    def predict(self, X=None, y=None, node=None, dataset=None):

        parents = tuple(self.predecessors(node))
        if parents:
            upstreams = [self.apply(parent, dataset, 'transform') for parent in parents]
            datas = [self.edge_exec(parent, node).transform(upstream)
                     for parent, upstream in zip(parents, upstreams)]

            for parent, upstream in zip(parents, upstreams):
                print parent
                print upstream.data.head()
                print '*'*10

            import pdb; pdb.set_trace()
            dataset = self.node_exec(node).combine(datas)

        predictions = self.node_exec(node).predict(dataset.designData)
        return predictions

    @data.package_dataset
    @fallback(node='terminal')
    def predict_proba(self, X=None, y=None, node=None, dataset=None):

        parents = tuple(self.predecessors(node))
        if parents:
            upstreams = [self.apply(parent, dataset, 'transform') for parent in parents]
            datas = [self.edge_exec(parent, node).transform(upstream)
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

        transformed = self.node_exec(node).transform(dataset=dataset)
        return transformed

    @data.package_dataset
    @fallback(node='terminal')
    def fit_transform(self, X=None, y=None, dataset=None, node=None, clean=True):
        """
        """
        if clean:
            self.clean()
        parents = tuple(self.predecessors(node))
        if parents:
            upstreams = [self.apply(parent, dataset, 'fit_transform') for parent in parents]
            datas = [self.edge_exec(parent, node).fit_transform(upstream)
                     for parent, upstream in zip(parents, upstreams)]

            for parent, upstream in zip(parents, upstreams):
                print parent
                print upstream.data.head()
                print '*'*10

            dataset = self.node_exec(node).combine(datas)

        transformed = self.node_exec(node).fit_transform(dataset=dataset)

        return transformed

    def apply(self, node, data, method):
        parents = tuple(self.predecessors(node))
        print(node)

        if parents:
            # output = [(self.node_exec(parent).information if self.node_exec(parent).information_available else
            output = [(
                      self.apply(parent,
                                 access(self.edge_exec(parent, node), [method])(dataset=data),
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


class OneHotEncoder(PandasTransformer):
    def __init__(self, columns=None, dropOne=False):
        self.columns = columns
        self.dropOne = dropOne

    @data.package_dataset
    @data.extract_fields
    def fit(self, X=None, y=None, dataset=None, *args, **kwargs):
        df = dataset.designData[self.columns] if self.columns else dataset.designData
        self.taxonomy = {i: df[i].unique() for i in df}
        return self

    @data.enforce_dataset
    @data.extract_features
    def transform(self, X=None, y=None, dataset=None, *args, **kwargs):
        design = dataset.designData.drop(self.taxonomy.keys(), errors='ignore')
        design = [design] if len(design.columns) > 1 else []

        try:
            _X = pd.concat([dataset.designData[column].astype('category').cat.set_categories(value)
                            for column, value in self.taxonomy.items()], axis=1)
        except Exception as e:
            # import pdb; pdb.set_trace()
            raise e
        X = pd.get_dummies(_X, drop_first=self.dropOne)
        design.append(X)
        dataset = dataset.with_params(X=pd.concat(design, axis=1), y=dataset.targetData)
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
    def __getattr__(self, attr):
        return getattr(self.steps[-1][1], attr)
