from warnings import warn

from abc import ABCMeta, abstractproperty
from copy import deepcopy

from sklearn import clone
from sklearn.base import BaseEstimator as _BaseEstimator
from sklearn.utils import Bunch

from donatello.components.data import Data
from donatello.components.splitter import Splitter
from donatello.components.hook import Local
from donatello.components.scorer import (Scorer,
                                         ScorerClassification,
                                         ScorerRegression)
from donatello.utils.helpers import has_nested_attribute, nvl, now_string
from donatello.utils.decorators import split_data, prepare_design
from donatello.utils.base import Dobject
from donatello.components.data import package_data


class DM(Dobject, _BaseEstimator):
    """
    Manager for model process. [a-z]*Kwargs parameters map 1:1 to
    component objects attached via property setters. Other parameters
    attached directly.

    :param dict dataKwargs: :py:class:`donatello.Data`
    :param dict splitterKwargs: arguments for :py:class:`donatello.Splitter`
    :param object combiner: object with fit_transform method to\
            combine multiple datasets to prepare design matrix -\
            leveraged in :py:func:`donatello.utils.decorators.combine_data`
    :param donatello.BaseEstimator estimator: estimator for\
            training and predicting
    :param dict scorerKwargs: arguments for :py:class:`donatello.Scorer`
    :param bool validation: flag for calculating scoring metrics from
            nested cross val of training + validation sets
    :param bool holdOut: flag for fitting estimator on entire training set
            and scoring test set
    :param iterable metrics: list or dict of metrics for scorer
    :param dict hookKwargs: arguments for :py:class:`donatello.Local`
    :param tuple writeAttrs: attributes to write out to disk
    :param str nowFormat: format for creation time string
    """

    def __init__(self, dataKwargs=None, splitterKwargs=None,
                 combiner=None, estimator=None, scorerKwargs=None,
                 validation=True, holdOut=True, entire=False,
                 metrics=None, hookKwargs=None,
                 storeReferences=True,
                 mlType='classification',
                 typeDispatch= {'scorer': {'classification': ScorerClassification,
                                            'regression': ScorerRegression
                                           },
                                 'splitter': Splitter,
                                 'hook': Local
                                 },
                 writeAttrs=('', 'estimator'),
                 timeFormat="%Y_%m_%d_%H_%M"):

        self._initTime = now_string(timeFormat)

        # Preserve params
        self.dataKwargs = dataKwargs
        self.splitterKwargs = splitterKwargs
        self.scorerKwargs = scorerKwargs
        self.hookKwargs = hookKwargs

        self._mlType = mlType
        self.typeDispatch = typeDispatch
        self.metrics = metrics
        self.combiner = combiner

        self.estimator = clone(estimator)

        # Build options
        self.validation = validation
        self.holdOut = holdOut
        self.entire = entire

        # Uses setters to instantiate components
        self.data = dataKwargs
        self.splitter = splitterKwargs
        self.scorer = scorerKwargs
        self.hook = hookKwargs

        # Other
        self.writeAttrs = writeAttrs
        self.storeReferences = storeReferences
        self._references = {}
        self.declaration = self.get_params(deep=False)
        self.scores = Bunch()

    # state
    @property
    def mlType(self):
        """
        Define type of learning
            #. Regression
            #. Classificaiton
            #. Clustering
       """

        return self._mlType

    @property
    def name(self):
        """
        Name of type
        """
        name = self.__class__.__name__
        return name

    @property
    def declaration(self):
        """
        Dictionary of kwargs given during instantiation
        """
        return {i: clone(j) for i, j in self._declaration.items()}

    @declaration.setter
    def declaration(self, value):
        self._declaration = value

    # components
    @property
    def data(self):
        """
        Data object attached to manager
        """
        return self._data

    @data.setter
    def data(self, kwargs):
        kwargs = kwargs if kwargs else {}
        self._data = Data(**kwargs)

    @property
    def splitter(self):
        """
        Splitter object attached to manager
        """
        return self._splitter

    @splitter.setter
    def splitter(self, kwargs):
        kwargs = kwargs if kwargs else {}
        self._splitter = self.typeDispatch.get('splitter')(**kwargs)

    @property
    def scorer(self):
        """
        Scorer object attached to manager
        """
        return self._scorer

    @scorer.setter
    def scorer(self, kwargs):
        kwargs = kwargs if kwargs else {}
        self._scorer = self.typeDispatch.get('scorer').get(self.mlType)(**kwargs)

    @property
    def hook(self):
        """
        Local object attached to manager
        """
        return self._hook

    @hook.setter
    def hook(self, kwargs):
        kwargs = {} if kwargs is None else kwargs
        self._hook = self.typeDispatch.get('hook')(**kwargs)

    def _build_cross_validation(self, data, **fitParams):
        """
        Build cross validated scores over training data of models
        """

        payload = {'estimator': self.estimator, 'metrics': self.metrics,
                   'X': data.designTrain, 'y': data.targetTrain}
        self.scorerCrossValidation = self.scorer.buildCV(**payload)
        self.scores.crossValidation = Bunch(**self.scorerCrossValidation['scores'])
        self._references['cross_validation'] = clone(self.estimator) if self.storeReferences else None

    def _build_holdout(self, data, **fitParams):
        """
        Build model over training data and score
        """
        self.estimator.fit(X=data.designTrain, y=data.targetTrain,
                           gridSearch=True, **fitParams)
        payload = {'estimator': self.estimator, 'metrics': self.metrics,
                   'X': data.designTest, 'y': data.targetTest}
        self.scorerHoldout = self.scorer.build_holdout(**payload)
        self.scores.holdout = Bunch(**self.scorerHoldout['scores'])
        self._references['holdout'] = clone(self.estimator) if self.storeReferences else None

    def _build_entire(self, data, **fitParams):
        """
        Build model over entire data set
        """
        self.estimator = clone(self.estimator)
        self.estimator.fit(X=data.designData, y=data.targetData,
                           gridSearch=True, **fitParams)
        self._references['entire'] = clone(self.estimator) if self.storeReferences else None

    @package_data
    @split_data
    @prepare_design
    def fit(self, data=None, X=None, y=None, **fitParams):
        """
        Build models, tune hyperparameters, and evaluate
        """

        self._build_cross_validation(data, **fitParams) if self.validation else None
        self._build_holdout(data, **fitParams) if self.holdOut else None
        self._build_entire(data, **fitParams) if self.entire else None

        self.write(self.writeAttrs) if self.writeAttrs else None

        return self

    def write(self, writeAttrs=()):
        """
        Write objects to disk
        """
        writeAttrs = nvl(writeAttrs, self.writeAttrs)
        writeAttrs = [i for i in writeAttrs if has_nested_attribute(self, i)]
        writePayloads = [{'attr': attr} for attr in writeAttrs]
        [writePayload.update({'obj': self}) for writePayload in writePayloads]
        [self.hook.write(**writePayload) for writePayload in writePayloads]

    def __getattr__(self, name):
        return getattr(self.estimator, name)
