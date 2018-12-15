"""
Donatello Manager for orchestrating end to end ML
"""
from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.utils import Bunch

from donatello.components.data import Dataset, package_dataset
from donatello.components.splitter import Splitter
from donatello.components.disk import Local
from donatello.components.scorer import (ScorerClassification,
                                         ScorerSupervised)
from donatello.utils.helpers import has_nested_attribute, now_string
from donatello.utils.decorators import split_dataset, prepare_design, fallback
from donatello.utils.base import Dobject


class DM(Dobject, BaseEstimator):
    """
    Manager for model process. [a-z]*Declaration parameters map 1:1 to
    component objects attached via property setters. Other parameters
    attached directly.

    Manager accessors will fallback to accessing from estimator attributes

        dataDeclaration (dict): :py:class:`donatello.Dataset`
        splitterDeclaration (dict): arguments for :py:class:`donatello.Splitter`
        combiner (object): object with fit_transform method to\
            combine multiple datasets to prepare design matrix -\
            leveraged in :py:func:`donatello.utils.decorators.combine_data`
        estimator (donatello.BaseEstimator): estimator for\
            training and predicting
        scorerDeclaration (dict): arguments for :py:class:`donatello.Scorer`
        validation (bool): flag for calculating scoring metrics from
            nested cross val of training + validation sets
        holdOut (bool): flag for fitting estimator on entire training set
            and scoring test set
        metrics (iterable): list or dict of metrics for scorer
        hookDeclaration (dict): arguments for :py:class:`donatello.Local`
        writeAttrs (tuple): attributes to write out to disk
        nowFormat (str): format for creation time string
    """

    def __init__(self, dataDeclaration=None, splitterDeclaration=None,
                 combiner=None, estimator=None, scorerDeclaration=None,
                 validation=True, holdOut=True, entire=False,
                 metrics=None, hookDeclaration=None,
                 storeReferences=True,
                 mlType='classification',
                 typeDispatch={'scorer': {'classification': ScorerClassification,
                                          'regression': ScorerSupervised
                                          },
                               'splitter': Splitter,
                               'hook': Local
                               },
                 writeAttrs=('', 'estimator'),
                 timeFormat="%Y_%m_%d_%H_%M"):

        self._initTime = now_string(timeFormat)

        self.mlType = mlType
        self.typeDispatch = typeDispatch
        self.metrics = metrics
        self.combiner = combiner

        self.estimator = estimator

        # Build options
        self.validation = validation
        self.holdOut = holdOut
        self.entire = entire

        # Uses setters to instantiate components
        self.dataset = dataDeclaration
        self.splitter = splitterDeclaration
        self.scorer = scorerDeclaration
        self.hook = hookDeclaration

        # Other
        self.writeAttrs = writeAttrs
        self.storeReferences = storeReferences
        self._references = {}
        self.declaration = self.get_params(deep=False)
        self.scores = Bunch()

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
    def dataset(self):
        """
        Dataset object attached to manager
        """
        return self._dataset

    @dataset.setter
    def dataset(self, kwargs):
        kwargs = kwargs if kwargs else {}
        kwargs.update({'mlType': self.mlType}) if 'mlType' not in kwargs else None
        self._dataset = Dataset(**kwargs)

    @property
    def splitter(self):
        """
        Splitter object attached to manager
        """
        return self._splitter

    @splitter.setter
    def splitter(self, kwargs):
        kwargs = kwargs if kwargs else {}
        kwargs.update({'mlType': self.mlType}) if 'mlType' not in kwargs else None
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
        kwargs.update({'mlType': self.mlType}) if 'mlType' not in kwargs else None
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

    def _build_cross_validation(self, dataset, **fitParams):
        """
        Build cross validated scores over training data of models
        """
        print('Building Over Cross Validation')
        payload = {'estimator': self.estimator, 'metrics': self.metrics, 'dataset': dataset}
        self.scorerCrossValidation = self.scorer.buildCV(**payload)
        self.scores.crossValidation = Bunch(**self.scorerCrossValidation['scores'])
        self._references['cross_validation'] = self.estimator if self.storeReferences else None

    def _build_holdout(self, dataset, **fitParams):
        """
        Build model over training data and score
        """
        print('Building Over Holdout')
        self.estimator.fit(X=dataset.designTrain, y=dataset.targetTrain, gridSearch=True, **fitParams)

        payload = {'estimator': self.estimator, 'metrics': self.metrics,
                   'X': dataset.designTest, 'y': dataset.targetTest}
        self.scorerHoldout = self.scorer.build_holdout(**payload)
        self.scores.holdout = Bunch(**self.scorerHoldout['scores'])
        self._references['holdout'] = self.estimator if self.storeReferences else None

    def _build_entire(self, dataset, **fitParams):
        """
        Build model over entire data set
        """
        print('Building Over Entire Dataset')
        self.estimator = clone(self.estimator)
        self.estimator.fit(X=dataset.designData, y=dataset.targetData,
                           gridSearch=True, **fitParams)
        self._references['entire'] = self.estimator if self.storeReferences else None

    @fallback('writeAttrs')
    @package_dataset
    @split_dataset
    @prepare_design
    def fit(self, dataset=None, X=None, y=None, writeAttrs=None, **fitParams):
        """
        Build models, tune hyperparameters, and evaluate
        """

        self._build_cross_validation(dataset, **fitParams) if self.validation else None
        self._build_holdout(dataset, **fitParams) if self.holdOut else None
        self._build_entire(dataset, **fitParams) if self.entire else None

        self.write(writeAttrs=writeAttrs)

        return self

    def has_attribute(self, attr):
        return has_nested_attribute(self, attr)

    @fallback('writeAttrs')
    def write(self, writeAttrs=None):
        """
        Write objects to disk
        """
        writeAttrs = filter(self.has_attribute, writeAttrs)
        writePayloads = [{'obj': self, 'attr': attr} for attr in writeAttrs]
        [self.hook.write(**writePayload) for writePayload in writePayloads]

    def __getattr__(self, name):
        return getattr(self.estimator, name)
