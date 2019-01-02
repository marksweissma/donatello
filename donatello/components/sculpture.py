"""
Manager for orchestrating end to end ML
"""
from copy import deepcopy
from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.utils import Bunch

from donatello.components.data import package_dataset, subset_dataset
from donatello.components.scorer import Scorer
from donatello.components.disk import Local

from donatello.utils.helpers import now_string
from donatello.utils.decorators import fallback
from donatello.utils.base import Dobject


class Sculpture(Dobject, BaseEstimator):
    """
    Instance for model processing. [a-z]*Declaration parameters map 1:1 to
    component objects attached via property setters. Other parameters
    attached directly.

    Manager accessors will fallback to accessing from estimator attributes

    Args:
        dataDeclaration (dict): :py:class:`donatello.components.data.Dataset`
        estimatorDeclaration (dict): arguments for :py:class:`donatello.components.estimator.Estimator`
        scorerDeclaration (dict): arguments for :py:class:`donatello.components.scorer.Scorer`
        validation (bool): flag for calculating scoring metrics from nested cross val of training + validation sets
        holdOut (bool): flag for fitting estimator on single training set and scoring on test set
        metrics (iterable): list or dict of metrics for scorer
        hookDeclaration (dict): arguments for :py:class:`donatello.Local`
        writeAttrs (tuple): attributes to write out to disk
        timeFormat (str): format for creation time string
    """

    def __init__(self,
                 dataset=None,
                 estimator=None,
                 validation=True, holdOut=False, entire=False,
                 scorer=Scorer(), hook=Local(), metrics=None,
                 storeReferences=True, writeAttrs=('', 'estimator'),
                 timeFormat="%Y_%m_%d_%H_%M"):

        self._initTime = now_string(timeFormat)

        self.dataset = dataset
        self.estimator = estimator

        self.scorer = scorer
        self.hook = hook

        self.metrics = metrics

        # Build options
        self.validation = validation
        self.holdOut = holdOut
        self.entire = entire

        # Other
        self.writeAttrs = writeAttrs
        self.storeReferences = storeReferences
        self._references = {}
        self.scores = Bunch()
        self._declaration = self.get_params(deep=False)

    @property
    def declaration(self):
        """
        Dictionary of kwargs given during instantiation
        """
        return {i: clone(j) for i, j in self._declaration.items()}

    @subset_dataset('train')
    def build_cross_validation(self, dataset, **fitParams):
        """
        Build cross validated scores over training data of models
        """
        print('Building Over Cross Validation')
        payload = {'estimator': self.estimator, 'metrics': self.metrics, 'dataset': dataset}
        self.scorerCrossValidation = self.scorer.buildCV(**payload)
        self.scores.crossValidation = Bunch(**self.scorerCrossValidation['scores'])
        self._references['cross_validation'] = deepcopy(self.estimator) if self.storeReferences else None

    def build_holdout(self, dataset, **fitParams):
        """
        Build model over training data and score
        """
        print('Building Over Holdout')
        self.estimator.fit(X=dataset.designTrain, y=dataset.targetTrain, gridSearch=True, **fitParams)

        payload = {'estimator': self.estimator, 'metrics': self.metrics,
                   'X': dataset.designTest, 'y': dataset.targetTest}
        self.scorerHoldout = self.scorer.build_holdout(**payload)
        self.scores.holdout = Bunch(**self.scorerHoldout['scores'])
        self._references['holdout'] = deepcopy(self.estimator) if self.storeReferences else None

    def build_entire(self, dataset, **fitParams):
        """
        Build model over entire data set
        """
        print('Building Over Entire Dataset')
        self.estimator = clone(self.estimator)
        self.estimator.fit(X=dataset.designData, y=dataset.targetData,
                           gridSearch=True, **fitParams)
        self._references['entire'] = deepcopy(self.estimator) if self.storeReferences else None

    @fallback('dataset', 'writeAttrs', 'validation', 'holdOut', 'entire')
    @package_dataset
    def fit(self, dataset=None, X=None, y=None, writeAttrs=None,
            validation=None, holdOut=None, entire=None, **fitParams):
        """
        Build models, tune hyperparameters, and evaluate
        """

        self.build_cross_validation(dataset, **fitParams) if validation else None
        self.build_holdout(dataset, **fitParams) if holdOut else None
        self.build_entire(dataset, **fitParams) if entire else None

        self.write(writeAttrs=writeAttrs)

        return self

    @fallback('writeAttrs')
    def write(self, writeAttrs=None):
        """
        Write objects
        """
        [self.hook.write(obj=self, attr=attr) for attr in writeAttrs]


class Garden(Dobject):
    """
    Collection of Sculpture with support for comparisons
    """
    pass
