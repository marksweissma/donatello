"""
Donatello Manager for orchestrating end to end ML
"""
from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.utils import Bunch

from donatello.components.folder import Folder
from donatello.components.data import Dataset, package_dataset
from donatello.components.estimator import Estimator
from donatello.components.scorer import ScorerSupervised
from donatello.components.disk import Local

from donatello.utils.helpers import has_nested_attribute, now_string
from donatello.utils.decorators import fallback, fold_dataset
from donatello.utils.base import Dobject


class Sculpture(Dobject, BaseEstimator):
    """
    Manager for model process. [a-z]*Declaration parameters map 1:1 to
    component objects attached via property setters. Other parameters
    attached directly.

    Manager accessors will fallback to accessing from estimator attributes

    Args:
        dataDeclaration (dict): :py:class:`donatello.Dataset`
        folderDeclaration (dict): arguments for :py:class:`donatello.folder.Folder`
        estimatorDeclaration (dict): arguments for :py:class:`donatello.Estimator.estimator`
        scorerDeclaration (dict): arguments for :py:class:`donatello.Scorer`
        validation (bool): flag for calculating scoring metrics from nested cross val of training + validation sets
        holdOut (bool): flag for fitting estimator on single training set and scoring on test set
        metrics (iterable): list or dict of metrics for scorer
        hookDeclaration (dict): arguments for :py:class:`donatello.Local`
        writeAttrs (tuple): attributes to write out to disk
        timeFormat (str): format for creation time string
    """

    def __init__(self, dataDeclaration=None, folderDeclaration=None,
                 estimatorDeclaration=None, scorerDeclaration=None,
                 validation=True, holdOut=False, entire=False, metrics=None,
                 scoreClay=None, foldClay=None,
                 scoreType=ScorerSupervised, foldType=Folder,
                 hook=Local(), storeReferences=True,
                 writeAttrs=('', 'estimator'), timeFormat="%Y_%m_%d_%H_%M"):

        self._initTime = now_string(timeFormat)

        self.folderDeclaration = folderDeclaration
        self.datasetDeclaration = dataDeclaration
        self.estimatorDeclaration = estimatorDeclaration
        self.scorerDeclaration = scorerDeclaration

        self.scoreClay = scoreClay
        self.scoreType = scoreType

        self.foldClay = foldClay
        self.foldType = foldType

        self.metrics = metrics

        # Build options
        self.validation = validation
        self.holdOut = holdOut
        self.entire = entire

        self.dataset = dataDeclaration
        self.dataParams = self.dataset.params

        self.folder = folderDeclaration
        self.estimator = estimatorDeclaration
        self.scorer = scorerDeclaration
        self.hook = hook

        # Other
        self.writeAttrs = writeAttrs
        self.storeReferences = storeReferences
        self._references = {}
        self.scores = Bunch()
        self.declaration = self.get_params(deep=False)

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
    def folder(self):
        """
        Folder object attached to manager
        """
        return self._folder

    @folder.setter
    def folder(self, kwargs):
        kwargs = self._update_to(kwargs,  'foldClay', 'scoreClay')
        self._folder = self.foldType(**kwargs)

    @property
    def dataset(self):
        """
        Dataset object attached to manager
        """
        return self._dataset

    @dataset.setter
    def dataset(self, kwargs):
        kwargs = self._update_to(kwargs,  'foldClay', 'scoreClay')
        self._dataset = Dataset(**kwargs)

    @property
    def estimator(self):
        """
        Dataset object attached to manager
        """
        return self._estimator

    @estimator.setter
    def estimator(self, kwargs):
        kwargs = self._update_to(kwargs,  'foldClay', 'scoreClay')
        self._estimator = Estimator(**kwargs)

    @property
    def scorer(self):
        """
        Scorer object attached to manager
        """
        return self._scorer

    @scorer.setter
    def scorer(self, kwargs):
        kwargs = self._update_to(kwargs,  'foldClay', 'scoreClay')
        self._scorer = self.scoreType(**kwargs)

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

    @fallback('writeAttrs', 'validation', 'holdOut', 'entire')
    @package_dataset
    @fold_dataset
    def fit(self, dataset=None, X=None, y=None, writeAttrs=None,
            validation=None, holdOut=None, entire=None, **fitParams):
        """
        Build models, tune hyperparameters, and evaluate
        """

        self._build_cross_validation(dataset, **fitParams) if validation else None
        self._build_holdout(dataset, **fitParams) if holdOut else None
        self._build_entire(dataset, **fitParams) if entire else None

        self.write(writeAttrs=writeAttrs)

        return self

    def has_attribute(self, attr):
        return has_nested_attribute(self, attr)

    @fallback('writeAttrs')
    def write(self, writeAttrs=None):
        """
        Write objects
        """
        [self.hook.write(obj=self, attr=attr) for attr in filter(self.has_attribute, writeAttrs)]
