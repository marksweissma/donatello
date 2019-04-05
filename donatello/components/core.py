"""
Manager for model building and evaluating
"""
from copy import deepcopy
from sklearn import clone
from sklearn.base import BaseEstimator
from sklearn.utils import Bunch

from donatello.components.data import package_dataset, subset_dataset
from donatello.components.measure import Measure

from donatello.utils.helpers import now_string, persist
from donatello.utils.decorators import fallback, coelesce
from donatello.utils.base import Dobject


class Sculpture(Dobject, BaseEstimator):
    """
    Instance for model processing. Accessors will fallback to accessing from estimator attributes

    Args:
        dataset (donatello.components.data.Dataset): accesses  dataset
        estimator(donatello.components.estimator.Estimator): executes ML
        validation (bool): flag for calculating scoring metrics from nested\
                cross val of training + validation sets
        holdout (bool): flag for fitting estimator on single training set and scoring on test set
        measure (donatello.components.measure.Measure): own calculateing stats
        metrics (iterable.Metric): :py:class:`donatello.components.measure.Metric` to score
        persist (func): function to write
        persistAttrs (tuple): attributes to write out to disk
        timeFormat (str): format for creation time string
    """

    def __init__(self,
                 dataset=None,
                 estimator=None,
                 validation=True, holdout=False, entire=False,
                 measure=Measure(), persist=persist, metrics=None,
                 storeReferences=True, writeAttrs=('', 'estimator'),
                 timeFormat="%Y_%m_%d_%H_%M"):

        self._initTime = now_string(timeFormat)

        self.dataset = dataset
        self.estimator = estimator

        self.measure = measure
        self.persist = persist

        self.metrics = metrics

        # Build options
        self.validation = validation
        self.holdout = holdout
        self.entire = entire

        # Other
        self.writeAttrs = writeAttrs
        self.storeReferences = storeReferences
        self._references = {}
        self.measurements = Bunch()
        self._declaration = self.get_params(deep=False).copy()

    @property
    def declaration(self):
        """
        Dictionary of kwargs given during instantiation
        """
        return self._declaration.copy()

    @fallback('estimator', 'metrics')
    @subset_dataset('train')
    def build_cross_validation(self, dataset, estimator=None, metrics=None, **fitParams):
        """
        Build cross validated measurements over training data of models
        """
        print('Cross Validation')
        estimator = clone(estimator)
        payload = {'estimator': estimator, 'metrics': metrics, 'dataset': dataset}
        self.measureCrossValidation = self.measure.buildCV(**payload)
        self.measurements.crossValidation = Bunch(**self.measureCrossValidation['measurements'])
        self._references['cross_validation'] = deepcopy(estimator) if self.storeReferences else None

    @fallback('estimator', 'metrics')
    def build_holdout(self, dataset, estimator=None, metrics=None, **fitParams):
        """
        Build model over training data and score
        """
        print('Holdout')
        estimator = clone(estimator)
        estimator.fit(dataset=dataset.subset('train'), gridSearch=True, **fitParams)

        payload = {'estimator': estimator, 'metrics': metrics,
                   'X': dataset.designTest, 'y': dataset.targetTest}
        self.measureHoldout = self.measure.build_holdout(**payload)
        self.measurements.holdout = Bunch(**self.measureHoldout['measurements'])
        self._references['holdout'] = deepcopy(estimator) if self.storeReferences else None

    @fallback('estimator')
    def build_entire(self, dataset, estimator=None, **fitParams):
        """
        Build model over entire data set
        """
        print('Entire Dataset')
        estimator = clone(estimator)
        estimator.fit(dataset=dataset, gridSearch=True, **fitParams)

        if self.storeReferences:
            self._references['entire'] = deepcopy(estimator)
            self.estimator = estimator
        else:
            self._references['entire'] = None

    @fallback('dataset', 'writeAttrs', 'validation', 'holdout', 'entire')
    @package_dataset
    def fit(self, X=None, y=None, dataset=None, writeAttrs=None,
            validation=None, holdout=None, entire=None, **fitParams):
        """
        Build models, tune hyperparameters, and evaluate
        """

        self.build_cross_validation(dataset, **fitParams) if validation else None
        self.build_holdout(dataset, **fitParams) if holdout else None
        self.build_entire(dataset, **fitParams) if entire else None

        self.write(writeAttrs=writeAttrs)

        return self

    @fallback('writeAttrs')
    @coelesce(writeAttrs=[])
    def write(self, writeAttrs=None):
        """
        Write objects
        """
        for attr in writeAttrs:
            payload = attr if isinstance(attr, dict) else {'attr': attr}
            self.persist(obj=self, **payload)

    def __getattr__(self, attr):
        return getattr(self.estimator, attr) if attr != '_name' else self.__class__.__name__


class Garden(Dobject):
    """
    Front end enabler for visualization and comparison
    """
    pass
