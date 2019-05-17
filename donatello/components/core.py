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
from donatello.utils.decorators import fallback
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
                 dataset=None, outsideData=None,
                 estimator=None,
                 validation='search', holdout='search', entire=False,
                 measure=Measure(), persist=persist, metrics=None,
                 storeReferences=True, writeAttrs=('', 'estimator'),
                 timeFormat="%Y_%m_%d_%H_%M"):

        self._initTime = now_string(timeFormat)

        self.dataset = dataset
        self.outsideData = outsideData
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
        # self.estimator = cv_rule(self.measureCrossValidation['estimators'].values())
        self.estimator = self.measureCrossValidation['estimators'].values()[0]

    @fallback('estimator', 'metrics')
    def build_holdout(self, dataset, estimator=None, metrics=None, **fitParams):
        """
        Build model over training data and score
        """
        print('Holdout')
        estimator = clone(estimator)
        estimator.fit(dataset=dataset.subset('train'), **fitParams)

        payload = {'estimator': estimator, 'metrics': metrics,
                   'X': dataset.designTest, 'y': dataset.targetTest}
        self.measureHoldout = self.measure.build_holdout(**payload)
        self.measurements.holdout = Bunch(**self.measureHoldout['measurements'])
        self._references['holdout'] = deepcopy(estimator) if self.storeReferences else None
        self.estimator = estimator

    @fallback('estimator', 'metrics', 'outsideData')
    def build_entire(self, dataset, estimator=None, metrics=None, outsideData=None, **fitParams):
        """
        Build model over entire data set
        """
        print('Entire Dataset')
        estimator = clone(estimator)
        estimator.fit(dataset=dataset, **fitParams)

        if self.storeReferences:
            self._references['entire'] = deepcopy(estimator)
            self.estimator = estimator
        else:
            self._references['entire'] = None

        if outsideData:
            payload = {'estimator': estimator, 'metrics': metrics,
                       'X': outsideData.designData, 'y': outsideData.targetData}
            self.measureOutside = self.measure.build_holdout(**payload)
            self.measurements.outside = Bunch(**self.measureOutside['measurements'])

    @fallback('dataset', 'writeAttrs', 'validation', 'holdout', 'entire', 'outsideData')
    @package_dataset
    def fit(self, X=None, y=None, dataset=None, writeAttrs=None,
            validation=None, holdout=None, entire=None, outsideData=None,
            **fitParams):
        """
        Build models, tune hyperparameters, and evaluate
        """

        if validation:
            self.build_cross_validation(dataset,
                    gridSearch=(validation=='search'),
                    **fitParams)

        if holdout:
            self.build_holdout(dataset,
                    gridSearch=(holdout=='search'),
                    **fitParams)

        if entire:
            self.build_entire(dataset,
                    gridSearch=(entire=='search'),
                    outsideData=outsideData,
                    **fitParams)

        self.write(writeAttrs=writeAttrs)

        return self

    @fallback('writeAttrs')
    def write(self, writeAttrs=()):
        """
        Write objects
        """
        [self.persist(obj=self, dap=attr) for attr in writeAttrs]

    @property
    def references(self):
        return self._references

    def __getattr__(self, attr):
        return getattr(self.estimator, attr) if attr != '_name' else self.__class__.__name__


class Garden(Dobject):
    """
    Front end enabler for visualization and comparison
    """
    pass
