import pandas as pd
import numpy as np

from collections import defaultdict
from abc import ABCMeta, abstractproperty
from warnings import warn

from sklearn import clone
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix

from donatello.utils.helpers import nvl
from donatello.utils.decorators import init_time
from donatello.utils.base import Dobject
from donatello.components.data import package_data


class BaseScorer(Dobject):
    """
    Base class for evaluating estimators and datasets

    :param type splitType: Type of bject with :py:meth:`split` for cross val scoring
    :param dict splitKwargs: arguments for splitType to instantiate :py:attr:`BaseScorer.splitter`
    :param str predict_method: method to call from estimator for predicting
    """
    __meta__ = ABCMeta

    @init_time
    def __init__(self,
                 splitType=None,
                 splitKwargs={'n_splits': 5,
                              'shuffle': True,
                              'random_state': 22},
                 method='score',
                 gridSearchFlag=True,
                 verbose=True,
                 nowFormat="%Y_%m_%d_%H_%M"
                 ):

        # Preserve Args
        self.splitType = splitType
        self.splitKwargs = splitKwargs
        self.gridSearchFlag = gridSearchFlag
        self.method = method
        self.verbose = verbose

        self.splitter = splitKwargs

    def __repr__(self):
        return '{name} created at {time}'.format(name=self.name,
                                                 time=self._initTime)

    @abstractproperty
    def name(self):
        name = self.__class__.__name__
        warn('Defaulting to *{name}*'.format(name=name))
        return name

    @property
    def splitter(self):
        """
        Splitter to create folds
        """
        return self._splitter

    @splitter.setter
    def splitter(self, splitKwargs):
        self._splitter = self.splitType(**splitKwargs)

    def _dev_features(self, estimator=None, attr='', **kwargs):
        """
        Extract feature weights from a model

        Will automatically pull `coef_` and `feature_importances_`

        :param donatello.Estimator estimator: has `features` and `model` attributes
        :param str attr: option to specify additional attribute to pull
        :return: featureValues
        :rtype: :py:class:`pandas.DataFrame`
        """

        names = estimator.features
        model = estimator.model
        columnNames = ['names']
        values = []
        if hasattr(model, attr):
            columnNames.extend(attr)
            values.append([getattr(model, attr)])
        if hasattr(model, 'feature_importances_'):
            columnNames.extend('feature_importances')
            values.append([model.feature_importances_])
        if hasattr(model, 'coef_'):
            columnNames.extend('coefficients')
            if hasattr(model, 'intercept_'):
                names.append('intercept_')
                values.append([np.hstack((model.coef_[0], model.intercept_))])
            else:
                values.append([model.coef_[0]])
        if values:
            names = np.asarray(names)
            featureValues = pd.DataFrame(data=np.c_[names, values], columns=columnNames)
            return featureValues

    @staticmethod
    def get_metric_name(metric, default=''):
        """
        Helper to get string name of metric
        """
        return metric if isinstance(metric, str) else getattr(metric, '__name__', str(default))

    def _score(self, estimator, designTest, targetTest):
        yhat = estimator.score(designTest)
        scored = pd.concat([targetTest.rename('truth'), yhat.rename('predicted')], axis=1)
        return scored

    def _evaluate(self, estimator, scored, metrics):
        _increment = 0
        scores = defaultdict(pd.DataFrame)
        for metric, definition in metrics.iteritems():
            _increment += 1
            name = self.get_metric_name(metric, _increment)

            print 'evaluating {name}'.format(name=name)

            if callable(metric):
                columnNames = definition.get('columnNames', ['score'])
                _output = metric(scored.truth, scored.predicted, **definition['metricKwargs'])
                output = pd.DataFrame([[1, _output]], columns=['_'] + columnNames)

            elif hasattr(self, metric):
                payload = {'estimator': estimator, 'scored': scored}
                payload.update(definition['kwargs'])
                output = getattr(self, metric)(**payload)
            else:
                warn('metric {metric} inaccesible'.format(metric=metric))

            scores[name] = scores[name].append(output)

        return scores

    def score_evaluate(self, estimator=None, X=None, y=None, metrics=None):
        """
        Score the fitted estimator on y and evaluate metrics

        :param BaseEstimator estimator: Fit estimator to evaluate
        :param pandas.DataFrame X: design
        :param pandas.Series y: target
        :param dict metrics: metrics to evaluate
        :return: scored, scores
        :rtype: pandas.Series, metric evaluations
        """
        scored = self._score(estimator, X, y)
        scores = self._evaluate(estimator, scored, metrics)
        return scored, scores

    @package_data
    def fit_score_folds(self, estimator=None, data=None, X=None, y=None, **kwargs):
        """
        Cross validating scorer, clones and fits estimator on each fold of X|y

        :param BaseEstimator estimator: Fit estimator to evaluate
        :param donatello.Data data: data object to cross val over
        :param pandas.DataFrame X: design
        :param pandas.Series y: target
        :param dict metrics: metrics to evaluate
        :return: scored, scores
        :rtype: pandas.Series, metric evaluations
        """
        scored = pd.DataFrame()
        estimators = {}

        for fold, (designTrain, designTest, targetTrain, targetTest) in enumerate(data):
            estimator = clone(estimator)
            estimator.fit(designTrain, targetTrain, gridSearch=self.gridSearchFlag)
            estimators[fold] = estimator

            _temp = self._score(estimator, designTest, targetTest)
            _temp['fold'] = fold
            scored = scored.append(_temp)
        return estimators, scored

    def evaluate_scored_folds(self, estimators=None, metrics=None, scored=None, X=None, **kwargs):
        """
        Calculate metrics from cross val scores
        """
        outputs = defaultdict(pd.DataFrame)

        def append_in_place(store, name, df2):
            store[name] = store[name].append(df2)

        for fold, df in scored.groupby('fold'):
            _outputs = self._evaluate(estimators[fold], df, metrics)
            [append_in_place(outputs, name, df) for name, df in _outputs.items()]

        scores = {self.get_metric_name(name): outputs[self.get_metric_name(name)]\
                                             .groupby(definition.get('key', ['_']))\
                                             .agg(definition.get('agg', pd.np.mean))
                  for metric, definition in metrics.iteritems()
                  }
        return scores

    @package_data
    def buildCV(self, estimator=None, metrics=None, data=None, X=None, y=None):
        """
        Build cross validated scoring report
        """
        estimators, scored = self.fit_score_folds(estimator=estimator, data=data)
        scores = self.evaluate_scored_folds(estimators=estimators, scored=scored, X=data.designData, metrics=metrics)
        return {'estimators': estimators, 'scored': scored, 'scores': scores}

    def build_holdout(self, estimator=None, metrics=None, X=None, y=None):
        """
        Build cross validated scoring report
        """
        scored, scores = self.score_evaluate(estimator=estimator, metrics=metrics,
                                             X=X, y=y)
        estimators = {0: estimator}
        return {'estimators': estimators, 'scored': scored, 'scores': scores}


class ScorerClassification(BaseScorer):
    """
    Scorer for classifcation models
    """
    _mlType = 'Classification'

    def __init__(self, thresholds=None, spacing=101, splitType=StratifiedKFold, **kwargs):
        payload = kwargs
        payload.update({'splitType': splitType})
        super(ScorerClassification, self).__init__(**payload)
        self.thresholds = thresholds
        self.spacing = spacing

    def find_thresholds(self, scored, thresholds=None, spacing=101, **kwargs):
        if self.thresholds is None:
            self.thresholds = np.linspace(0, 1, spacing)
        else:
            self.spacing = len(thresholds)

        percentiles = np.linspace(0, 1, spacing)
        self.thresholds = scored.predicted.quantile(percentiles)

    def score_folds(self, estimator=None, data=None):
        estimators, scored = super(ScorerClassification, self).score_folds(self, estimator=estimator, data=data)
        self.find_tresholds(scored, thresholds=self.thresholds, spacing=self.spacing)
        return estimators, scored

    def threshold_rates(self, scored=None, thresholds=None, spacing=101, **kwargs):
        """
        """
        thresholds = nvl(thresholds, self.thresholds)

        # Vectorize this, threshold iter is slow
        data = np.array([np.hstack((i,
                                    confusion_matrix(scored.truth.values, (scored.predicted > i).values).reshape(4,)
                                    )
                                   ) for i in thresholds])

        output = pd.DataFrame(data=data,
                              columns=['thresholds', 'true_negative', 'false_positive', 'false_negative', 'true_positive'],
                              index=range(spacing))
        return output

    @staticmethod
    def build_threshold_rates(df):
        df = df / df.sum(axis=1)
        df['false_omission_rate'] = df.false_negative / (df.false_negative + df.true_negative)
        df['f1'] = 2 * df.true_positive / (2 * df.true_positive + df.false_positive + df.false_negative)
        df['sensitivity'] = df.true_positive / (df.true_positive + df.false_negative)
        df['specificity'] = df.true_negative / (df.true_negative + df.false_positive)
        df['precision'] = df.true_positive / (df.true_positive + df.false_positive)
        df['negative_predictive_value'] = df.true_negative / (df.true_negative + df.false_negative)
        df['fall_out'] = 1 - df.specificity
        df['false_discovery_rate'] = 1 - df.precision

        return df


class ScorerRegression(BaseScorer):
    """
    Scorer for regression models
    """
    _mlType = 'Regression'

    def __init__(self, splitType=KFold, **kwargs):
        payload = kwargs
        payload.update({'splitType': splitType})
        super(ScorerClassification, self).__init__(**payload)
