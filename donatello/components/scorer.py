import pandas as pd
import numpy as np

from collections import defaultdict
from abc import ABCMeta, abstractproperty
from warnings import warn

from sklearn import clone
from sklearn.utils import Bunch
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix

from donatello.utils.helpers import nvl
from donatello.utils.decorators import init_time
from donatello.utils.base import Dobject
from donatello.components.data import package_data


class Scorer(Dobject):
    @init_time
    def __init__(self,
                 splitType=None,
                 splitKwargs={'n_splits': 5,
                              'shuffle': True,
                              'random_state': 22},
                 method=None,
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

    def feature_weights(self, estimator=None, attr='', **kwargs):
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
            columnNames.append(attr)
            values.append(getattr(model, attr))
        if hasattr(model, 'feature_importances_'):
            columnNames.append('feature_importances')
            values.append(model.feature_importances_)
        if hasattr(model, 'coef_'):
            columnNames.append('coefficients')
            if hasattr(model, 'intercept_'):
                names.append('intercept_')
                values.append(np.hstack((model.coef_[0], model.intercept_)))
            else:
                values.append(model.coef_[0])
        if values:
            names = pd.Series(np.asarray(names), name=columnNames[0])
            vectors = pd.DataFrame(np.asarray(values).T, columns=columnNames[1:])

            data = pd.concat([names, vectors], axis=1)
            return data

    @staticmethod
    def get_metric_name(metric, default=''):
        """
        Helper to get string name of metric
        """
        return metric if isinstance(metric, str) else getattr(metric, '__name__', str(default))


class ScorerSupervised(Scorer):
    """
    Base class for evaluating estimators and datasets

    :param type splitType: Type of bject with :py:meth:`split` for cross val scoring
    :param dict splitKwargs: arguments for splitType to instantiate :py:attr:`BaseScorer.splitter`
    :param str predict_method: method to call from estimator for predicting
    """
    def __init__(self,
                 method='score',
                 **kwargs
                 ):

        super(ScorerSupervised, self).__init__(method=method, **kwargs) 

    def _score(self, estimator, designTest, targetTest):
        yhat = getattr(estimator, self.method)(designTest)
        scored = pd.concat([targetTest.rename('truth'), yhat.rename('predicted')], axis=1)
        return scored

    def _evaluate(self, estimator, scored, metrics):
        _increment = 0
        scores = defaultdict(pd.DataFrame)
        for metric, definition in metrics.items():
            _increment += 1
            name = self.get_metric_name(metric, _increment)

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
            estimator.fit(designTrain, y=targetTrain, gridSearch=self.gridSearchFlag)
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

        def _option_sort(df, sort):
            df = df.sort_values(sort) if sort else df
            return df

        def _unwrap_multiple(df, definitionSort):
            levels = df.columns.nlevels
            current = levels - 1
            if not current:
                output = df
            else:
                output = {key: _option_sort(df.xs(key, level=current, axis=1), definitionSort)
                          for key in set(df.columns.get_level_values(current))}
            return output

        for fold, df in scored.groupby('fold'):
            _outputs = self._evaluate(estimators[fold], df, metrics)
            [append_in_place(outputs, name, df) for name, df in _outputs.items()]

        scores = {self.get_metric_name(metric): _unwrap_multiple(
                                                outputs[self.get_metric_name(metric)]\
                                               .groupby(definition.get('key', ['_']))\
                                               .agg(definition.get('agg', pd.np.mean)),
                                               definition.get('sort', None)
                                               )
                  for metric, definition in metrics.items()
                  }

        # fix this, move to metric obj
        for metric, definition in metrics.items():
            callback = definition.get('callback', '')
            callbackKwargs = definition.get('callbackKwargs', {})
            name = self.get_metric_name(metric)
            func = callback if callable(callback) else getattr(self, callback, {})
            information =  scores[name]
            # fix this with dispatch
            if isinstance(information, dict):
                _hold = Bunch(**{agg: func(df, **callbackKwargs) if func else df for agg, df in information.items()})
            else:
                _hold = func(information, **callbackKwargs) if func else information
            scores[name] = _hold

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
        scored = self._score(estimator, X, y)
        scored['fold'] = 0
        estimators = {0: estimator}
        scores = self.evaluate_scored_folds(estimators=estimators, scored=scored, X=X, metrics=metrics)
        return {'estimators': estimators, 'scored': scored, 'scores': scores}


class ScorerClassification(ScorerSupervised):
    """
    Scorer for classifcation models
    """
    _mlType = 'classification'

    def __init__(self, thresholds=None, spacing=101, splitType=StratifiedKFold, **kwargs):
        payload = kwargs
        payload.update({'splitType': splitType})
        super(ScorerClassification, self).__init__(**payload)
        self.thresholds = thresholds
        self.spacing = spacing

    def find_thresholds(self, scored, thresholds=None, spacing=101, **kwargs):
        if not thresholds:
            percentiles = np.linspace(0, 1, spacing)
            self.thresholds = scored.predicted.quantile(percentiles)
        else:
            self.thresholds = thresholds

    def evaluate_scored_folds(self, estimators=None, metrics=None, scored=None, X=None, **kwargs):
        self.find_thresholds(scored)
        return super(ScorerClassification, self).evaluate_scored_folds(
                     estimators=estimators, metrics=metrics, scored=scored, X=X, **kwargs)

    def threshold_rates(self, scored=None, thresholds=None, spacing=101, threshKwargs={}, **kwargs):
        """
        """
        thresholds = nvl(thresholds, self.thresholds)

        # Vectorize this, threshold iter is slow
        data = np.array([np.hstack((i,
                                    confusion_matrix(scored.truth.values, (scored.predicted > i).values).reshape(4,)
                                    )
                                   ) for i in thresholds])

        df = pd.DataFrame(data=data,
                          columns=['thresholds', 'true_negative', 'false_positive', 'false_negative', 'true_positive'],
                          index=range(spacing))

        df = df.apply(lambda x: x / np.sum(x))
        df['false_omission_rate'] = df.false_negative / (df.false_negative + df.true_negative)
        df['f1'] = 2 * df.true_positive / (2 * df.true_positive + df.false_positive + df.false_negative)
        df['sensitivity'] = df.true_positive / (df.true_positive + df.false_negative)
        df['specificity'] = df.true_negative / (df.true_negative + df.false_positive)
        df['precision'] = df.true_positive / (df.true_positive + df.false_positive)
        df['negative_predictive_value'] = df.true_negative / (df.true_negative + df.false_negative)
        df['fall_out'] = 1 - df.specificity
        df['false_discovery_rate'] = 1 - df.precision
        return df


class ScorerRegression(ScorerSupervised):
    """
    Scorer for regression models
    """
    _mlType = 'regression'

    def __init__(self, splitType=KFold, **kwargs):
        payload = kwargs
        payload.update({'splitType': splitType})
        super(ScorerClassification, self).__init__(**payload)


class ScorerUnsupervised(Scorer):
    pass
