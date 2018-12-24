import pandas as pd
import numpy as np

from collections import defaultdict
from abc import abstractproperty
from warnings import warn

from sklearn import clone
from sklearn.utils import Bunch
from sklearn.metrics import confusion_matrix

from donatello.utils.decorators import init_time, fallback
from donatello.utils.base import Dobject
from donatello.components.data import package_dataset


class Scorer(Dobject):
    """
    Object for scoring model performance

    Args:
        mlClay (str): denotes ml context classification / regression / clustering etc
        method (str): name of prediction method from estimator to call
        gridSearchFlag (bool): whether or not to grid search during fitting
    """
    @init_time
    def __init__(self,
                 mlClay=None,
                 method=None,
                 gridSearchFlag=True
                 ):

        # Preserve Args
        self.mlClay = mlClay
        self.gridSearchFlag = gridSearchFlag
        self.method = method

    @abstractproperty
    def name(self):
        name = self.__class__.__name__
        warn('Defaulting to *{name}*'.format(name=name))
        return name

    def feature_weights(self, estimator=None, attr='', **kwargs):
        """
        Extract feature weights from a model

        Will automatically pull `coef_` and `feature_importances_`

        Args:
            estimator (donatello.Estimator): has `features` and `model` attributes
            attr (str): option to specify additional attribute to pull

        Returns:
            pandas.DataFrame: featureValues
        """

        names = estimator.features
        model = estimator.model
        columnNames = ['names']
        values = []
        if hasattr(model, attr):
            columnNames.append(attr)
            values.append(getattr(model, attr))
        if hasattr(model, 'coef_'):
            columnNames.append('coefficients')
            if hasattr(model, 'intercept_'):
                names.append('intercept_')
                values.append(np.hstack((model.coef_[0], model.intercept_)))
            else:
                values.append(model.coef_[0])
        if hasattr(model, 'feature_importances_'):
            columnNames.append('feature_importances')
            values.append(model.feature_importances_)
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

    Args:
        predict (str)_method: method to call from estimator for predicting
        **kwargs: kwargs for Scorer
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
                _output = metric(scored.truth, scored.predicted, **definition.get('metricKwargs', {}))
                output = pd.DataFrame([[1, _output]], columns=['_'] + columnNames)

            elif hasattr(self, metric):
                payload = {'estimator': estimator, 'scored': scored}
                payload.update(definition.get('kwargs', {}))
                output = getattr(self, metric)(**payload)
            else:
                warn('metric {metric} inaccesible'.format(metric=metric))
                output = None

            # collect and concat
            scores[name] = scores[name].append(output)

        return scores

    def score_evaluate(self, estimator=None, X=None, y=None, metrics=None):
        """
        Score the fitted estimator on y and evaluate metrics

        Args:
            estimator (BaseEstimator): Fit estimator to evaluate
            X (pandas.DataFrame): design
            y (pandas.Series): target
            metrics (dict): metrics to evaluate

        Returns:
            tuple(pandas.Series, metric evaluations): scored, scores
        """
        scored = self._score(estimator, X, y)
        scores = self._evaluate(estimator, scored, metrics)
        return scored, scores

    @package_dataset
    def fit_score_folds(self, estimator=None, dataset=None, X=None, y=None, **kwargs):
        """
        Cross validating scorer, clones and fits estimator on each fold of X|y

        Args:
            estimator (BaseEstimator): Fit estimator to evaluate
            dataset (donatello.data.Dataset) object to cross val over
            X (pandas.DataFrame): design
            y (pandas.Series): target
            metrics (dict): metrics to evaluate

        Returns:
            tuple(pandas.Series, metric evaluations): scored, scores
        """
        scored = pd.DataFrame()
        estimators = {}

        for fold, (designTrain, designTest, targetTrain, targetTest) in enumerate(dataset):
            estimator = clone(estimator)
            estimator.fit(X=designTrain, y=targetTrain, gridSearch=self.gridSearchFlag)
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
                output = Bunch(**{key: _option_sort(df.xs(key, level=current, axis=1), definitionSort)
                                  for key in set(df.columns.get_level_values(current))})
            return output

        [metric.fit(scored) for metric in metrics]

        for fold, df in scored.groupby('fold'):
            _outputs = self._evaluate(estimators[fold], df, metrics)
            [append_in_place(outputs, name, df) for name, df in _outputs.items()]

        scores = {self.get_metric_name(metric): _unwrap_multiple(
                                                outputs[self.get_metric_name(metric)]\
                                               .groupby(definition.get('key', ['_']))\
                                               .agg(definition.get('agg', ['mean', 'std'])),
                                                definition.get('sort', None)
                                                )
                  for metric, definition in metrics.items()
                  }

        # fix this, move to metric obj
        for metric, definition in metrics.items():
            callback = definition.get('callback', '')
            callbackKwargs = definition.get('callbackKwargs', {})
            name = self.get_metric_name(metric)
            func = callback if callable(callback) else None
            scores.update({name: func(scores[name], **callbackKwargs)}) if func else None

        return scores

    @package_dataset
    def buildCV(self, estimator=None, metrics=None, dataset=None, X=None, y=None):
        """
        Build cross validated scoring report
        """
        estimators, scored = self.fit_score_folds(estimator=estimator, dataset=dataset)
        scores = self.evaluate_scored_folds(estimators=estimators, scored=scored, X=dataset.designData, metrics=metrics)
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
