import pandas as pd

from collections import defaultdict

from sklearn import clone
from sklearn.utils import Bunch

from donatello.utils.decorators import init_time
from donatello.utils.base import Dobject
from donatello.components.data import package_dataset


class Scorer(Dobject):
    """
    Object for scoring model performance

    Args:
        scoreClay (str): denotes ml context classification / regression / clustering etc
        method (str): name of prediction method from estimator to call
        gridSearchFlag (bool): whether or not to grid search during fitting
    """
    @init_time
    def __init__(self,
                 foldClay=None,
                 scoreClay=None,
                 method=None,
                 gridSearchFlag=True
                 ):

        # Preserve Args
        self.foldClay = foldClay
        self.scoreClay = scoreClay
        self.gridSearchFlag = gridSearchFlag
        self.method = method


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

    def _evaluate(self, estimator, scored, metrics, X):
        scores = {metric.name: metric(scored.truth, scored.predicted, X) for metric in metrics}
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
                output = Bunch(**{key: _option_sort(df.xs(key, level=current, axis=1).astype(float), definitionSort)
                                  for key in set(df.columns.get_level_values(current))})
            return output

        [metric.fit(scored) for metric in metrics]

        # move to list of dict -> concat
        outputs = defaultdict(pd.DataFrame)
        for fold, df in scored.groupby('fold'):
            _outputs = self._evaluate(estimators[fold], df, metrics, X)
            [append_in_place(outputs, name, df) for name, df in _outputs.items()]

        scores = {metric.name: metric.callback(_unwrap_multiple(outputs[metric.name]\
                                                                .groupby(metric.key)\
                                                                .agg(metric.agg),
                                                                metric.sort))
                  for metric in metrics}

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
        score already fit estimator
        """
        scored = self._score(estimator, X, y)
        scored['fold'] = 0
        estimators = {0: estimator}
        scores = self.evaluate_scored_folds(estimators=estimators, scored=scored, X=X, metrics=metrics)
        return {'estimators': estimators, 'scored': scored, 'scores': scores}
