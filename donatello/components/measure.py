import pandas as pd
import numpy as np

from collections import defaultdict

from sklearn import clone
from sklearn.utils import Bunch
from sklearn.metrics import confusion_matrix

from donatello.utils.decorators import init_time, coelesce, name
from donatello.utils.base import Dobject
from donatello.components.data import package_dataset


def pass_through(x):
    """
    returns x
    """
    return x


# FIX THIS
def _append_in_place(store, name, df2):
    store[name] = store[name].append(df2)


def _option_sort(df, sort):
    df = df.sort_values(sort) if sort else df
    return df


# move off xs
def _unwrap_multiple(df, definitionSort):
    levels = df.columns.nlevels
    current = levels - 1
    if not current:
        output = df
    else:
        output = Bunch(**{key: _option_sort(df.xs(key, level=current, axis=1).astype(float),
                                            definitionSort) for key in set(df.columns.get_level_values(current))})
    return output


class Measure(Dobject):
    """
    Object for scoring model performance

    Args:
        method (str): name of prediction method from estimator to call
        gridSearchFlag (bool): whether or not to grid search during fitting
    """
    @init_time
    def __init__(self,
                 method='score',
                 gridSearchFlag=True
                 ):

        self.method = method
        self.gridSearchFlag = gridSearchFlag

    def _score(self, estimator, designTest, targetTest):
        yhat = getattr(estimator, self.method)(designTest)
        scored = pd.concat([targetTest.rename('truth'), yhat.rename('predicted')], axis=1)
        return scored

    def _evaluate(self, estimator, scored, metrics, X):
        measurements = {
            metric.name: metric(
                estimator,
                scored.truth,
                scored.predicted,
                X) for metric in metrics}
        return measurements

    def score_evaluate(self, estimator=None, X=None, y=None, metrics=None):
        """
        Score the fitted estimator on y and evaluate metrics

        Args:
            estimator (BaseEstimator): Fit estimator to evaluate
            X (pandas.DataFrame): design
            y (pandas.Series): target
            metrics (dict): metrics to evaluate

        Returns:
            tuple(pandas.Series, metric evaluations): scored, measurements
        """
        scored = self._score(estimator, X, y)
        measurements = self._evaluate(estimator, scored, metrics)
        return scored, measurements

    @package_dataset
    def fit_score_folds(self, estimator=None, X=None, y=None, dataset=None, **kwargs):
        """
        Cross validating measure, clones and fits estimator on each fold of X|y

        Args:
            estimator (BaseEstimator): Fit estimator to evaluate
            dataset (donatello.data.Dataset) object to cross val over
            X (pandas.DataFrame): design
            y (pandas.Series): target
            metrics (dict): metrics to evaluate

        Returns:
            tuple(pandas.Series, metric evaluations): scored, measurements
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
        Calculate metrics from cross val measurements
        """
        [metric.fit(scored) for metric in metrics]

        # move to list of dict -> concat
        outputs = defaultdict(pd.DataFrame)
        for fold, df in scored.groupby('fold'):
            _outputs = self._evaluate(estimators[fold], df, metrics, X)
            [_append_in_place(outputs, name, df) for name, df in _outputs.items()]

        measurements = {metric.name: metric.callback(_unwrap_multiple(outputs[metric.name]
                                                                      .groupby(metric.key)
                                                                      .agg(metric.agg),
                                                                      metric.sort))
                        for metric in metrics}

        return measurements

    @package_dataset
    def buildCV(self, estimator=None, metrics=None, X=None, y=None, dataset=None):
        """
        Build cross validated scoring report
        """
        estimators, scored = self.fit_score_folds(estimator=estimator, dataset=dataset)
        measurements = self.evaluate_scored_folds(
            estimators=estimators,
            scored=scored,
            X=dataset.designData,
            metrics=metrics)
        return {'estimators': estimators, 'scored': scored, 'measurements': measurements}

    def build_holdout(self, estimator=None, metrics=None, X=None, y=None):
        """
        score already fit estimator
        """
        scored = self._score(estimator, X, y)
        scored['fold'] = 0
        estimators = {0: estimator}
        measurements = self.evaluate_scored_folds(
            estimators=estimators, scored=scored, X=X, metrics=metrics)
        return {'estimators': estimators, 'scored': scored, 'measurements': measurements}


class Metric(Dobject):
    @init_time
    @name
    @coelesce(columns=['score'])
    def __init__(self, scorer=None, columns=None, name='', key=None, scoreClay=None,
                 callback=pass_through, agg=['mean', 'std'], sort=None):
        self.columns = columns
        self.scorer = scorer
        _name = getattr(scorer, '__name__', self.__class__.__name__)
        self._name = name if name else _name
        self.scoreClay = scoreClay
        self.callback = callback
        self.agg = agg
        if key:
            self.key = key
        self.sort = sort

    @property
    def key(self):
        return getattr(self, '_key', ['_'])

    @key.setter
    def key(self, value):
        self._key = value

    def fit(self, scored):
        return self

    def evaluate(self, estimator, truth, predicted, X):
        df = pd.DataFrame([self.scorer(truth, predicted)])
        if not hasattr(self, '_key'):
            df['_'] = range(len(df))
        return df

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)


class FeatureWeights(Metric):
    def __init__(self, attr='', name='feature_weights', key='names',
                 callback=pass_through, agg=['mean', 'std'], sort=None):

        super(FeatureWeights, self).__init__(name=name, key=key,
                                             callback=callback, agg=agg, sort=sort)

    def evaluate(self, estimator, truth, predicted, X):
        """
        Args:
            estimator (donatello.estimator.Estimator): has `features` and `model` attributes
            attr (str): option to specify additional attribute to pull

        Returns:
            pandas.DataFrame: values of feature weights
        """

        names = estimator.features if estimator.features else X.columns.tolist()

        model = estimator.model
        columnNames = ['names']
        values = []

        if hasattr(model, 'coef_'):
            coef_ = model.coef_[0] if (len(model.coef_.shape) ==
                                       2 and model.coef_.shape[0] == 1) else model.coef_
        else:
            coef_ = None

        intercept_ = getattr(model, 'intercept_', None)
        feature_importances_ = getattr(model, 'feature_importances_', None)

        if coef_ is not None:
            columnNames.append('coefficients')
            if intercept_ is not None:
                names.append('intercept_')
                values.append(np.hstack((coef_, intercept_)))
            else:
                values.append(coef_)

        if feature_importances_ is not None:
            columnNames.append('feature_importances')
            values.append(feature_importances_)

        if values:
            names = pd.Series(np.asarray(names), name=columnNames[0])
            vectors = pd.DataFrame(np.asarray(values).T, columns=columnNames[1:])

            data = pd.concat([names, vectors], axis=1)
            return data


class ThresholdRates(Metric):
    def __init__(self, key='points', sort='points'):
        super(ThresholdRates, self).__init__(key=key, sort=sort)

    def fit(self, scored, points=None, spacing=101, **kwargs):
        if points is None:
            percentiles = np.linspace(0, 1, spacing)
            self.points = scored.predicted.quantile(percentiles)
        else:
            self.points = points

    def evaluate(self, estimator, truth, predicted, X):
        """
        """
        data = [confusion_matrix(truth, predicted > i).reshape(4,) for i in self.points]

        df = pd.DataFrame(data=data,
                          columns=['true_negative', 'false_positive',
                                   'false_negative', 'true_positive'],
                          index=pd.Series(self.points, name='points')
                          )

        df = df.apply(lambda x: x / np.sum(x), axis=1).reset_index()

        df['precision'] = df.true_positive / (df.true_positive + df.false_positive)
        df['recall'] = df.true_positive / (df.true_positive + df.false_negative)
        df['specificity'] = df.true_negative / (df.true_negative + df.false_positive)
        df['false_omission_rate'] = df.false_negative / (df.false_negative + df.true_negative)
        df['negative_predictive_value'] = df.true_negative / (df.true_negative + df.false_negative)
        df['f1'] = 2 * df.true_positive / \
            (2 * df.true_positive + df.false_positive + df.false_negative)

        df['fall_out'] = 1 - df.specificity
        df['false_discovery_rate'] = 1 - df.precision

        return df
