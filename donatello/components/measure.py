import pandas as pd
import numpy as np

from inflection import underscore

from sklearn import clone
from sklearn.utils import Bunch
from sklearn.metrics import confusion_matrix

from donatello.utils.helpers import view_sk_metric
from donatello.utils.decorators import init_time, coelesce
from donatello.utils.base import Dobject
from donatello.components.data import package_dataset


# FIX THIS
# move to access
def _append_in_place(store, name, value):
    store[name] = store[name].append(value)


# something better here
def _option_sort(df, sort):
    df = df.sort_values(sort) if (sort and sort in df) else df
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
        # scored = pd.DataFrame()
        scored = []
        estimators = {}

        for fold, (designTrain, designTest, targetTrain, targetTest) in enumerate(dataset):
            estimator = clone(estimator)
            _dataset = dataset.with_params(X=designTrain, y=targetTrain)
            estimator.fit(dataset=_dataset, gridSearch=self.gridSearchFlag)
            estimators[fold] = estimator

            _temp = self._score(estimator, designTest, targetTest)
            _temp['fold'] = fold
            scored.append(_temp)
        scored = pd.concat(scored)
        return estimators, scored

    def evaluate_scored_folds(self, estimators=None, metrics=None, scored=None, X=None, **kwargs):
        """
        Calculate metrics from cross val measurements
        """
        [metric.fit(scored) for metric in metrics]

        outputs = {metric.name: [] for metric in metrics}
        for fold, df in scored.groupby('fold'):
            _outputs = self._evaluate(estimators[fold], df, metrics, X)
            [outputs[name].append(df) for name, df in _outputs.items()]

        measurements = {metric.name: metric.callback(_unwrap_multiple(pd.concat(outputs[metric.name])
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
    @coelesce(columns=['score'], agg=['mean', 'std'], evalKwargs={})
    def __init__(self, scorer=None, columns=None, name='', key=None,
                 callback=None, agg=None, sort=None,
                 withX=False, view=None,
                 evalArgs=(), evalKwargs=None):
        self.columns = columns
        self.scorer = scorer
        _name = getattr(scorer, '__name__', underscore(self.__class__.__name__))
        self._name = name if name else _name
        self.callback = callback
        self.agg = agg
        # hack: fix this
        if key:
            self.key = key
        self.withX = withX
        self.view = view
        self.sort = sort
        self.evalArgs = evalArgs
        self.evalKwargs = evalKwargs

    @property
    def callback(self):
        func = self._callback if self._callback else lambda x: x
        return func

    @callback.setter
    def callback(self, func):
        self._callback = func

    @property
    def key(self):
        return getattr(self, '_key', ['_'])

    @key.setter
    def key(self, value):
        self._key = value

    def fit(self, scored):
        return self

    def evaluate(self, estimator, truth, predicted, X):
        """
        Evaluate scorer on data
        """
        # move to access
        if self.withX:
            scores = self.scorer(truth, predicted, X=X, *self.evalArgs, **self.evalKwargs)
        else:
            scores = self.scorer(truth, predicted, *self.evalArgs, **self.evalKwargs)
        df = pd.DataFrame([scores])
        if not hasattr(self, '_key'):
            df['_'] = range(len(df))
        return df

    def display(self, bunch, *args, **kwargs):
        """
        Display metric's data
        """
        if self.view:
            self.view(view_sk_metric(bunch))
        else:
            print('Metric does not have viewiing function, set self.view')

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)


class FeatureWeights(Metric):
    """
    Metric to collect model weights. can sort on column name via
    attribute name (with trailing `_` stripped)
    i.e. coefficients in linear model or feature_importances in tree model

    attr (str): option to specify additional attribute to pull
    """
    def __init__(self, attr='', name='feature_weights', key='names',
                 callback=None, agg=['mean', 'std'], sort=None):

        super(FeatureWeights, self).__init__(name=name, key=key,
                                             callback=callback, agg=agg, sort=sort)

    def evaluate(self, estimator, truth, predicted, X):
        """
        Args:
            estimator (donatello.estimator.Estimator): has `features` and `model` attributes

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
    """
    Confusion matrix parameterized by evaluation threshold. Samples
    points from score distribution
    """
    def __init__(self, key='points', sort='points'):
        super(ThresholdRates, self).__init__(key=key, sort=sort)

    def fit(self, scored, points=None, spacing=21, **kwargs):
        if points is None:
            percentiles = np.linspace(0, 1, spacing)
            self.points = scored.predicted.quantile(percentiles)
        else:
            self.points = points

    def evaluate(self, estimator, truth, predicted, X):
        """
        """
        # rewrite with dyadic product
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
