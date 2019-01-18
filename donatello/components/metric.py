import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix

from donatello.utils.base import Dobject
from donatello.utils.decorators import (init_time,
                                        coelesce,
                                        name
                                        )


def pass_through(x):
    return x


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

        self.attr = attr

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
        if hasattr(model, self.attr):
            columnNames.append(self.attr)
            values.append(getattr(model, self.attr))
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


class ThresholdRates(Metric):
    def __init__(self, key='thresholds', sort='thresholds'):
        super(ThresholdRates, self).__init__(key=key, sort=sort)

    def fit(self, scored, thresholds=None, spacing=101, **kwargs):
        if thresholds is None:
            percentiles = np.linspace(0, 1, spacing)
            self.thresholds = scored.predicted.quantile(percentiles)
        else:
            self.thresholds = thresholds

    def evaluate(self, estimator, truth, predicted, X):
        """
        """
        data = [confusion_matrix(truth.values, (predicted > i).values).reshape(4,)
                for i in self.thresholds]

        df = pd.DataFrame(data=data,
                          columns=['true_negative', 'false_positive',
                                   'false_negative', 'true_positive'],
                          index=pd.Series(self.thresholds, name='thresholds')
                          )

        df = df.apply(lambda x: x / np.sum(x), axis=1).reset_index()

        df['false_omission_rate'] = df.false_negative / (df.false_negative + df.true_negative)
        df['f1'] = 2 * df.true_positive / (2 * df.true_positive + df.false_positive + df.false_negative)
        df['recall'] = df.true_positive / (df.true_positive + df.false_negative)
        df['specificity'] = df.true_negative / (df.true_negative + df.false_positive)
        df['precision'] = df.true_positive / (df.true_positive + df.false_positive)
        df['negative_predictive_value'] = df.true_negative / (df.true_negative + df.false_negative)

        df['fall_out'] = 1 - df.specificity
        df['false_discovery_rate'] = 1 - df.precision

        return df
