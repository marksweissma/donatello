"""
Quickstart introduction
"""
import pandas as pd
import random

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

from donatello.components.data import Dataset
from donatello.components.estimator import Estimator
from donatello.components.measure import Metric, FeatureWeights, ThresholdRates
from donatello.components.core import Sculpture


def load_sklearn_bc_dataset(group=False):
    """
    Helper to load sklearn dataset into a pandas dataframe

    Returns:
        pd.DataFrame: X and y combined
    """
    dataset = load_breast_cancer()
    df = pd.DataFrame(data=pd.np.c_[dataset['data'], dataset['target']],
                      columns=(dataset['feature_names'].tolist() + ['is_malignant'])
                      )
    if group:
        df['groups_column'] = df.apply(lambda x: random.choice(['a', 'b', 'c', 'd']), axis=1)
    return df


def load_data(asDf, group):
    # loading dataframe directly vs specifying queries
    if asDf:
        data = {'raw': load_sklearn_bc_dataset(group)}
    else:
        data = {'queries': {None: {'querier': load_sklearn_bc_dataset, 'group': group}}}

    data['target'] = 'is_malignant'

    # Declare intent for partitioning data over groups (rather than all rows being independent)
    if group:
        data['clay'] = 'group'
        data['groupDap'] = {'attrPath': ['groups_column'], 'slicers': (pd.DataFrame, dict)}

    dataset = Dataset(**data)
    return dataset


def load_sculpture(asDf, group):
    """
    Helper to load sculpture
    """
    dataset = Dataset(raw=load_sklearn_bc_dataset(), target='is_malignant')

    estimator = Estimator(model=LogisticRegression(),
                          paramGrid={'model__C': list(pd.np.logspace(-2, 0, 5))},
                          searchKwargs={'scoring': 'roc_auc', 'cv': 3},
                          method='predict_proba',
                          scorer='score_second'
                          )

    metrics = [Metric(roc_auc_score), Metric(average_precision_score),
               FeatureWeights(sort='coefficients'), ThresholdRates()]

    sculpture = Sculpture(dataset=dataset, estimator=estimator, metrics=metrics)

    return sculpture


def load_metrics(metrics=None, featureName='coefficients'):

    if not metrics:
        metrics = [Metric(roc_auc_score), Metric(average_precision_score),
                   FeatureWeights(sort=featureName), ThresholdRates()]

    return metrics
