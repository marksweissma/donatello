import random
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score

from donatello.components.core import Sculpture
from donatello.components.measure import Metric, FeatureWeights, ThresholdRates
from donatello.components.data import Dataset
from donatello.components.estimator import Estimator, score_column


def _load_sklearn_bc_dataset(group=True):
    dataset = load_breast_cancer()
    df = pd.DataFrame(data=pd.np.c_[dataset['data'], dataset['target']],
                      columns=(dataset['feature_names'].tolist() +
                               ['is_malignant'])
                      )
    if group:
        df['a_column'] = df.apply(lambda x: random.choice(range(10)), axis=1)
    return df


def load_data_fold(asDf, group):
    data = {'raws': _load_sklearn_bc_dataset(group)} if asDf else {'queries': {None: {'querier':
                                                                                      _load_sklearn_bc_dataset,
                                                                                      'group': group}}}
    data['target'] = 'is_malignant'
    if group:
        data['foldClay'] = 'group'
        data['dap'] = {'groups': {'attrPath': ['a_column'], 'slicers': (pd.DataFrame, dict)}}

    return data


def load_metrics(metrics=None, featureName='coefficients'):

    metrics = [Metric(roc_auc_score), Metric(average_precision_score),
               FeatureWeights(sort=featureName), ThresholdRates()]

    return metrics


def load_logit_declaration(group=True, asDf=False, metrics=None):

    data = load_data_fold(asDf, group)

    estimator = {'model': LogisticRegression(),
                 'paramGrid': {'model__C': list(pd.np.logspace(-2, 0, 10))},
                 'gridKwargs': {'scoring': 'roc_auc', 'cv': 5},
                 'method': 'predict_proba',
                 'scorer': score_column()

                 }

    metrics = load_metrics(metrics)
    declaration = {'dataset': Dataset(**data),
                   'estimator': Estimator(**estimator),
                   'metrics': metrics,
                   'validation': True,
                   'holdOut': True
                   }

    return declaration


def load_random_forest_declaration(group=True, asDf=True, metrics=None):
    data, fold = load_data_fold(asDf, group)

    estimator = {'model': RandomForestClassifier(n_estimators=10),
                 'paramGrid': {'model__max_depth': [3, 5, 7]},
                 'gridKwargs': {'scoring': 'f1', 'cv': 5},
                 'scoreClay': 'classification'
                 }

    metrics = load_metrics(metrics, 'feature_importances')

    declaration = {'dataDeclaration': data,
                   'folderDeclaration': fold,
                   'estimatorDeclaration': estimator,
                   'metrics': metrics,
                   'foldClay': 'classification',
                   'scoreClay': 'classification',
                   'validation': True,
                   'holdOut': False
                   }

    return declaration


def load_isolation_forest_declaration(group=True, asDf=False, metrics=['roc_auc_score', 'average_percision_score', 'threshold_rates']):
    data = load_data_fold(asDf, group)

    metrics = load_metrics(metrics)

    estimator = {'model': IsolationForest(), 'scoreClay': 'anomaly'}

    declaration = {'data': Dataset(**data),
                   'estimator': Estimator(**estimator),
                   'metrics': metrics,
                   'scoreClay': 'anomaly',
                   'foldClay': 'stratify',
                   'validation': True,
                   'holdOut': True
                   }

    return declaration


def load_dm(model='lr', group=True, asDf=False):
    """
    Helper function to load donatello manager for sklearn
    breast cancer data set to classify malignancy
    """
    if model == 'if':
        declaration = load_isolation_forest_declaration
    elif model == 'rf':
        declaration = load_random_forest_declaration
    elif model == 'logit':
        declaration = load_logit_declaration

    declaration = declaration(group=group, asDf=asDf)
    m = Sculpture(**declaration)
    return m
