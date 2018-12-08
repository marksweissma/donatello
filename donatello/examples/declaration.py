import random
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold

from donatello.components.manager import DM
from donatello.components.estimator import Estimator
from donatello.utils.transformers import Selector
from donatello.utils.helpers import reformat_aggs
from donatello.components.scorer import Scorer


def _load_sklearn_bc_dataset(group=True):
    dataset = load_breast_cancer()
    df = pd.DataFrame(data=pd.np.c_[dataset['data'], dataset['target']],
                      columns=(dataset['feature_names'].tolist() +
                               ['is_malignant'])
                      )
    if group:
        df['grouper'] = df.apply(lambda x: random.choice(range(10)), axis=1)
    return df


def load_data_split(asDf, group):
    data = {'raws': _load_sklearn_bc_dataset(group)} if asDf else {'queries': {None: {'querier': _load_sklearn_bc_dataset, 'group': group}}}
    split = {'target': 'is_malignant'}
    if group:
        typeDispatch = {'splitter': {'classification': GroupKFold}}
        data.update({'typeDispatch': typeDispatch, 'groupKey': 'grouper',
                     'splitKwargs': {}
                     })

        split.update({'mlType': 'group',
                      'runTimeAccess': {'group': ['grouper', 'values']}
                      })

    return data, split

def load_metrics(metrics=None, featureName='coefficients'):
    _metrics = {roc_auc_score: {},
                average_precision_score: {},
                'feature_weights': {'key': 'names',
                                    'sort': featureName,
                                    'callback': reformat_aggs,
                                    'agg': 'describe',
                                    'callbackKwargs': {'sortValues': 'mean',
                                                       'indexName': 'features'
                                                       }
                                    },
                'threshold_rates': {'key': 'thresholds',
                                    'sort': 'thresholds',
                                    }
                }

    _filter = Scorer()
    metrics = {i: j for i, j in _metrics.items() if _filter.get_metric_name(i) in metrics} if metrics else _metrics
    return metrics

def load_logit_declaration(group=True, asDf=False, metrics=None):

    data, split = load_data_split(asDf, group)

    estimator = Estimator(model=LogisticRegression(),
                          transformer=Selector(['grouper'], reverse=True),
                          paramGrid={'model__C': list(pd.np.logspace(-2, 0, 10))},
                          gridKwargs={'scoring': 'f1', 'cv': 5},
                          mlType='classification'
                          )

    metrics = load_metrics(metrics)
    declaration = {'dataKwargs': data,
                   'splitterKwargs': split,
                   'estimator': estimator,
                   'metrics': metrics,
                   'mlType': 'classification',
                   'validation': True,
                   'holdOut': True
                   }

    return declaration


def load_random_forest_declaration(group=True, asDf=True, metrics=None):
    data, split = load_data_split(asDf, group)

    estimator = Estimator(model=RandomForestClassifier(n_estimators=100),
                          transformer=Selector(['grouper'], reverse=True),
                          paramGrid={'model__max_depth': [3, 5, 7]},
                          gridKwargs={'scoring': 'f1', 'cv': 5},
                          mlType='classification'
                          )

    metrics = load_metrics(metrics, 'feature_importances')

    declaration = {'dataKwargs': data,
                   'splitterKwargs': split,
                   'estimator': estimator,
                   'metrics': metrics,
                   'mlType': 'classification',
                   'validation': True,
                   'holdOut': True
                   }

    return declaration


def load_isolation_forest_declaration(group=True, asDf=False, metrics=['roc_auc_score', 'average_percision_score', 'threshold_rates']):
    data, split = load_data_split(asDf, group)

    estimator = Estimator(model=IsolationForest(),
                          typeDispatch={'classification': {'method': 'decision_function', 'score': 'score_all'}},
                          gridKwargs={'scoring': 'f1', 'cv': 5},
                          mlType='classification'
                          )

    metrics = load_metrics(metrics)
    declaration = {'dataKwargs': data,
                   'splitterKwargs': split,
                   'estimator': estimator,
                   'metrics': metrics,
                   'mlType': 'classification',
                   'validation': True,
                   'holdOut': True
                   }

    return declaration


def load_dm(model='lr', group=True, asDf=False):
    """
    Helper function to load donatello manager  for sklearn
    breast cancer data set to classify malignancy
    """
    if model == 'if':
        declaration = load_isolation_forest_declaration
    elif model == 'rf':
        declaration = load_random_forest_declaration
    elif model == 'logit':
        declaration = load_logit_declaration

    declaration = declaration(group=group, asDf=asDf)
    m = DM(**declaration)
    return m
