from collections import defaultdict
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score

from donatello.components.manager import DM
from donatello.components.estimator import Estimator
from donatello.utils.helpers import reformat_aggs


def _load_sklearn_bc_dataset():
    dataset = load_breast_cancer()
    df = pd.DataFrame(data=pd.np.c_[dataset['data'], dataset['target']],
                      columns=(dataset['feature_names'].tolist() +
                               ['is_malignant'])
                      )
    return df

def load_declaration(asDf=False):
    data = {'raws': _load_sklearn_bc_dataset()} if asDf else {'queries': {None: {'querier': _load_sklearn_bc_dataset}}}
    split = {'target': 'is_malignant'}
    estimator = Estimator(model=LogisticRegression(),
                          paramGrid={'model__C': list(pd.np.logspace(-2, 0, 10))},
                          gridKwargs={'scoring': 'f1', 'cv': 5},
                          mlType='classification'
                          )

    metrics = {roc_auc_score: {},
               average_precision_score: {},
               'feature_weights': {'key': 'names',
                                   'sort': 'coefficients',
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

    declaration = {'dataKwargs': data,
                   'splitterKwargs': split,
                   'estimator': estimator,
                   'metrics': metrics,
                   'mlType': 'classification',
                   'validation': True,
                   'holdOut': True
                   }

    return declaration


def load_declaration(asDf=False):
    data = {'raws': _load_sklearn_bc_dataset()} if asDf else {'queries': {None: {'querier': _load_sklearn_bc_dataset}}}
    split = {'target': 'is_malignant'}
    estimator = Estimator(model=IsolationForest(),
                          typeDispatch={'classification': {'method': 'decision_function', 'score': 'score_all'}},
                          gridKwargs={'scoring': 'f1', 'cv': 5},
                          mlType='classification'
                          )

    metrics = {roc_auc_score: {},
               average_precision_score: {},
               # 'feature_weights': {'key': 'names',
                                   # 'sort': 'coefficients',
                                   # 'callback': reformat_aggs,
                                   # 'agg': 'describe',
                                   # 'callbackKwargs': {'sortValues': 'mean',
                                                      # 'indexName': 'features'
                                                      # }
                                   # },
               'threshold_rates': {'key': 'thresholds',
                                   'sort': 'thresholds',
                                   }
               }

    declaration = {'dataKwargs': data,
                   'splitterKwargs': split,
                   'estimator': estimator,
                   'metrics': metrics,
                   'mlType': 'classification',
                   'validation': True,
                   'holdOut': True
                   }

    return declaration


def load_dm(asDf=False):
    """
    Helper function to load donatello manager  for sklearn
    breast cancer data set to classify malignancy
    """
    declaration=load_declaration(asDf)
    m = DM(**declaration)
    return m
