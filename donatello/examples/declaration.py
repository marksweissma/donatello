import random
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold

from donatello.components.manager import DM
from donatello.components.estimator import Estimator
from donatello.utils.transformers import Selector
from donatello.utils.helpers import reformat_aggs


def _load_sklearn_bc_dataset():
    dataset = load_breast_cancer()
    df = pd.DataFrame(data=pd.np.c_[dataset['data'], dataset['target']],
                      columns=(dataset['feature_names'].tolist() +
                               ['is_malignant'])
                      )
    df['grouper'] = df.apply(lambda x: random.choice(range(10)), axis=1)
    return df


def load_declaration(asDf=False):
    data = {'raws': _load_sklearn_bc_dataset()} if asDf else {'queries': {None: {'querier': _load_sklearn_bc_dataset}}}
    split = {'target': 'is_malignant'}
    estimator = Estimator(model=LogisticRegression(),
                          transformer=Selector(['grouper'], reverse=True),
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


def load_group_declaration(asDf=True):

    data = {'raws': _load_sklearn_bc_dataset()} if asDf else {'queries': {None: {'querier': _load_sklearn_bc_dataset}}}

    typeDispatch = {'splitter': {'classification': GroupKFold}}

    # changing defaults
    data.update({'typeDispatch': typeDispatch, 'groupKey': 'grouper',
                 'splitKwargs': {}
                 })

    split = {'target': 'is_malignant', 'splitOver': 'grouper'}
    estimator = Estimator(model=LogisticRegression(),
                          transformer=Selector(['grouper'], reverse=True),
                          # paramGrid={'model__C': list(pd.np.logspace(-2, 0, 10))},
                          # gridKwargs={'scoring': 'f1', 'cv': 5},
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


def load_dm(group=True, asDf=False):
    """
    Helper function to load donatello manager  for sklearn
    breast cancer data set to classify malignancy
    """
    func = load_group_declaration if group else load_declaration
    declaration = func(asDf)
    m = DM(**declaration)
    return m
