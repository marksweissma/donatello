from collections import defaultdict
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

from donatello.components.manager import DM
from donatello.components.estimator import Estimator


def _load_sklearn_bc_dataset():
    dataset = load_breast_cancer()
    df = pd.DataFrame(data=pd.np.c_[dataset['data'], dataset['target']],
                      columns=(dataset['feature_names'].tolist() +
                               ['is_malignant'])
                      )
    return df

def load_declaration():
    data = {'queries': {}, 'etl': _load_sklearn_bc_dataset}
    split = {'target': 'is_malignant'}
    estimator = Estimator(modelType=LogisticRegression,
                          paramGrid={'model__C': list(pd.np.logspace(-2, 0, 10))},
                          gridKwargs={'scoring': 'f1', 'cv': 5},
                          mlType='classification'
                          )

    metrics = {roc_auc_score: defaultdict(dict),
               average_precision_score: defaultdict(dict),
               'feature_weights': defaultdict(dict, {'key': 'names',
                                                     'sort': 'coefficients',
                                                     }),
               'threshold_rates': defaultdict(dict, {'key': 'thresholds',
                                                     'sort': 'thresholds',
                                                     })
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


def load_sklearn_bc_declaration(declaration=load_declaration()):
    """
    Helper function to load declaration for sklearn
    breast cancer data set to classify malignancy
    """
    m = DM(**declaration)
    return m
