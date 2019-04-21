"""
Quickstart introduction
"""
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

from donatello.components.data import Dataset
from donatello.components.estimator import Estimator
from donatello.components.measure import Metric, FeatureWeights, ThresholdRates
from donatello.components.core import Sculpture


def load_sklearn_bc_dataset():
    """
    Helper to load sklearn dataset into a pandas dataframe

    Returns:
        tuple(pd.DataFrame, pd.Series): X and y
    """
    dataset = load_breast_cancer()
    X = pd.DataFrame(data=pd.np.c_[dataset['data'], ],
                     columns=(dataset['feature_names'].tolist())
                     )
    y = pd.Series(dataset['target'], name='is_malignant')
    return X, y


def load_sculpture():
    """
    Helper to load sculpture
    """
    X, y = load_sklearn_bc_dataset()
    dataset = Dataset(X=X, y=y)

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
