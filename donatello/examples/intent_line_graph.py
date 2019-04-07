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
from donatello.components import transformers


def load_sklearn_bc_dataset():
    """
    Helper to load sklearn dataset into a pandas dataframe

    Returns:
        pd.DataFrame: X and y combined
    """
    dataset = load_breast_cancer()
    df = pd.DataFrame(data=pd.np.c_[dataset['data'], dataset['target']],
                      columns=(dataset['feature_names'].tolist() + ['is_malignant'])
                      )
    return df


def load_model(model=LogisticRegression(C=5)):
    """
    Helper to construct model execution graph

    Args:
        model (sklearn.base.BaseEstimator): ml prediction object

    Returns:
        donatello.components.transformer.ModelDAG: execution graph
    """

    graph = transformers.ModelDAG(set([]), {}, graphKwargs={'name': 'model_sklearn_breast_cancer'})

    n1 = transformers.TransformNode('scale', transformer=transformers.StandardScaler(), enforceTarget=True)
    n2 = transformers.TransformNode('ml', transformer=model)

    graph.add_edge_conductor(n1, n2)  # defaults to passing desing AND target

    return graph


def load_scuplture():
    """
    Helper to load sculpture
    """
    dataset = Dataset(raw=load_sklearn_bc_dataset(), target='is_malignant')

    model = load_model()

    estimator = Estimator(model=model,
                          paramGrid={'model__ml__C': list(pd.np.logspace(-2, 0, 5))},
                          searchKwargs={'scoring': 'roc_auc', 'cv': 3},
                          method='predict_proba',
                          scorer='score_second'
                          )

    metrics = [Metric(roc_auc_score), Metric(average_precision_score),
               FeatureWeights(sort='coefficients'), ThresholdRates()]

    scuplture = Sculpture(dataset=dataset, estimator=estimator, metrics=metrics)

    return scuplture
