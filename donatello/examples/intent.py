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
from donatello.components import transformers


def load_model(model=LogisticRegression(C=5)):

    t = transformers.DatasetConductor(reverse=True, passTarget=True)

    s = transformers.StandardScaler()

    n1 = transformers.TransformNode('n1', transformer=t)
    n2 = transformers.TransformNode('n2', transformer=s)
    n3 = transformers.TransformNode('n3', transformer=model)

    g = transformers.ModelDAG(graphKwargs={'name': 'sklearn_breast_cancer'})

    g.add_edge_conductor(n1, n2)
    g.add_edge_conductor(n1, n3, passDesign=False)
    g.add_edge_conductor(n2, n3)

    return g


def load_sklearn_bc_dataset(group=True):
    dataset = load_breast_cancer()
    df = pd.DataFrame(data=pd.np.c_[dataset['data'], dataset['target']],
                      columns=(dataset['feature_names'].tolist() +
                               ['is_malignant'])
                      )
    if group:
        df['a_column'] = df.apply(lambda x: random.choice(range(10)), axis=1)
    return df


def load_data(asDf, group):
    if asDf:
        data = {'raws': load_sklearn_bc_dataset(group)}
    else:
        data = {'queries': {None: {'querier': load_sklearn_bc_dataset, 'group': group}}}

    data['target'] = 'is_malignant'

    if group:
        data['foldClay'] = 'group'
        data['dap'] = {'groups': {'attrPath': ['a_column'], 'slicers': (pd.DataFrame, dict)}}

    return data


def load_metrics(metrics=None, featureName='coefficients'):

    if not metrics:
        metrics = [Metric(roc_auc_score), Metric(average_precision_score),
                   FeatureWeights(sort=featureName), ThresholdRates()]

    return metrics


def load_logit():
    estimator = {'model':  load_model(),
                 'paramGrid': {'model__n3__C': list(pd.np.logspace(-2, 0, 5))},
                 'gridKwargs': {'scoring': 'roc_auc', 'cv': 5},
                 'method': 'predict_proba',
                 'scorer': score_column()

                 }
    return estimator


def load_random_forest():

    estimator = {'model': load_model(RandomForestClassifier(n_estimators=100)),
                 'paramGrid': {'model__n3__max_depth': range(1, 4)},
                 'gridKwargs': {'scoring': 'f1', 'cv': 5},
                 'method': 'predict_proba',
                 'scorer': score_column()
                 }
    return estimator


def load_isolation_forest(group=True, asDf=False):

    estimator = {'model': load_model(IsolationForest()),
                 'method': 'predict_proba',
                 'scorer': score_column()
                 }

    return estimator


def load_declaration(load_estimator, group=True, asDf=False, metrics=None, featureName='coefficients'):
    data = load_data(asDf, group)
    estimator = load_estimator()

    metrics = load_metrics(metrics, featureName)
    # metrics = []
    declaration = {'dataset': Dataset(**data),
                   'estimator': Estimator(**estimator),
                   'metrics': metrics,
                   'validation': True,
                   'holdOut': True,
                   # 'entire': True
                   }

    return declaration


def load_dm(model='logit', group=True, asDf=False):
    """
    Helper function to load donatello manager for sklearn
    breast cancer data set to classify malignancy
    """
    if model == 'logit':
        declaration = load_declaration(load_logit, group, asDf)
    elif model == 'rf':
        declaration = load_declaration(load_random_forest, group, asDf, featureName='feature_importances')
    elif model == 'if':
        metrics = ['roc_auc_score', 'average_percision_score', 'threshold_rates']
        declaration = load_declaration(load_isolation_forest, metrics=metrics)

    m = Sculpture(**declaration)
    return m
