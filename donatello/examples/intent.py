import random
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score

from donatello.components.core import Sculpture
from donatello.components.measure import Metric, FeatureWeights, ThresholdRates
from donatello.components.data import Dataset
from donatello.components.estimator import Estimator
from donatello.components import transformers


def load_model(model=LogisticRegression(C=5)):

    t = transformers.DatasetFlow(invert=True, passTarget=True)

    s = transformers.StandardScaler()

    n1 = transformers.Node('n1', transformer=t)
    n2 = transformers.Node('n2', transformer=s)
    n3 = transformers.Node('n3', transformer=model)

    g = transformers.ModelDAG(set([]), {}, graphKwargs={'name': 'model_sklearn_breast_cancer'})

    g.add_edge_flow(n1, n2)
    g.add_edge_flow(n1, n3, passDesign=False)
    g.add_edge_flow(n2, n3)

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
        data = {'raw': load_sklearn_bc_dataset(group)}
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
    estimator = {'model': load_model(),
                 'paramGrid': {'model__n3__C': list(pd.np.logspace(-2, 0, 5))},
                 'searchKwargs': {'scoring': 'roc_auc', 'cv': 3},
                 'method': 'predict_proba',
                 'scorer': 'score_second'

                 }
    return estimator


def load_random_forest():

    estimator = {'model': load_model(RandomForestClassifier(n_estimators=40)),
                 'paramGrid': {'model__n3__max_depth': range(2, 5)},
                 'searchKwargs': {'scoring': 'roc_auc', 'cv': 3},
                 'method': 'predict_proba',
                 'scorer': 'score_second'
                 }
    return estimator


def load_isolation_forest(group=True, asDf=False):

    estimator = {'model': load_model(IsolationForest()),
                 'method': 'predict_proba',
                 'scorer': 'score_second'
                 }

    return estimator


def load_declaration(load_estimator, group=True, asDf=False,
                     metrics=None, featureName='coefficients'):
    data = load_data(asDf, group)
    estimator = load_estimator()

    metrics = load_metrics(metrics, featureName)
    # metrics = []
    declaration = {'dataset': Dataset(**data),
                   'estimator': Estimator(**estimator),
                   'metrics': metrics,
                   'validation': True,
                   'holdout': True,
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
        declaration = load_declaration(
            load_random_forest,
            group,
            asDf,
            featureName='feature_importances')
    elif model == 'if':
        metrics = ['roc_auc_score', 'average_percision_score', 'threshold_rates']
        declaration = load_declaration(load_isolation_forest, metrics=metrics)

    m = Sculpture(**declaration)
    return m
