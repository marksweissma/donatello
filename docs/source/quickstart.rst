===========
Quick Start
===========

Sculpture
=========

This quick start is built of this notebook (notebooks/quick_start.rst)

To get started, connect a dataset via
:py:class:`donatello.components.data.Dataset` and
:py:class:`donatello.components.estimator.Estimator` to
:py:class:`donatello.components.core.Sculpture`


Intent
------

.. code:: python


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
            pd.DataFrame: X and y combined
        """
        dataset = load_breast_cancer()
        df = pd.DataFrame(data=pd.np.c_[dataset['data'], dataset['target']],
                          columns=(dataset['feature_names'].tolist() + ['is_malignant'])
                          )
        return df


    def load_scuplture():
        """
        Helper to load sculpture
        """
        dataset = Dataset(raw=load_sklearn_bc_dataset(), target='is_malignant')

        estimator = Estimator(model=LogisticRegression(),
                              paramGrid={'model__C': list(pd.np.logspace(-2, 0, 5))},
                              gridKwargs={'scoring': 'roc_auc', 'cv': 3},
                              method='predict_proba',
                              scorer='score_second'
                              )

        metrics = [Metric(roc_auc_score), Metric(average_precision_score),
                   FeatureWeights(sort='coefficients'), ThresholdRates()]

        scuplture = Sculpture(dataset=dataset, estimator=estimator, metrics=metrics)

        return scuplture


    sculpture = load_scuplture()


Declaration
-----------

Sculptures are declaratively defined modeling objects enabling
repeatable and traceable experimentation. Donetallo's framework
attempts to follow scikit-learn's pattern with an eye toward
improving utilization with pandas.

.. code:: python

    sculpture.declaration


.. parsed-literal::

    {'dataset': Dataset_2019_03_24_13_02,
     'entire': False,
     'estimator': Estimator_2019_03_24_13_02,
     'holdOut': True,
     'measure': Measure_2019_03_24_13_02,
     'metrics': [roc_auc_score_2019_03_24_13_02,
                 average_precision_score_2019_03_24_13_02,
                 feature_weights_2019_03_24_13_02,
                 ThresholdRates_2019_03_24_13_02],
     'persist': <function donatello.utils.helpers.persist>,
     'storeReferences': True,
     'timeFormat': '%Y_%m_%d_%H_%M',
     'validation': True,
     'writeAttrs': ('', 'estimator')}



The ``validation``, ``holdOut``, and ``entire`` flags dictate
over which (subsets) of data estimators are fit and metrics are calculated


.. parsed-literal::

    'validation': True,
    'holdOut': True,
    'entire': False


The metrics list is a collection of :py:class:`donatello.components.measure.Metric` objects
which fit calculate statistics around model performance, which can either wrap a 
scikit-learn metric or execute custom scoring functionality. If information needs
to be shared across folds for computation, it can be stored during the ``fit`` method.


.. code:: python

     'metrics': [roc_auc_score_2019_03_24_13_02,
                 average_precision_score_2019_03_24_13_02,
                 feature_weights_2019_03_24_13_02,
                 ThresholdRates_2019_03_24_13_02]



Fitting
-------

The sculputre's fit method defaults to instructions provided during instantiation.

``sculpture.fit() == sculpture.fit(dataset=sculpture.dataset)``
                    
.. code:: python

    sculpture.fit()


.. parsed-literal::

    Building Over Cross Validation
    grid searching
    grid searching
    grid searching
    grid searching
    grid searching

    Building Over Cross Validation
    grid searching

    Sculpture_2019_03_24_13_02



Evaluating
----------

During the fitting process, metrics are calculated over the
specified samples of data and stored in a :py:class:`sklearn.utils.Bunch`
(a lighlty wrapped dict, with attribute style accessors)
and stored in the measurements attribute

.. code:: python

    sculpture.measurements.keys()

    ['crossValidation', 'holdOut']



.. code:: python

    sculpture.measurements.crossValidation.keys()

    ['ThresholdRates',
     'roc_auc_score',
     'feature_weights',
     'average_precision_score']




The default aggregations are ``mean`` and ``std`` which are collected into a Bunch as well.

.. code:: python

    sculpture.measurements.crossValidation.average_precision_score

    {'mean':           0
     _          
     0  0.994795, 'std':           0
     _          
     0  0.003311}



.. code:: python

    sculpture.measurements.crossValidation.feature_weights.mean


.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>coefficients</th>
        </tr>
        <tr>
          <th>names</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>worst concavity</th>
          <td>-1.266728</td>
        </tr>
        <tr>
          <th>worst compactness</th>
          <td>-0.841691</td>
        </tr>
        <tr>
          <th>mean concavity</th>
          <td>-0.465869</td>
        </tr>
        <tr>
          <th>worst concave points</th>
          <td>-0.459076</td>
        </tr>
        <tr>
          <th>worst texture</th>
          <td>-0.372945</td>
        </tr>
        <tr>
          <th>worst symmetry</th>
          <td>-0.287106</td>
        </tr>
        <tr>
          <th>mean compactness</th>
          <td>-0.275708</td>
        </tr>
        <tr>
          <th>mean concave points</th>
          <td>-0.226367</td>
        </tr>
        <tr>
          <th>worst smoothness</th>
          <td>-0.201951</td>
        </tr>
        <tr>
          <th>worst perimeter</th>
          <td>-0.176177</td>
        </tr>
        <tr>
          <th>mean symmetry</th>
          <td>-0.108250</td>
        </tr>
        <tr>
          <th>area error</th>
          <td>-0.096981</td>
        </tr>
        <tr>
          <th>mean smoothness</th>
          <td>-0.094524</td>
        </tr>
        <tr>
          <th>worst fractal dimension</th>
          <td>-0.081174</td>
        </tr>
        <tr>
          <th>concavity error</th>
          <td>-0.070118</td>
        </tr>
        <tr>
          <th>concave points error</th>
          <td>-0.029708</td>
        </tr>
        <tr>
          <th>worst area</th>
          <td>-0.021795</td>
        </tr>
        <tr>
          <th>smoothness error</th>
          <td>-0.013807</td>
        </tr>
        <tr>
          <th>compactness error</th>
          <td>-0.011120</td>
        </tr>
        <tr>
          <th>mean fractal dimension</th>
          <td>-0.009681</td>
        </tr>
        <tr>
          <th>mean area</th>
          <td>-0.009288</td>
        </tr>
        <tr>
          <th>symmetry error</th>
          <td>-0.003870</td>
        </tr>
        <tr>
          <th>fractal dimension error</th>
          <td>0.001776</td>
        </tr>
        <tr>
          <th>radius error</th>
          <td>0.044685</td>
        </tr>
        <tr>
          <th>mean perimeter</th>
          <td>0.078273</td>
        </tr>
        <tr>
          <th>mean texture</th>
          <td>0.138544</td>
        </tr>
        <tr>
          <th>perimeter error</th>
          <td>0.269350</td>
        </tr>
        <tr>
          <th>intercept_</th>
          <td>0.326572</td>
        </tr>
        <tr>
          <th>texture error</th>
          <td>1.225060</td>
        </tr>
        <tr>
          <th>worst radius</th>
          <td>1.362734</td>
        </tr>
        <tr>
          <th>mean radius</th>
          <td>1.558584</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    sculpture.measurements.crossValidation.ThresholdRates.mean[['precision', 'recall']].loc[::5]


.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>precision</th>
          <th>recall</th>
        </tr>
        <tr>
          <th>points</th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>1.612093e-49</th>
          <td>0.629792</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>2.911910e-11</th>
          <td>0.661888</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>1.498740e-08</th>
          <td>0.699280</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>5.683305e-06</th>
          <td>0.741465</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>4.253043e-04</th>
          <td>0.786078</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>3.806747e-03</th>
          <td>0.839318</td>
          <td>1.000000</td>
        </tr>
        <tr>
          <th>9.748648e-02</th>
          <td>0.895883</td>
          <td>0.993331</td>
        </tr>
        <tr>
          <th>4.145121e-01</th>
          <td>0.942478</td>
          <td>0.971273</td>
        </tr>
        <tr>
          <th>8.199383e-01</th>
          <td>0.985902</td>
          <td>0.939640</td>
        </tr>
        <tr>
          <th>9.046890e-01</th>
          <td>0.992590</td>
          <td>0.864397</td>
        </tr>
        <tr>
          <th>9.464445e-01</th>
          <td>0.996154</td>
          <td>0.786247</td>
        </tr>
        <tr>
          <th>9.787446e-01</th>
          <td>0.996000</td>
          <td>0.709326</td>
        </tr>
        <tr>
          <th>9.851476e-01</th>
          <td>0.995652</td>
          <td>0.630941</td>
        </tr>
        <tr>
          <th>9.912757e-01</th>
          <td>0.994872</td>
          <td>0.551700</td>
        </tr>
        <tr>
          <th>9.944655e-01</th>
          <td>1.000000</td>
          <td>0.477573</td>
        </tr>
        <tr>
          <th>9.961186e-01</th>
          <td>1.000000</td>
          <td>0.397381</td>
        </tr>
        <tr>
          <th>9.981246e-01</th>
          <td>1.000000</td>
          <td>0.317077</td>
        </tr>
        <tr>
          <th>9.987119e-01</th>
          <td>1.000000</td>
          <td>0.241480</td>
        </tr>
        <tr>
          <th>9.993354e-01</th>
          <td>1.000000</td>
          <td>0.161605</td>
        </tr>
        <tr>
          <th>9.996519e-01</th>
          <td>1.000000</td>
          <td>0.080097</td>
        </tr>
        <tr>
          <th>9.999980e-01</th>
          <td>NaN</td>
          <td>0.000000</td>
        </tr>
      </tbody>
    </table>
    </div>



Persisting
----------

Donatello produces artifacts to enable traceability and distribution.
Objects are persribed through donatello's data accesss protocal (dap)
which gives  nested access and lazy evaluation to enable simple and narrow
user defined flexibility

.. code:: python

    ls *pkl


    Estimator.pkl  Sculpture.pkl
