Sculpture
=========

To get started, connect a dataset via
:py:class:`donatello.components.data.Dataset` and
:py:class:`donatello.components.estimator.Estimator` to
:py:class:`donatello.components.core.Sculpture`

Sculptures are modeling object that uphold scikit-learn's estimator contracts

    #. ``__init__`` cannot mutate parameters
    #. ``get_params`` and ``set_params`` support
    #. ``fit``, ``transform``, and ``fit_transform`` support


Sculptures can be embedded as nodes in :py:class:`donatello.components.transformers.ModelDAG` as well 
as transformers in :py:class:`sklearn.pipeline.Pipeline`

Donatello approaches model configuration as declaring intent. Everything required
to build evaluate and produce artifacts is specified during instantiation.


Intent
------

.. literalinclude::  ../../donatello/examples/intent_simple.py

Dataset
-------

The dataset can be specifed through 

    #. explicit ``X`` and ``y`` (if supervised)
    #. a ``raw`` table and a reference to the ``target`` (if supervised)
    #. a collection of ``raw`` tables with a ``primaryKey`` to merge along + ``target`` (if supervised)


Estimator
---------

The estimator object requires a ``model``, and a reference to the ``method`` of the model to call. Optionally
a callback to transform the raw output can be supplied through the ``scorer``. To enable hyperparameter tuning
a parameter grid and search arguments can be supplied. Currently donatello only supports grid searching through
the scikit-learn API, which prevents searching over input datasets which are collections of tables. Until this
functionality is built out it can be hacked around via :py:class:`donatello.components.transformers.ModelDAG`
and embedding a ``Sculpture`` as a node downstream of a node that combines the data. The ``Dataset`` will manage
the indexing to prevent leakage.

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
over which (subsets) of data estimators are fit and metrics are calculated (if applicable)


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

Declared by the given flags above, this sculpture will perform a 5 fold
stratified K Fold cross validation within the training subset of the data
and then fit a model over the entire training set and evaluate on the hold out set.

Per scikit-learn Transformer pattern, fitting returns the object itslef

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

    Building Over Holdout
    grid searching

    Sculpture_2019_03_24_13_02



Evaluating
----------

During the fitting process, metrics are calculated over the
specified samples of data and stored in a :py:class:`sklearn.utils.Bunch`
(a lighlty wrapped dict, with attribute style accessors)

This information is attached to the Sculpture in the ``measurements`` attribute


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


Because all scikit-learn scorers and many metrics are scalar returns
:py:func:`donatello.utils.helpers.view_sk_metric` will unfurl the bunch
into a flat table


.. code:: python

    from donatello.utils import helpers
    helpers.view_sk_metric(sculpture.measurements.crossValidation.average_precision_score)

.. include:: tables/pr_score.rst



The feature weights metric is a short cut to pulling coefficients for glms
and feature_importances from ensemble method


.. code:: python

    sculpture.measurements.crossValidation.feature_weights.mean


.. include:: tables/feature_weights.rst


The Threshold Rates Metric helps parameterize the binary confusion matrix
by sampling scores from the held out data and evaluting the rate

.. code:: python

    sculpture.measurements.crossValidation.ThresholdRates.mean.loc[::5]

.. include:: tables/threshold_rates.rst

Persisting
----------

Donatello produces artifacts to enable traceability and distribution.
Objects are persribed through donatello's data accesss protocal (dap)
which gives  nested access and lazy evaluation to enable simple and narrow
user defined flexibility

.. code:: python

    ls *pkl


    Estimator.pkl  Sculpture.pkl
