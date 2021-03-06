===========
Model Types
===========

The previous example demonstrates building a model without transforming data or engineering features.

The  :py:class:`sklearn.pipeline.Pipeline` is a great place to start for feature engineering.
Donatello supports scikit-learn with a light wrapper to capture the metadata neccesary for
the model to uphold it's contracts.


Pipeline
--------

To support basic transformation donatello packages a lightly wrapped
:py:class:`sklearn.pipeline.Pipeline`. Replacing the 
Logit in the last section with a pipeline that scales is identical (save the import path)
to standard operating procedure in scikit-learn


.. code:: python

    from donatello.components.transformers import Pipeline

    def load_sculpture():
        """
        Helper to load sculpture
        """
        dataset = Dataset(raw=load_sklearn_bc_dataset(), target='is_malignant')

        model = Pipeline([('scale', StandardScaler()),
                          ('ml', LogisticRegression())])

        estimator = Estimator(model=model,
                              paramGrid={'model__ml__C': list(pd.np.logspace(-2, 0, 5))},
                              searchKwargs={'scoring': 'roc_auc', 'cv': 3},
                              method='predict_proba',
                              scorer='score_second'
                              )

        metrics = [Metric(roc_auc_score), Metric(average_precision_score),
                   FeatureWeights(sort='coefficients'), ThresholdRates()]

        sculpture = Sculpture(dataset=dataset, estimator=estimator, metrics=metrics)

        return sculpture


ModelDAG
--------

While scikit-learn pipelines (and feature unions) are a great place to start, in my experience
they have two flaws.

    #. No y or row based transforms. the imblearn has attempted to combat part of this but the nature
       of the metal of scikit-learn's Pipeline prevents this support
    #. Flexibility and composition - creating complex transforms which require different subsets\
       of different columns, while possible I've felt painful to set up and even more to edit

Enter donatello's :py:class:`donatello.components.transformers.ModelDAG`

This graph is designed to execute transformations at nodes and has configurable
transmission or *flow* along edges. The directed graph is built from a collection
:py:class:`donatello.components.transformers.Node` objects which are connected by 
:py:class:`donatello.components.transformers.DatasetFlow` objects.

Donatello's graph is designed to pass :py:class:`donatello.components.data.Dataset`
objects around. Nodes can wrap scikit-learn transformers or udfs through the 
:py:class:`donatello.components.transformers.Apply` & :py:class:`donatello.components.transformers.Access`
transformers or by creating new classes with the :py:class:`donatello.components.transformers.PandasMixin`


A scikit-learn model can be replaced directly. Using a helper method to build the model
provides a more digestable form. This next example is the same structure as building a standard
scikit-learn pipeline

Line Graph
----------

.. code:: python

   from donatello.components import transformers

   def load_model():

      # initialize model
	  model = transformers.ModelDAG(set([]), {})
	 
	  # initialize Nodes
	  n1 = transformers.Node('scale', transformers.StandardScaler(), enforceTarget=True)
	  n2 = transformers.Node('rm_outliers', transformers.Apply(func=transform, fitOnly=True))
	  n3 = transformers.Node('ml', LinearRegression())

	  # Add nodes to graph by declaring edges
	  # Edges default to the model's default Flow (which defaults to donatello's base Flow)
	  # The flow can be tuned by passing `**kwargs` if applicable
	  model.add_edge_flow(n1, n2)
	  model.add_edge_flow(n2, n3)
	  return model


Here we've built a 3 node line graph.

    #. Scale the input design data - this Node wraps a scikit-learn transformer,
       which will not return the target so we can flip the node's ``enforceTarget``
       parameter and push the dataset object 
    #. A custom udf function (transform) that will only be applied during the fit process
       (for more info see the housing prices notebook - the transform referenced 
       is an outlier remover)
    #. A Linear Regeression to execute predictions


The are many benefits to using a transformation graph but two of the most pronounced
are 

    #. reusability (components of the graph can be excised simply through the networkx api)
    #. configurability - complex **branching** and packaging more complex datasets (i.e. dicts of dataframes)


Branching Example
-----------------

This example shows a graph which 

    #. selects the numeric fields from a single table
    #. sends one field (zipcode) to a OneHotEncoder 
    #. send the remaining fields through
    #. sends the scaled fields through a outlier remover (during fit only, pass through during predict)
    #. recombines the sub tables and feeds them to a linear regression


.. code:: python

   from donatello.components import transformers

   def load_model():

       # initialize model
       model = transformers.ModelDAG(set([]), {})
       # intitate branching by selecting numeric fields
       extractor = transformers.DatasetFlow(selectMethod='dtype', selectValue=[pd.np.number], invert=False)
       n0 = transformers.Node('select', extractor)

 
       # first branch (one hot encode - we'll specify the fields to ohe in via Flow)
       n11 = transformers.Node('ohe', transformers.OneHotEncoder(dropOne=True))
       
       # second branch (scale non ohe data, and remove outliers)
       n21 = transformers.Node('scale', transformers.StandardScaler(), enforceTarget=True)
       n22 = transformers.Node('rm_outliers', transformers.Apply(func=transform, fitOnly=True))
       
       # terminal node for predicting
       n3 = transformers.Node('ml', LinearRegression())

       # send zipcode data only to OHE and don't pass target through first branch
       model.add_edge_flow(n0, n11, passTarget=False, selectValue=['zipcode'], invert=False)
       # send output of ohe to Linear Regression 
       model.add_edge_flow(n11, n3)
       
       # send all other design data and the target through second branch
       model.add_edge_flow(n0, n21, selectValue=['zipcode'], invert=True)
       model.add_edge_flow(n21, n22)
       model.add_edge_flow(n22, n3)

       return model
