Model Types
===========


The above example shows an example with out transforming data or engineering features.

Pipeline
--------

To support basic transformation donatello packages a lightly wrapped
:py:class:`sklearn.pipeline.Pipeline`. Replacing the 
Logit in the last section with a pipeline that scales is identical (save the import path)
to standard operating procedure in scikit-learn


.. code:: python
   :emphasize-lines: 1, 9,10

    from donatello.components.transformers import Pipeline

    def load_scuplture():
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

        scuplture = Sculpture(dataset=dataset, estimator=estimator, metrics=metrics)

        return scuplture


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
transmission or _flow_ along edges. The directed graph is built from a collection
:py:class:`donatello.components.transformers.Node` objects which are connected by 
:py:class:`donatello.components.transformers.DatasetFlow` objects.

Donatello's graph requires is designed to pass :py:class:`donatello.components.data.Dataaset`
objects around. Transform Nodes can wrap scikit-learn transformers or udfs through the 
:py:class:`donatello.components.transformers.Apply` & :py:class:`donatello.components.transformers.Access`
transformers or by creating new classes with the :py:class:`donatello.components.transformers.PandasMixin`


A scikit-learn model can be replaced directly. Using a helper method to build the model
provides a more digestable form, such as:

.. code:: python

   from donatello.components.transformers import ModelDAG

   def load_model():

      # initialize model
	  model = transformers.ModelDAG(set([]), {})
	 
	  # initialize Nodes
	  n1 = transformers.TransformNode('scale', transformers.StandardScaler(), enforceTarget=True)
	  n2 = transformers.TransformNode('rm_outliers', transformers.ApplyTransformer(func=transform, fitOnly=True))
	  n3 = transformers.TransformNode('ml', LinearRegression())

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
    #. configurability - complex branching and packaging more complex datasets (i.e. dicts of dataframes)


Branching Examples:

.. code:: python

   from donatello.components.transformers import ModelDAG

   def load_model():

      # initialize model
       model = transformers.ModelDAG(set([]), {})
       # intitate branching by selecting numeric fields
	   extractor = transformers.DatasetFlow(selectMethod='dtype', selectValue=[pd.np.number], invert=False)
       n0 = transformers.TransformNode('select', extractor)
                                        
 
       # first branch (one hot encode - we'll specify the fields to ohe in via Flow)
       n11 = transformers.TransformNode('ohe', transformers.OneHotEncoder(dropOne=True))
       
       # second branch (scale non ohe data, and remove outliers)
       n21 = transformers.TransformNode('scale', transformers.StandardScaler(), enforceTarget=True)
       n22 = transformers.TransformNode('rm_outliers', transformers.ApplyTransformer(func=transform, fitOnly=True))
       
       # terminal node for predicting
       n3 = transformers.TransformNode('ml', LinearRegression())

      return model

