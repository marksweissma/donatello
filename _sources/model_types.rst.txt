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
                              gridKwargs={'scoring': 'roc_auc', 'cv': 3},
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
