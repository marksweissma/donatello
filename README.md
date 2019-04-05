Note donatello is not officially released, there is a test wheel on `test.pypi.org`
if you want to take a look. The core is intact and basic examples are all turn key

`pip install -i https://test.pypi.org/simple/ donatello`


Docs via [github pages](https://marksweissma.github.io/donatello/)

Donatello is a modeling framework designed to improve iteration speed of developemnt while bringing traceability and reproducibility
without adding overhead to data scientist's development time

Donatello is comprised of `components` supported by a set of `utilities`. 
Components are the core, and utilities help manage the codebase as well aid binding metadata with the objects.
The `transformers` components  support robust bindings between `pandas` and `scikit-learn` for a range of applications from helping understand feature importances to enabling the DataFrame functionality that `numpy` does not support (well).

Supported model types:

    #. Scikit-learn model objects (fit, predict/predict_proba) directly. 
    #. Donatello has a light wrapper around scikit-learn's Pipeline to support the required metadata
    #. Donatello comes with a DAG executor for building, managing, sharing, and reusing transformations, and supports composibility throughout objects.
  
Components:
  
  1. The `Scuplpture` object coordinates data, leakage free data subsetting, fitting, cross validating, measuring, and persisting. 
  2. The `Estimator` objects encapsulate the transformation process for the expected requirements to serve and the fitting/predicting/scoring of ML algorithims. 
  3. The `Data` objects are responsible for subsetting, storing, and accessing data.
  4. The `Measure` objects are responsbile for evaluating models on unseen data, via `Metric` *functors* through cross validaiton and hold out scoring.

Questions, comments, feature requests please reach out!
