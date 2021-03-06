Note donatello is not officially released, there is a test wheel on `test.pypi.org`
if you want to take a look. The core is intact and basic examples are all turn key

The test wheel is only python 2 compatible. Python 3 compatibility (and tests) are in the works.
Donatello will be py2/3 compatible upon official release to pypi. 

For now:

`pip install -i https://test.pypi.org/simple/ donatello`


For walk through see `notebooks`, for interactive runnable examples in `donatello/examples` see simple, pipeline or line_graph intents.

Docs via [github pages](https://marksweissma.github.io/donatello/)

Donatello is a modeling framework designed to improve iteration speed of developemnt while bringing traceability and reproducibility
without adding overhead to data scientist's development time

Donatello is comprised of `components` supported by a set of `utilities`. 
Components are the core, and utilities help manage the codebase and generic non-sevice oriented functionality.
The `transformers` components support robust bindings between `pandas` and `scikit-learn` for a range of applications from helping understand feature importances to enabling the DataFrame functionality that `numpy` does not support (well).

Supported model types:

    1. Donatello comes with a DAG executor for building, managing, sharing, and reusing transformations, while supporting composibility throughout objects.
    2. Scikit-learn model objects (fit, predict/predict_proba) directly. 
    3. Donatello has a light wrapper around scikit-learn's Pipeline to support the required metadata
  
Components:
  
  1. The `Scuplpture` object coordinates data access, fitting, cross validating, measuring, and persisting. 
  2. The `Estimator` objects encapsulate the transformation process for the expected requirements to serve and the fitting/predicting/scoring of ML algorithims. 
  3. The `Data` objects are responsible for subsetting, storing, and giving data access.
  4. The `Measure` objects are responsbile for evaluating models on unseen data, via `Metric` *functors* through cross validaiton and hold out scoring.

Questions, comments, feature requests please reach out!


Author Note:
This project was incepted as a way to learn about decorators (before starting I had never written one). While I've
learned a lot through that process, some of the technical debt in this project is still an artifact of that motivation and that lack of experience.
