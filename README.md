Docs via [github pages](https://marksweissma.github.io/donatello/)

Donatello is a modeling framework designed to improve iteration speed of developemnt while bringing traceability and reporducibility
without additional overhead

A few notes before full docs are out - 
Donatello is comprised of `components` supported by a set of `utilities`. Components are the core, and utilities help manage the codebase as well aid binding metadata with the objects. The `transformers` utility current supports robust bindings between `pandas` and `scikit-learn` for a range of applications from helping understand feature importances to enabling the DataFrame functionality that `numpy` does not support (well).
  
  Components:
  
  1. The `Scuplpture` object is coordinates pulling data, splitting, fitting, cross validating, measuring, and persisting. 
  2. The `Estimator` objects encapsulate the transformation process for the expected requirements to serve and the fitting/predicting/scoring of ML algorithims. 
  3. The `Measure` objects are responsbile for evaluating models on unseen data, via cross validaiton and hold out
  4. The `Data` objects are responsible for fetching, storing, and managing data

Questions, comments, feature requests please reach out!
