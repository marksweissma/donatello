Docs to come via github pages (please check back in the coming weeks)!

Donatello is a modeling framework designed to help enforce best practice while (hopefully) making it as easy possible to spin up new models with quality reporting through Jupyter notebooks.

A few notes before full docs are out - 
Donatello is comprised of `components` supported by a set of `utilities`. Components are the core, and utilities help manage the codebase as well aid binding metadata with the objects. The `transformers` utility current supports robust bindings between `pandas` and `scikit-learn` for a range of applications from helping understand feature importances to enabling the DataFrame functionality that `numpy` does not support (well).
  
  Components:
  
  1. The `DM` (donatello manager) object is  the backbone coordinating pulling data, splitting, fitting, cross validating, scoring, and storing. 
  2. The `Estimator` objects encapsulate the transformation process for the expected requirements to serve and the fitting/predicting/scoring of ML algorithims. 
  3. The `Scorer` objects are responsbile for evaluating models on an unseen data, via cross validaiton and hold out
  4. The `Data` objects are responsible for storing and managing data

Questions, comments, feature requests please reach out!
