# kaggle-titanic

Code for [my attempts](https://www.kaggle.com/hodapp) at
the [Titanic competition](https://www.kaggle.com/c/titanic) on Kaggle.

Thus far, this is just a `shell.nix` to set up a Python environment,
and some scratch work that loads the data via Pandas, removes some
features, and uses logistic regression and a random forest classifier
from scikit-learn.

This assumes that the `train.csv` and `test.csv` from Kaggle are
already in the local directory.
