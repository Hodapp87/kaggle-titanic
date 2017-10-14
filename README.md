# kaggle-titanic

Code for [my attempts](https://www.kaggle.com/hodapp) at
the [Titanic competition](https://www.kaggle.com/c/titanic) on Kaggle.

This assumes that the `train.csv` and `test.csv` from Kaggle are
already in the local directory.

So far this contains:

- A `shell.nix` to set up a Python environment
- Some early scratch work in `scratch.py` to load data in Pandas, do
  some feature transformation, and build models for logistic
  regression and a random forest classifier.
- A Jupyter notebook that copies most of the above and has a variety
  of other exploration.
