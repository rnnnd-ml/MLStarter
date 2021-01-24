A general overview on how scikit learn works

### Instantiate a model

Always start off with initializing a learning model. It could be 

* `LinearSVC, LinearSVR, SVC, SVR` from `sklearn.svc`

* `LinearRegression, LogisticRegression` from `sklearn.linearmodel`

  and so on.

```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
```

### Fit your model

Say your data is in X, y. There is a way to split it automatically into training and test data.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0) 
```

Then you fit the model on the training data

```
model = lr.fit(X_train, y_train) # Parameters needed here generally
```

### Predict with the model

```python
pred = model.predict(X_test)
```

This is the basic workflow with scikit-learn.

### Get a model score

All models have a score method, all you need to do is `model.score(X_test, y_test)`. For more info, import relevant metrics from sklearn.models.

### Pipelines

You need to do a lot of preprocessing for the more complex models (decision trees, svms), so create a pipeline which does preprocessing and training both.

```python
from sklearn.pipeline import Pipeline
pl = Pipeline([('preprocessor', PreProcessor()), ('fitter', LinearRegression())])
mdl = pipe.fit(X_train, y_train)
print(mdl.score(X_test, y_test))
```

### Hyperparameter search

You can use GridSearchCV or RandomizedSearchCV. Basically, create a dictionary, with parameter: [paramvalues to test] and pass this to the search method

```python
param_grid = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth': range(4:4+5)
}
from sklearn.model_selection import GridSearchCV
CV = GridSearchCV(classifier, param_grid)
CV.fit(X_train, y_train)
print(CV.best_params_, CV.best_score_)
```



### Importing reference datasets

```
from sklearn import datasets

digits = datasets.load_digits()
```

There is tons more - boston, diabetes, digits, iris, moons...

Or if you have csv, json files, use the appropriate read method from pandas. 









### Useful links

Basically the first page of google search!

1. https://towardsdatascience.com/a-beginners-guide-to-scikit-learn-14b7e51d71a4

2. https://kapernikov.com/tutorial-image-classification-with-scikit-learn/
3. https://machinelearningmastery.com/a-gentle-introduction-to-scikit-learn-a-python-machine-learning-library/
4. Cheatsheets for pandas, numpy, keras and more! - https://www.datacamp.com/community/data-science-cheatsheets







