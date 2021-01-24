
* **Boosting** refers to using a bunch of weak estimators to make a strong estimator.
* A large part of XGBoost's popularity rests on its use in Kaggle.

`xgboost` has `XGBRegressor, XGBClassifier, XGBRanker` models.

Let us see how a plain DecisionTree would work

```python
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
pred = clf.fit(X_test)
print(clf.score(pred, y_test))
```

### Hyperparameters

* `max_depth, min_samples_leaf, max_leaf_nodes, max_features (how many features considered in split), min_samples_split (min. needed for split to happen), criterion (defaults to mse for ref, gini for class)`
* For GridSearchCV, the important ones are cv (what type of cross-validation? it does cv-fold stratified CV)

### Putting it all together for the heart-disease dataset

Preliminaries

```python
df = pd.read_csv("")
X = df.iloc[:, :-1]; y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y)
```



```python
model = DecisionTreeClassifier()
scores = cross_val_score(model, X, y, cv=5)
#scores will have the accuracy scores
```

Using RandomizedSearchCV

```python
def randomized_search_clf(params, runs=20, clf=DecisionTreeClassifier(random_state=42)):
    rand_clf = RandomizedSearchCV(clf, params, n_iter=runs, cv=5, random_state=42)
    rand_clf.fit(X_train, y_train)
    best_model = rand_cld.best_estimator_
    best_score = rand_clf.best_score_ # this is the training score
    accuracy = accuracy_score(y_train, best_model.predict(X_test))
    return best_model

params = {
    'criterion': ['entropy', 'gini'],
    'splitter': ['random', 'best'],
    'max_leaf_nodes': [10, 15, 20] # add more and run with this dictionary
}
```

After training, you can use the `feature_importances_` to find which ones are actually important.

In the examples, before training and even after training, to get the right 'score', they use the cross_val_score method to get a score. This is not a score for a model, but for a set of hyperparameters. It is also what happens with the cv variable used in grid search and randomized search.

## Using RandomForestClassifier

Hyperparameters (for the ensemble method) are

* oob_score - set it to True and oob_scores are retained which you can access later using the variable oob_score_
* n_estimators

`from sklearn.ensemble import RandomForestClassifier`

### Using GradientBoostingClassifier and xgboost

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
grb = GradientBoostingClassifier(n_estimators=100, max_depth=2, random_state=2)
gbr.fit(X_train, y_train)
y_pred = gbr.predict(X_test)
score = accuracy_score(y_pred, y_test)
```

```python
from xgboost import XGBClassifier
xg_reg = XGBClassifier(n_estimators=100, max_depth=2, random_state=2)
# rest is the same
```

### XGBoost specifics

The hyperparameters are

* n_estimators, learning_rate, max_depth
* objective - xgboost can work out the relevant one ('multi:softprob', 'binary:logistic' and more)
* booster - the base learner. Default works well
* subsample - 100% means all rows are used, decrease to prevent overfitting



xgb can also handle missing data. Use something like

```python
xgb_clf = xgb.DMatrix(X, y, missing=-999, weight=df['test_weight'])
```

### TODO

* Learn about weights
* using xgboost's train is better than the scikit compatible fit, similarly converting data into DMatrices is very useful.



