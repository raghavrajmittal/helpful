# Random Forest

Do a RandomizedSearch first to see what might be a good hyper parameters to further search around. This doesn't run all possible combinations, but instead runs ```n_iters``` different random combinations from the param grid specified. If needed, save the results to a JSON for later reference.

```python
# Random Search for estimate of the best model parameters
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

explanation = '''
n_estimators = number of trees in random forest
max_features = number of features to consider at every split
max_depth = maximum number of levels in tree
min_samples_split = minimum number of samples required to split a node
min_samples_leaf = minimum number of samples required at each leaf node
bootstrap = method of selecting samples for training each tree
'''

# Create the random grid
param_grid = {'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
               'max_features': ['sqrt'],
               'max_depth': [10, 25, 50, 100, None],  
               'min_samples_split': [2, 5, 8, 11],
               'min_samples_leaf': [1, 2, 4],
               'bootstrap': [True]}


# Random search across 100 different combinations of the parameters using 5 fold CV 5 and all available cores
clf = RandomForestClassifier()
random_search = RandomizedSearchCV(estimator=clf, param_distributions=param_grid, n_iter=100, cv=4, random_state=0, n_jobs=-1, scoring=None, verbose=2)
random_search = random_search.fit(X_train, y_train)

print(random_search.best_params_)

# Save the params to json file for reference
with open('best_random_search_params.json', 'w') as fp:
    json.dump(random_search.best_params_, fp)
```

Next, you could do a GridSearch based on the results of the RandomizedSearch. You could even do a GridSearch directly without doing a RandomizedSearch if the search space is not too large. If needed, save the results to a JSON for later reference.
```python
# GridSearch to search for best hyper parameter values

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

explanation = '''
n_estimators = number of trees in random forest
max_features = number of features to consider at every split
max_depth = maximum number of levels in tree
min_samples_split = minimum number of samples required to split a node
min_samples_leaf = minimum number of samples required at each leaf node
bootstrap = method of selecting samples for training each tree
'''

# Create the parameter grid to search through
param_grid = {'n_estimators': [1000, 2000, 2500],
               'max_features': ['sqrt'],
               'max_depth': [100, 120],  
               'min_samples_split': [10, 11, 12],
               'min_samples_leaf': [4, 6, 8],
               'bootstrap': [True]}

clf = RandomForestClassifier()
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=4, n_jobs=-1, scoring=None, verbose=2)
grid_search = grid_search.fit(X_train, y_train)
print(grid_search.best_params_)

with open('best_grid_search_params.json', 'w') as fp:
    json.dump(grid_search.best_params_, fp)
```


Evaluate how the model is performing for unseen data (test data and test labels). Modify error metric as needed.
```python
# See how the model performs for a given dataset
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    accuracy =  (len(predictions) - sum(abs(predictions - test_labels))) / len(predictions)
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}.'.format(accuracy))
    return accuracy


base_model = RandomForestClassifier()
base_model.fit(X_train, y_train)
base_accuracy = evaluate(base_model, X_test, y_test)

best_random_search_model = random_search.best_estimator_
random_accuracy = evaluate(best_random_search_model, X_test, y_test)

best_grid_search_model = grid_search.best_estimator_
grid_accuracy = evaluate(best_grid_search_model, X_test, y_test)
```
