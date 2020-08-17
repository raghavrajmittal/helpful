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
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier

# base model
base_model = RandomForestClassifier(random_state=0)
base_results = model_selection.cross_validate(base_model, X, y, cv=cv_split, return_train_score=True)

print("BEFORE Search - Training score mean: {:.2f}". format(base_results['train_score'].mean()*100))
print("BEFORE Search - Test score mean: {:.2f}". format(base_results['test_score'].mean()*100))
print("BEFORE Search - Test score 3*std: +/- {:.2f}". format(base_results['test_score'].std()*100*3))


param_grid = {'n_estimators': [50, 100, 500],
               'criterion': ['gini', 'entropy'],
               'max_features': ['sqrt'],
               'max_depth': [6, 8, 10, None],
               'min_samples_split': [2, 5, 8, 11],
               'bootstrap': [True],
               'random_state': [0]
             }

grid_search = model_selection.GridSearchCV(RandomForestClassifier(random_state=0), param_grid=param_grid, scoring='roc_auc', cv=cv_split, return_train_score=True)
grid_search = grid_search.fit(X, y)

print('AFTER Search - Parameters: ', grid_search.best_params_)
print("AFTER Search - Training score mean: {:.2f}". format(grid_search.cv_results_['mean_train_score'][grid_search.best_index_]*100))
print("AFTER Search - Test score mean: {:.2f}". format(grid_search.cv_results_['mean_test_score'][grid_search.best_index_]*100))
print("AFTER Search - Test score 3*std: +/- {:.2f}". format(grid_search.cv_results_['std_test_score'][grid_search.best_index_]*100*3))

with open('best_grid_search_params.json', 'w') as fp:
    json.dump(grid_search.best_params_, fp)
```


Evaluate how the model is performing for unseen data (test data and test labels). Modify error metric as needed.
```python
# See how the model performs for a given dataset
from sklearn.metrics import confusion_matrix

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    accuracy =  (len(predictions) - sum(abs(predictions - test_labels))) / len(predictions)
    cm = confusion_matrix(test_labels, predictions)

    print('Model Performance')
    print('Average Error: {:0.4f}.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}'.format(accuracy))
    print("Confusion Matrix:\n", cm)

base_model = base_model.fit(X_train, y_train)
evaluate(base_model, X_test, y_test)

best_grid_search_model = grid_search.best_estimator_.fit(X_train, y_train)
evaluate(best_grid_search_model, X_test, y_test)

```
