# Kaggle



&nbsp;
## Set up
The default Notebook on Kaggle includes a Python 3 environment that comes with many helpful analytics libraries already installed (including Numpy, Pandas, Scikit-arn etc) and is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python.

Usually ```df.head()``` or any such value printing only works if its on the last line of the cell, but one can change that to happen for all cases by changing the  interactivity settings!

```python
# import useful libraries
import numpy as np
import pandas as pd
import json

# Useful hack so variable value is printed even if its not on the last line of the cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
```


The directory structure is:
```bash
└── kaggle
    └── input
        ├── train.csv
        └── test.csv
    └── working
        ├── notebook.ipynb
        └── my_submission.csv
```
* The current directory is /kaggle/working/
* Input data files are available in the read-only "../input/" directory.
* You can write up to 5GB to /kaggle/working/ that gets preserved as output when you create a version using 'Save & Run All'
* You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
 
```python
# to list all the input files available
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```



&nbsp;
## Load and view data
Load in the input files, and take a peek at them using ```df.head()```. To generate descriptive statistics about the data, run  ```df.describe()```.
```python
# load data
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()

# view statistics of data
train_data.describe()
test_data.describe()
```


Pre-process the data as needed, and split into training and validation sets:
```python
from sklearn.model_selection import train_test_split

# Preprocess data, set up X and y
features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data[features])
X_final_test = pd.get_dummies(test_data[features])
y = train_data["Survived"]

# Split into validation and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```



&nbsp;
## Random Forest
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


Make final predictions using the best model. The CSV is saved as output when the 'Save & Run All' option is selected on Kaggle.
```python
# train best model on all the data and make final predictions on test data
best_model = base_model.fit(X, y)
predictions = best_model.predict(X_final_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
```
