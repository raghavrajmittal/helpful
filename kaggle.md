# Kaggle


## Set up
The default Python 3 environment comes with many helpful analytics libraries installed (including Numpy, Pandas, Scikit-arn etc) and is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python.

```python
import numpy as np 
import pandas as pd
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



## Load and view data
Usually ```df.head()``` or any such value printing only works if its on the last line of the cell, but one can change that to happen for all cases by changing the  interactivity settings! ```df.describe()``` generates escriptive statistics about the data.

```python
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# load data
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.head()

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()

# view statistics of data
train_data.describe()
test_data.describe()
```



## Random Forest

To look at parameters used by our current forest
```python
# [arameters currently in use
from pprint import pprint
pprint(rf_model.get_params())
```
```
{'bootstrap': True,
 'criterion': 'mse',
 'max_depth': None,
 'max_features': 'auto',
 'max_leaf_nodes': None,
 'min_impurity_decrease': 0.0,
 'min_impurity_split': None,
 'min_samples_leaf': 1,
 'min_samples_split': 2,
 'min_weight_fraction_leaf': 0.0,
 'n_estimators': 10,
 'n_jobs': 1,
 'oob_score': False,
 'random_state': 42,
 'verbose': 0,
 'warm_start': False}
```
