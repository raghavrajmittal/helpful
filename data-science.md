# Data Science Framework

Combines information from
[this](https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy),
[this](),
[this](),
and my head. It was originally written for approaching the Titanic competition on Kaggle, but should be applicable elsewhere too.


### Step 1: Define the Problem
"Problems before requirements, requirements before solutions, solutions before design, and design before technology"




### Step 2: Gather the data
A training dataset is provided for most Kaggle competitions, but it might be useful to get some more data sources if needed.




### Step 3: Prepare Data for Consumption
Data wrangling includes implementing data architectures for storage and processing, developing data governance standards for quality and control, and data cleaning to identify aberrant, missing, or outlier data points.


#### 3.1 Set-up
The default Notebook on Kaggle includes a Python 3 environment that comes with many helpful analytics libraries already installed (including Numpy, Pandas, Scikit-arn etc) and is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python.

Usually ```df.head()``` or any such value printing only works if its on the last line of the cell, but one can change that to happen for all cases by changing the  interactivity settings!

```python

# import useful packages
import numpy as np
import pandas as pd
import matplotlib
import sklearn
import json

#misc libraries
import random
import time

# Useful hack so variable value is printed even if its not on the last line of the cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
```

The directory structure on Kaggle is:
```
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
for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```


#### 3.2 Meet and Greet Data
This is the meet and greet step. Get to know your data by first name and learn a little bit about it. What does it look like (datatype and values), what makes it tick (independent/ feature variables), and its goals in life (dependent/target variables). Think of it like a first date, before you jump in and start poking it in the bedroom.


Load in the input files, take a peek at them using ```df.head()```, generate descriptive statistics about the data using ```df.describe()```, and get an idea about the variable datatypes using ```df.info()```.
```python
# load data
train_data = pd.read_csv("../input/train.csv")
train_data.head()

test_data = pd.read_csv("../input/test.csv")
test_data.head()

# view statistics of data
train_data.describe()
test_data.describe()
```


#### 3.3 Data cleaning
In this stage, we will clean our data by 1) correcting aberrant values and outliers, 2) completing missing information, 3) creating new features for analysis, and 4) converting fields to the correct format for calculations and presentation, 5) Setting up X and y.

```python
# Set up X and y
features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data[features])
X_final_test = pd.get_dummies(test_data[features])
y = train_data["Survived"]
```


#### 3.4 Split training and testing data
```python
# Split into validation and training sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```


### Step 4: Exploratory Analysis
Deploy descriptive and graphical statistics to look for potential problems, patterns, classifications, correlations and comparisons in the dataset.



### Step 5: Model Data
Based on the data, machine learning can be categorized as: supervised learning, unsupervised learning, and reinforced learning. Based on the depending on your target variable and data modeling goals, machine learning algorithms can be reduced to four categories: classification, regression, clustering, or dimensionality reduction.

Supervised Learning Classification Algorithms include:
- Ensemble Methods
- Generalized Linear Models (GLM)
- Naive Bayes
- Nearest Neighbors
- Support Vector Machines (SVM)
- Decision Trees, Random Forests
- Discriminant Analysis


Process would be: determine a baseline accuracy (eg. random coin toss, or maybe predicting the most common answer), create and train a model, test model performance using cross validation, and then tune model hyper-parameters. Some times the key is using an ensemble!

PS: Keep optimizing and strategizing: iterate back through the process to make it better, faster, stronger than before!


Don't forget to train the final model on all the data!
```python
# train best model on all the data and make final predictions on test data
best_model = base_model.fit(X, y)
predictions = best_model.predict(X_final_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
```
