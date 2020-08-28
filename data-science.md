# Data Science Framework

It was originally written for approaching the Titanic competition on Kaggle, but should be applicable elsewhere too.

Ideas, process, and code is heavily influenced by some amazing kaggle notebooks, including but not limited to [this](https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy), [this](https://www.kaggle.com/shreyasvedpathak/titanic-survival-prediction-top-4#Defining-Models), and [this](https://www.kaggle.com/gunesevitan/titanic-advanced-feature-engineering-tutorial#2.-Feature-Engineering).



## Step 1: Define the Problem
> If you define the problem correctly, you almost have the solution.
> \- Steve Jobs

Hmm not entirely true, but regardless, you get the idea.



## Step 2: Gather the data
A training dataset is provided for most Kaggle competitions, but it might be useful to get some more data sources if needed.




## Step 3: Prepare Data for Consumption
Data wrangling includes implementing data architectures for storage and processing, developing data governance standards for quality and control, and data cleaning to identify aberrant, missing, or outlier data points.


#### 3.1 Set-up
The default Notebook on Kaggle includes a Python 3 environment that comes with many helpful analytics libraries already installed (including Numpy, Pandas, Scikit-arn etc) and is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python.

Usually ```df.head()``` or any such value printing only works if its on the last line of the cell, but one can change that to happen for all cases by changing the  interactivity settings!

```python
# import useful libraries
import numpy as np
import pandas as pd
import json

from pprint import pprint

import matplotlib.pyplot as plt
import seaborn as sns

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

Load in the input files, take a peek at them using ```df.head()```, generate descriptive statistics about the data using ```df.describe()```, and get an idea about the variable datatypes using ```df.info()```.
```python
# load data
train_data = pd.read_csv("input/train.csv")
test_data = pd.read_csv("input/test.csv")

# Take a peek at the data
train_data.head()
test_data.head()

# view statistics of data
train_data.describe()
test_data.describe()

# view info of data
train_data.info()
test_data.info()
```

Set up some basic functions to join/separate the testing and training sets. They are usually very handy.

```python
# Returns a concatenated df of training and test set
def concat_df(train_data, test_data):
    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)

# Returns divided dfs of training and test set
def divide_df(all_data):
    return all_data.loc[:890], all_data.loc[891:].drop(['Survived'], axis=1)
```

#### 3.2 Meet and Greet Data
> This is the meet and greet step. Get to know your data by first name and learn a little bit about it. What does it look like (datatype and values), what makes it tick (independent/ feature variables), and its goals in life (dependent/target variables). Think of it like a first date, before you jump in and start poking it in the bedroom.
> \- ldfreeman3


To plot the relationship between a numerical and one or more categorical variables, use ```seaborn.catplot```:
```python
sns.catplot(x="SibSp", col = 'Survived', data=train_data, kind = 'count', palette='pastel')
plt.show()
```

To plot any conditional relationships:
```python
g = sns.FacetGrid(train_data, col='Survived')
g = g.map(sns.distplot, "Age")
```

To flexibly plot a univariate distribution of observations, use ```seaborn.distplot```:
```python
f, axes = plt.subplots(2, 1, figsize = (10, 6))

g1 = sns.distplot(train_data["Fare"], color="red", label="Skewness : %.2f"%(train_data["Fare"].skew()), ax=axes[0])
axes[0].title.set_text('Fare')
axes[0].legend()

g2 = sns.distplot(train_data["Age"], color="blue", label="Skewness : %.2f"%(log_fare.skew()), ax=axes[1])
axes[1].title.set_text('Age')
axes[1].legend()

plt.tight_layout()
```


To get a list of the highest correlating attributes:
```python
# get correlations between features
df_train_corr = train_data.drop(['PassengerId'], axis=1).corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
df_train_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
df_train_corr.drop(df_train_corr.iloc[1::2].index, inplace=True)
df_train_corr_nd = df_train_corr.drop(df_train_corr[df_train_corr['Correlation Coefficient'] == 1.0].index)

df_test_corr = test_data.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
df_test_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
df_test_corr.drop(df_test_corr.iloc[1::2].index, inplace=True)
df_test_corr_nd = df_test_corr.drop(df_test_corr[df_test_corr['Correlation Coefficient'] == 1.0].index)

# Show high correlations
high_train_corr = df_train_corr_nd['Correlation Coefficient'] > 0.1
high_test_corr = df_test_corr_nd['Correlation Coefficient'] > 0.1

df_train_corr_nd[high_train_corr]
df_test_corr_nd[high_test_corr]
```


To plot the heatmap of correlations between the attributes:
```python
# see correlation heatmap
fig, axs = plt.subplots(nrows=2, figsize=(20, 20))

sns.heatmap(train_data.drop(['PassengerId'], axis=1).corr(), ax=axs[0], annot=True, square=True, cmap='coolwarm', annot_kws={'size': 14})
sns.heatmap(test_data.drop(['PassengerId'], axis=1).corr(), ax=axs[1], annot=True, square=True, cmap='coolwarm', annot_kws={'size': 14})

for i in range(2):    
    axs[i].tick_params(axis='x', labelsize=14)
    axs[i].tick_params(axis='y', labelsize=14)

axs[0].set_title('Training Set Correlations', size=15)
axs[1].set_title('Test Set Correlations', size=15)

plt.show()
```




#### 3.3 Data cleaning
To display the number of missing values for each column:
```Python
# show number of missing values for each column
def display_missing(df):
    print('\n')
    for col in df.columns.tolist():         
        num_missing = df[col].isnull().sum()
        if num_missing > 0:
            print('{} column missing values: {}'.format(col, num_missing))

display_missing(train_data)
display_missing(test_data)
```

Next, complete the missing data:
```python
all_data = concat_df(train_data, test_data)

# complete missing data
all_data['Age'] = all_data.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
all_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace = True)
all_data['Fare'].fillna(train_data['Fare'].median(), inplace = True)

train_data, test_data = divide_df(all_data)
```

#### 3.4 Feature Engineering
Next, do some feature engineering to create new features or modify existing ones:
```Python
all_data = concat_df(train_data, test_data)

#feature_engineering
all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1
all_data['IsAlone'] = 0
all_data.loc[all_data['FamilySize'] > 1, 'IsAlone'] = 1

all_data["FareLog"] = all_data["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
all_data['FareLogBin'] = pd.qcut(all_data['FareLog'], 13)
all_data['FareBin'] = pd.qcut(all_data['Fare'], 13)

all_data['AgeBin'] = pd.cut(all_data['Age'].astype(int), 10)

all_data['Title'] = all_data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
title_names = (all_data['Title'].value_counts() < 10)
all_data['Title'] = all_data['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)

train_data, test_data = divide_df(all_data)
```

#### 3.4 Feature Selection
Next, convert all features to numerical features (create dummies for categorical features), scale the training and testing sets based on the training set, and set up X and y:
```Python
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


# encode labels
non_numeric_features = ['AgeBin', 'FareBin', 'FareLogBin']
all_data = concat_df(train_data, test_data)
for feature in non_numeric_features:
    all_data[feature] = LabelEncoder().fit_transform(all_data[feature])
train_data, test_data = divide_df(all_data)


# feature selection
numerical_features = ["AgeBin", "FareBin", "IsAlone", "FamilySize", "Pclass"]
categorical_features = ["Sex", 'Embarked', 'Title']

X = train_data[numerical_features + categorical_features]
X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

X_final_test = test_data[numerical_features + categorical_features]
X_final_test = pd.get_dummies(X_final_test, columns=categorical_features, drop_first=True)

y = train_data["Survived"]


# #Scaling
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
X_final_test = scaler.transform(X_final_test)

print('X_train shape: {}'.format(X.shape))
print('y_train shape: {}'.format(y.shape))
print('X_test shape: {}'.format(X_final_test.shape))
```




#### 3.4 Split training and testing data
```python
# Split into validation and training sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# OR

# Define a CV split, and pass it to the cross_validate or GridSearchCV
from sklearn import model_selection
cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .7, random_state = 0)


```



## Step 4: Model Data

Deploy descriptive and graphical statistics to look for potential problems, patterns, classifications, correlations and comparisons in the dataset.

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


Don't forget to train the final model on all the data if required!
```python
# train best model on all the data and make final predictions on test data
best_model = grid_search.best_estimator_
best_model = best_model.fit(X, y)

predictions = best_model.predict(X_final_test).astype(int)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
```
