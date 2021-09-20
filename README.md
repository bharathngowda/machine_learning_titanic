
# Titanic - Machine Learning from Disaster

![App Screenshot](https://github.com/bharathngowda/machine_learning_titanic_survival_prediction/blob/main/titanic.jpg)

### Table of Contents

1. [Problem Statement](#Problem-Statement)
2. [Data Pre-Processing](#Data-Pre-Processing)
3. [Exploratory Data Analysis](#Exploratory-Data-Analysis)
4. [Feature Engineering](#Feature-Engineering)
5. [Model Training](#Model-Building)
6. [Model Selection](#Model-Selection)
7. [Hyperparameter Tuning](#Hyperparameter-Tuning)
8. [Model Evaluation](#Model-Evaluation)
9. [Dependencies](#Dependencies)
10. [Installation](#Installation)

### Problem Statement

The sinking of the Titanic is one of the most infamous shipwrecks in history.

On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).

|Variable|	Definition	|Key|
| :-------- | :------- |:------- |
|survival|	Survival|	0 = No, 1 = Yes|
|pclass	|Ticket class|	1 = 1st, 2 = 2nd, 3 = 3rd|
|sex	|Sex	|
|Age	|Age in years	|
|sibsp|	# of siblings / spouses aboard the Titanic	|
|parch|	# of parents / children aboard the Titanic	|
|ticket|	Ticket number	|
|fare|	Passenger fare	|
|cabin|	Cabin number	|
|embarked|	Port of Embarkation|	C = Cherbourg, Q = Queenstown, S = Southampton|

**Quick Start:** [View](https://github.com/bharathngowda/machine_learning_titanic_survival_prediction/blob/main/Titanic%20Survival%20Prediction.ipynb) a static version of the notebook in the comfort of your own web browser

### Data Pre-Processing

- Loaded the train and test data
- Checking if the data is balanced i.e. whether the count of survived and not survived is equal or not in train set.
- Checking for null values 
- Imputing null values 


### Exploratory Data Analysis

- Survival Status based on Sex
- Survival Status based on Age
- Survival Status based on Family Name
- Survival Status based on Ticket Fare
- Survival Status based on Passenger Class
- Survival Status based on from where they embarked
- Survival Status based on family type i.e., large, small or single
- Survival Status based on Title i.e., Mr., Ms., Master, Royalty, Officer
- Correlation plot 

### Feature Engineering

New features created are 'Title', 'Family Type', 'Last Name'

### Model Training

Models used for the training the dataset are - 

- [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Linear Discriminant Analysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)
- [Linear Support Vector Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC)
- [Support Vector Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [Decision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- [Bagging](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html)
- [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [K-Nearest Neighbours](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- [XGB Classifier](https://xgboost.readthedocs.io/en/latest/)

### Model Selection

Since the dataset is imbalanced, I have used [f1 score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) as my scorer and used [k-fold cross-validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)
to select the model with highest **'f1 score'**.

### Hyperparameter Tuning

I have performed Hyperparameter Tuning on the selected model using [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) and selected the best parameters to obtain the final model.


### Model Evaluation

I fit the final model on the train data and predicted the survival status for the test data and obtained the below result-

| Metric    | Score    |
| :-------- | :------- |
| Accuracy  |0.76315   |

### Dependencies
* [NumPy](http://www.numpy.org/)
* [IPython](http://ipython.org/)
* [Pandas](http://pandas.pydata.org/)
* [SciKit-Learn](http://scikit-learn.org/stable/)
* [SciPy](http://www.scipy.org/)
* [Matplotlib](http://matplotlib.org/)
* [Seaborn](https://seaborn.pydata.org/)

### Installation

To run this notebook interactively:

1. Download this repository in a zip file by clicking on this [link](https://github.com/bharathngowda/machine_learning_titanic_survival_prediction/archive/refs/heads/main.zip) or execute this from the terminal:
`git clone https://github.com/bharathngowda/machine_learning_titanic_survival_prediction.git`

2. Install [virtualenv](http://virtualenv.readthedocs.org/en/latest/installation.html).
3. Navigate to the directory where you unzipped or cloned the repo and create a virtual environment with `virtualenv env`.
4. Activate the environment with `source env/bin/activate`
5. Install the required dependencies with `pip install -r requirements.txt`.
6. Execute `ipython notebook` from the command line or terminal.
7. Click on `Titanic Survival Prediction.ipynb` on the IPython Notebook dasboard and enjoy!
8. When you're done deactivate the virtual environment with `deactivate`.
