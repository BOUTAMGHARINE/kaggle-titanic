# Titanic - Machine Learning from Disaster

## Project Overview

This project is based on the Kaggle Titanic competition.

The objective is to build a machine learning model that predicts whether a passenger survived the Titanic disaster using information such as passenger class, sex, age, family relationships, ticket information, and fare.

The project was developed using Python and scikit-learn.

The main goal of the project is not only to obtain a good prediction score, but also to understand the complete machine learning pipeline:

* Data preprocessing
* Exploratory data analysis
* Feature engineering
* Model training
* Validation
* Cross-validation
* Prediction
* Kaggle submission

---

## Project Structure

```text
project/
│
├── README.md
├── requirements.txt
├── username.txt
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── gender_submission.csv
│
├── notebook/
│   └── main.ipynb
│
└── scripts/
    ├── data_preprocessing.py
    ├── feature_engineering.py
    └── train.py
```

---

## Dataset

The project uses the Titanic dataset provided by Kaggle.

The training dataset contains passenger information together with the target variable `Survived`.

The main columns include:

* `PassengerId`
* `Survived`
* `Pclass`
* `Name`
* `Sex`
* `Age`
* `SibSp`
* `Parch`
* `Ticket`
* `Fare`
* `Cabin`
* `Embarked`

The `Survived` column is the target variable:

```text
0 = Did not survive
1 = Survived
```

The test dataset does not contain the `Survived` column. The trained model is used to predict it.

---

## Data Preprocessing

Several preprocessing operations were applied before training the model.

### Missing Age values

Missing values in `Age` were replaced using the median age calculated from the training dataset.

```python
age_median = train["Age"].median()

train["Age"] = train["Age"].fillna(age_median)
test["Age"] = test["Age"].fillna(age_median)
```

The median was used because it is less sensitive to extreme values than the mean.

### Missing Embarked values

Missing values in `Embarked` were replaced using the most frequent value in the training dataset.

```python
embarked_mode = train["Embarked"].mode()[0]

train["Embarked"] = train["Embarked"].fillna(embarked_mode)
test["Embarked"] = test["Embarked"].fillna(embarked_mode)
```

### Missing Fare values

The missing `Fare` value in the test dataset was replaced using the median fare from the training dataset.

```python
fare_median = train["Fare"].median()

train["Fare"] = train["Fare"].fillna(fare_median)
test["Fare"] = test["Fare"].fillna(fare_median)
```

### Cabin

The `Cabin` column contains a large number of missing values, so it was removed from the final dataset.

---

## Feature Engineering

Feature engineering was one of the most important parts of the project.

The objective was to transform the original passenger information into features that could provide useful information to the machine learning model.

### FamilySize

`FamilySize` represents the total number of people in a passenger's family.

```python
FamilySize = SibSp + Parch + 1
```

The `+1` represents the passenger themselves.

This feature can help the model distinguish between passengers travelling alone and passengers travelling with family members.

---

### IsAlone

A binary feature was created to indicate whether a passenger was travelling alone.

```python
IsAlone = (FamilySize == 1).astype(int)
```

Values:

```text
0 = Passenger was travelling with other family members
1 = Passenger was travelling alone
```

---

### Sex

The categorical `Sex` feature was converted into numerical values.

```python
male = 0
female = 1
```

Sex is an important feature for the Titanic problem because survival rates were significantly different between male and female passengers.

---

### Embarked

The `Embarked` feature was converted into numerical values.

```text
S = 0
C = 1
Q = 2
```

---

### Sex_Pclass

An interaction feature was created between `Sex` and `Pclass`.

```python
Sex_Pclass = Sex * Pclass
```

This combines information about the passenger's sex and class.

For example, the model can distinguish more easily between female passengers from different passenger classes and male passengers from different passenger classes.

This feature improved the local validation accuracy compared with the simpler baseline.

---

### Child

A binary feature was created to identify children.

```python
Child = (Age < 16).astype(int)
```

Values:

```text
0 = Passenger is 16 or older
1 = Passenger is younger than 16
```

This provides additional information about young passengers.

---

### TicketGroupSize

The number of passengers sharing the same ticket was calculated.

```python
ticket_counts = pd.concat([
    train["Ticket"],
    test["Ticket"]
]).value_counts()
```

The resulting value was stored as `TicketGroupSize`.

This provides information about whether a passenger was travelling as part of a larger group.

---

### WomanOrChild

A binary feature was created to identify female passengers or children.

```python
WomanOrChild = (
    (Sex == 1) | (Age < 16)
).astype(int)
```

This combines two passenger characteristics that are relevant to the survival patterns in the dataset.

---

### Age_Pclass

An interaction between age and passenger class was created.

```python
Age_Pclass = Age * Pclass
```

This allows the model to use the relationship between a passenger's age and their class.

---

### Title

The passenger's title was extracted from the `Name` column.

Examples include:

```text
Mr
Mrs
Miss
Master
Capt
Col
Major
```

Rare titles were grouped into broader categories such as:

```text
Military
Nobility
Ms
Mrs
```

The resulting `Title` feature was converted into numerical columns using one-hot encoding.

---

## Final Features

The final model uses the following features:

```text
Pclass
Sex
Age
SibSp
Parch
Fare
Embarked
FamilySize
IsAlone
Sex_Pclass
Child
TicketGroupSize
WomanOrChild
Age_Pclass
Title_*
```

The following original columns were removed before training:

```text
PassengerId
Name
Ticket
Cabin
```

`PassengerId` was removed because it is an identifier rather than a meaningful passenger characteristic.

`Name` was replaced by the extracted `Title` feature.

`Ticket` was replaced by the more useful `TicketGroupSize` feature.

`Cabin` was removed because of the large number of missing values.

---

## Model

The final model used for the main prediction pipeline is Logistic Regression.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)
```

Logistic Regression was selected because:

* It is suitable for binary classification.
* The Titanic target variable is binary.
* It is relatively simple and interpretable.
* It provides a strong baseline for this dataset.
* It is fast to train.
* Its behavior can be explained clearly during the project audit.

Other machine learning models could potentially achieve different results, but the project focuses on building a clear and understandable pipeline rather than using a more complex model without understanding its behavior.

---

## Train and Validation Split

The training data was divided into two parts:

```python
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

The split uses:

```text
80% training data
20% validation data
```

`stratify=y` was used to preserve approximately the same proportion of survivors and non-survivors in both datasets.

The `random_state=42` value makes the split reproducible.

---

## Model Evaluation

The model was evaluated using validation accuracy, confusion matrix, and classification report.

The local validation accuracy obtained with the final feature set was approximately:

```text
83.2%
```

This validation score is higher than the earlier baseline results.

The confusion matrix and classification report are also available in the notebook.

---

## Cross-Validation

A single train/validation split can give a result that depends on how the data happened to be divided.

To obtain a more reliable estimate, cross-validation was also considered.

Five-fold cross-validation was used to evaluate the model across multiple training and validation splits.

The average cross-validation accuracy was around:

```text
80%
```

The difference between the single validation score and the cross-validation average is normal because the validation set is only one particular split of the dataset.

Cross-validation is useful for checking whether the model's performance is reasonably stable instead of relying on one random split.

---

## Overfitting

Overfitting happens when a model learns the training data too specifically, including patterns that do not generalize well to unseen data.

For example, a model could obtain very high training accuracy while performing significantly worse on validation data.

To reduce the risk of overfitting, this project uses:

* A separate validation set
* Stratified train/validation splitting
* Cross-validation
* A relatively simple Logistic Regression model
* Feature engineering based on meaningful passenger information

The Kaggle score is also useful because Kaggle evaluates predictions on a separate hidden test set.

---

## Kaggle Result

The best submitted Kaggle score obtained during the project was approximately:

```text
0.792
```

This corresponds to approximately:

```text
79.2%
```

The project therefore achieved a Kaggle score above the required project threshold of 78.9%.

The local validation accuracy is higher than the Kaggle score. This difference is expected because the local validation set and the hidden Kaggle test set are different datasets.

---

## Why a Score Above 85% Can Be Suspicious

A very high Titanic leaderboard score is not automatically impossible, but it should be investigated carefully.

The original Titanic dataset is relatively small, and the test labels are hidden. Therefore, unusually high leaderboard performance can indicate problems such as:

* Data leakage
* Using information that would not be available when making a real prediction
* Accidentally including the target variable in the features
* Overly complex feature engineering
* Using information derived from the test labels
* Errors in preprocessing or validation

For this reason, a score above 85% should be examined carefully rather than automatically considered proof of a better model.

The objective of this project is to obtain a reliable result using a reproducible pipeline without using information from the hidden test labels.

---

## Reproducibility

The project can be reproduced using the provided scripts.

From the `scripts` directory:

```bash
python train.py
```

The training script:

1. Loads the datasets.
2. Preprocesses missing values.
3. Creates engineered features.
4. Encodes categorical features.
5. Splits the training data.
6. Trains the Logistic Regression model.
7. Evaluates the model.
8. Predicts the test dataset.
9. Creates the Kaggle submission file.

The generated submission contains:

```text
PassengerId
Survived
```

---

## Notebook

The notebook is located at:

```text
notebook/main.ipynb
```

The notebook contains:

* Dataset exploration
* Missing-value analysis
* Exploratory data analysis
* Survival analysis
* Feature engineering
* Model training
* Model evaluation
* Cross-validation
* Test prediction
* Submission generation

The notebook is also used to explain why the selected features were created and how they affected the model.

---

## Live Passenger Prediction

The trained model can be used to predict the survival of an individual passenger.

For example, after training the model, a passenger from the processed dataset can be selected and passed to the model:

```python
i = 0

passenger = X.iloc[[i]]
actual = y.iloc[i]

predicted = model.predict(passenger)[0]

print("Actual:", actual)
print("Predicted:", predicted)
```

The result can then be compared with the real `Survived` value from the training dataset.

This demonstrates that the model can be used to make a prediction for an individual passenger profile.

---

## Environment

The project uses Python and the following main libraries:

```text
Python
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
```

The exact package versions used for the project are documented in:

```text
requirements.txt
```

The project can be installed using:

```bash
pip install -r requirements.txt
```

A virtual environment is recommended:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Username

The Kaggle username associated with the project is anouar Boutamgharin and it stored in:

```text
username.txt
```

Replace the content of `username.txt` with the exact Kaggle username used for the final submission.

---

## Conclusion

This project demonstrates a complete machine learning workflow using the Titanic dataset.

The final pipeline includes data preprocessing, feature engineering, categorical encoding, Logistic Regression, validation, cross-validation, and Kaggle submission.

Feature engineering improved the local validation performance, particularly through features such as:

```text
FamilySize
IsAlone
Sex_Pclass
Child
TicketGroupSize
WomanOrChild
Age_Pclass
Title
```

The final Kaggle submission achieved approximately 79.2%, which is above the required project threshold.

The main objective of the project is to demonstrate understanding of the machine learning pipeline and the reasoning behind the preprocessing, feature engineering, model selection, and evaluation steps.
