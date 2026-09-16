import pandas as pd




def load_data(train_path,test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    return train,test


def preprocess_data(train,test):
    #missing values
    #print(train.isnull().sum())
    #print(test.isnull().sum())
    age_median = train["Age"].median()
    train["Age"] = train["Age"].fillna(age_median)
    test["Age"] = test["Age"].fillna(age_median)

    embarked_mode = train["Embarked"].mode()[0]
    train["Embarked"] = train["Embarked"].fillna(embarked_mode)
    test["Embarked"] = test["Embarked"].fillna(embarked_mode)

    fare_median = train["Fare"].median()
    test["Fare"] = test["Fare"].fillna(fare_median)

    train.drop("Cabin",axis=1,inplace=True)
    test.drop("Cabin",axis=1,inplace=True)
    #print(train.isnull().sum())
    #print(test.isnull().sum())
    return train,test

    





