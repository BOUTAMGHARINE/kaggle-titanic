import pandas as pd
from data_preprocessing import load_data
from data_preprocessing import preprocess_data
from feature_engineering import creat_features
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix ,classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV






train_path = "../data/train.csv"
test_path = "../data/test.csv"


train ,test = load_data(train_path,test_path)

train ,test = preprocess_data(train,test)

train ,test =  creat_features(train ,test)






X = train.drop("Survived", axis=1)
y = train["Survived"]

X = X.drop(["PassengerId", "Name", "Ticket"], axis=1)

X_test = test.drop(["PassengerId", "Name", "Ticket"], axis=1)

# 1. Encode Title
X = pd.get_dummies(X, columns=["Title"], dtype=int)
X_test = pd.get_dummies(X_test, columns=["Title"], dtype=int)

# 2. Make sure X_test has exactly the same columns as X
X_test = X_test.reindex(columns=X.columns, fill_value=0)

# 3. Split train into training and validation sets

print("-----------------------------------------------------------------")

print(X.columns.tolist())
print(X_test.columns.tolist())


X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Features:")
print(X.columns.tolist())



model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)
#hsdgsgdjfj



y_pred = model.predict(X_val)
print("accurancy",accuracy_score(y_val,y_pred))
test_predictions = model.predict(X_test)

print(y_pred)






#final training 


submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": test_predictions
})

submission.to_csv("../submission.csv", index=False)
print("\nSubmission created successfully.")










