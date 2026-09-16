import pandas as pd
import numpy as np





def extract_title(df):
    titles = []

    for name in df["Name"]:
        if ", " in name:
            last_name, name_part = name.split(", ", 1)
        else:
            name_part = name

        if ". " in name_part:
            title, first_name = name_part.split(". ", 1)
        else:
            title = ""

        titles.append(title)

    titles = np.array(titles)

    # Military titles
    indexes = np.where(
        np.isin(titles, ("Capt", "Col", "Major"))
    )
    titles[indexes] = "Military"

    # Nobility titles
    indexes = np.where(
        np.isin(titles, ("Don", "Dona", "Jonkheer", "Lady", "Sir", "the Countess"))
    )
    titles[indexes] = "Nobility"

    # Female titles
    indexes = np.where(
        np.isin(titles, ("Miss", "Mlle", "Ms"))
    )
    titles[indexes] = "Ms"

    indexes = np.where(
        np.isin(titles, ("Mrs", "Mme"))
    )
    titles[indexes] = "Mrs"

    df["Title"] = titles

    return df



def creat_features(train,test):


    train["FamilySize"] =  train["SibSp"] + train["Parch"] + 1
    test["FamilySize"] = test["SibSp"] + test["Parch"] + 1

    train["IsAlone"] = (train["FamilySize"] == 1).astype(int)
    test["IsAlone"] = (test["FamilySize"] == 1).astype(int)

    train["Sex"] = train["Sex"].map({
    "male": 0,
    "female": 1
    })
    test["Sex"] = test["Sex"].map({
    "male": 0,
    "female": 1
    })


    extract_title(train)
    extract_title(test)

    train["Embarked"] = train["Embarked"].map({
    "S": 0,
    "C": 1,
    "Q": 2
    })

    test["Embarked"] = test["Embarked"].map({
    "S": 0,
    "C": 1,
    "Q": 2
    })

    train["Sex_Pclass"] = train["Sex"] * train["Pclass"] 
    test["Sex_Pclass"] = test["Sex"] * test["Pclass"]

    train["Child"] = (train["Age"] < 16).astype(int)
    test["Child"] = (test["Age"] < 16).astype(int)

    ticket_counts = pd.concat([
    train["Ticket"],
    test["Ticket"]
    ]).value_counts()

    train["TicketGroupSize"] = train["Ticket"].map(ticket_counts)
    test["TicketGroupSize"] = test["Ticket"].map(ticket_counts)

    train["WomanOrChild"] = (
    (train["Sex"] == 1) |
    (train["Age"] < 16)
    ).astype(int)
    test["WomanOrChild"] = (
    (test["Sex"] == 1) |
    (test["Age"] < 16)
    ).astype(int)

    train["Age_Pclass"] = train["Age"] * train["Pclass"]
    test["Age_Pclass"] = test["Age"] * test["Pclass"]



    return train ,test



    







    
    



