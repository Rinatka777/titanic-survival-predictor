import pandas as pd

def preprocess_data(df):
    df = pd.read_csv('train.csv')
    dfcopy = df.copy(deep= True)


    dfcopy['Age'] = dfcopy['Age'].fillna(dfcopy['Age'].median())
    dfcopy['Embarked'] = dfcopy['Embarked'].fillna(dfcopy['Embarked'].mode())
    dfcopy = df.drop("Name", "PassengerId", "Cabin")
    dfcopy["Sex"] = dfcopy["Sex"].map({"male":0, "female":1})
    dfcopy = pd.get_dummies(df, column = ["Embarked"])

    x = dfcopy.iloc[:,:]
    y = dfcopy.iloc[:,:]

    train,test = train_test_split(dfcopy, test_size=0.2)

    model = LogisticRegression()
    model.fit(x, y )