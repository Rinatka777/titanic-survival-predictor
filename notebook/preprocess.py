import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# Load data
df = pd.read_csv('data/train.csv')  # You can adjust this path if needed

def preprocess_data(df):
    dfcopy = df.copy(deep=True)

    # Fill missing values
    dfcopy['Age'].fillna(dfcopy['Age'].median(), inplace=True)
    dfcopy['Embarked'].fillna(dfcopy['Embarked'].mode()[0], inplace=True)

    # Drop irrelevant columns
    dfcopy.drop(['Name', 'PassengerId', 'Cabin', 'Ticket'], axis=1, inplace=True)

    # Encode categorical variables
    dfcopy['Sex'] = dfcopy['Sex'].map({'male': 0, 'female': 1})
    dfcopy = pd.get_dummies(dfcopy, columns=['Embarked'], drop_first=True)

    return dfcopy

# Preprocess
df_clean = preprocess_data(df)

# Split into features and target
X = df_clean.drop('Survived', axis=1)
y = df_clean['Survived']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_pred))
