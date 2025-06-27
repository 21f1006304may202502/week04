import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def load_data(path):
    return ps.read_csv(path)

def validate_data(df):
    assert not df.isnull().values.any(), "dataset contains null values"
    assert df.shape[1] == 5, "dataset must have 5 columns"

def train_and_evaluate(df):
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model = DecisionTreeClassifier()

    y_pred = model.predict(X_test)
    return accuracy_score(y_test,y_pred)
