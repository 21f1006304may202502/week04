import pandas as pd
from src.utils import train_and_evaluate

## check 1


def test_model_evaluation():
    df = pd.DataFrame({
        "sepal_length" : [5.1,4.9,6.3,5.8],
        "sepal_width" : [3.5,3.0,3.3,2.7],
        "petal_length" : [1.4,1.4,6.0,5.1],
        "petal_width" : [0.2,0.2, 2.5,1.9],
        "species" : ["setosa","setosa","virginica","virginica"]
        })
    accuracy = train_and_evaluate(df)
    assert 0 <= accuracy <= 1
