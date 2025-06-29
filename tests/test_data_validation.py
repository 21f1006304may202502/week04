import pytest
import pandas as pd
from src.utils import validate_data

def test_data_validation_pass():
    df = pd.DataFrame({
        "sepal_length" : [5.1,4.9],
        "sepal_width" : [3.5,3.0],
        "petal_length" : [1.4,1.4],
        "species" : ["setosa","setosa"]
        })
    validate_data(df)

def test_data_validation_fail():
    df = pd.DataFrame({
        "sepal_length" : [5.1,None],
        "sepal_width" : [3.5,3.0],
        "petal_length" : [0.2,0.2],
        "species" : ["setosa","setosa"]
        })
    with pytest.raises(AssertionError):
        validate_data(df)
