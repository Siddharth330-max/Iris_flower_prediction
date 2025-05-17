import pandas as pd  
from sklearn.datasets import load_iris  

def load_data():
    iris = load_iris()  
    X  = pd.DataFrame(iris.data , columns = iris.feature_names) 
    y = pd.Series(iris.target , name = 'Species')  
    return X , y, iris.target_names   