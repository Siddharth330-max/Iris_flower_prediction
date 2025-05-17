from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC  
from sklearn.tree import DecisionTreeClassifier  
from sklearn.ensemble import RandomForestClassifier    
from sklearn.metrics import accuracy_score  

models = [
    LogisticRegression(),
    SVC(),
    DecisionTreeClassifier(),
    RandomForestClassifier()
]

def model_trainer(X_train_scaled , X_test_scaled,y_train , y_test):
    results = []
    for model in models:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        score = accuracy_score(y_test, y_pred)
        results.append((model.__class__.__name__, score))  # store name and score
    return results     





        
         
