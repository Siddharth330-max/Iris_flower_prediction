from data_loader import load_data  
from preprocessor import train_N_split  
from model_train import model_trainer   

X , y , target_names = load_data()   
X_train_scaled , X_test_scaled , y_train , y_test = train_N_split(X , y)   
results = model_trainer(X_train_scaled , X_test_scaled , y_train , y_test)    

for model_name, score in results:
    print(f"{model_name}: {score:.2f}")
