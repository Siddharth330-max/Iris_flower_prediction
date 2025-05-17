from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  

def train_N_split(X , y , test_size= 0.25, random_state =42):
    X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = test_size, random_state = random_state)
    
    ##After doing train_test_split we perform standardisation
    scaler = StandardScaler()   

    X_train_scaled = scaler.fit_transform(X_train)  
    X_test_scaled = scaler.transform(X_test)   

    return X_train_scaled , X_test_scaled , y_train , y_test