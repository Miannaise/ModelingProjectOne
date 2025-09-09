import scipy.io as scio
import numpy as np
#Load the matlab file - Mia
training_data=scio.loadmat('../proj1files/trainingdata.mat')
#Weird Dict, one key being "trainingdata" and the value is an np.ndarray
#Extract that out of that
training_data=training_data["trainingdata"]
print(type(training_data))
#X data
X_train=training_data[:,0:78]
print(np.shape(X_train))
#violent crimes.
Y_violent=training_data[:,79]
print(np.shape(Y_violent))
#non violent crime
Y_non_violent=training_data[:,80]
#Insert Ones into X_train for simple linear regression
X_train = np.insert(X_train, 0, np.ones((X_train.shape[0],)), axis=1)
print(np.shape(X_train))
print(X_train)


#Least Squares Starts Here
betaviolent, *_ = np.linalg.lstsq(X_train, Y_violent, rcond=None)      # shape (80,)
betanonviolent, *_ = np.linalg.lstsq(X_train, Y_non_violent, rcond=None) # shape (80,)
print(len(betaviolent))
print(len(betanonviolent))

#Evaluate model performance
#Recall MAPE is mean absolute percentage error 
# Violent crime MAPE
y_pred_violent = X_train @ betaviolent
MAPE_violent = 100 * np.sum(np.abs(y_pred_violent - Y_violent) / Y_violent) / len(Y_violent)
print(f"MAPE for the violent crime rate training set is {MAPE_violent:.1f}")

# Nonviolent crime MAPE
y_pred_nonviolent = X_train @ betanonviolent
MAPE_nonviolent = 100 * np.sum(np.abs(y_pred_nonviolent - Y_non_violent) / Y_non_violent) / len(Y_non_violent)
print(f"MAPE for the nonviolent crime rate training set is {MAPE_nonviolent:.1f}")