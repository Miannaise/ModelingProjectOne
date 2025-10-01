import scipy.io as scio
import numpy as np
#Load the matlab file - Mia
training_data=scio.loadmat("./proj1files/trainingdata.mat")
#Weird Dict, one key being "trainingdata" and the value is an np.ndarray
#Extract that out of that
training_data=training_data["trainingdata"]
#print(type(training_data))
#X data
X_train=training_data[:,0:78]
#print(np.shape(X_train))
#violent crimes.
Y_violent=training_data[:,79]
#print(np.shape(Y_violent))
#non violent crime
Y_non_violent=training_data[:,80]
