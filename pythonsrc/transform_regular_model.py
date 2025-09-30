import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os import makedirs, path
#Load the matlab file - Mia
training_data=scio.loadmat("./proj1files/trainingdata.mat")
#Weird Dict, one key being "trainingdata" and the value is an np.ndarray
#Extract that out of that
training_data=training_data["trainingdata"]
#print(type(training_data))
#X data
X_train=training_data[:,0:78]
print(np.shape(X_train)) 
#violent crimes.
Y_violent=training_data[:,79]
print(np.shape(Y_violent))
#non violent crime
Y_non_violent=training_data[:,80]

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
def normalize_feature(x, y):
    mean = np.mean(y,axis = 0,keepdims=True)
    std = np.std(y,axis = 0,keepdims=True)
    return (x - mean) / (std + 1e-8)

normalize_features = normalize_feature(X_train,X_train)
# print(np.shape(normalize_features))
# print(normalize_features.mean(axis = 0))
# print(normalize_features.var(axis = 0))
covar= np.cov(normalize_features,rowvar=False)

def covar_heatmap(X):
    plt.figure(figsize=(10, 8))
    cov_matrix = np.cov(X, rowvar=False)
    plt.imshow(cov_matrix, cmap='coolwarm', aspect='auto')
    plt.colorbar(label='Covariance')
    plt.title('Covariance Heatmap')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Index')
    plt.tight_layout()
    plt.show()

#covar_heatmap(covar)
def correlation_heatmap(X):
    plt.figure(figsize=(10, 8))
    corr_matrix = np.corrcoef(X, rowvar=False)
    plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation Coefficient')
    plt.title('Correlation Heatmap')
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Index')
    plt.tight_layout()
    plt.show()
#correlation_heatmap(np.corrcoef(normalize_features,rowvar=False))

#Implement Dropout for highly correlated features
def dropout_highly_correlated_features(X, threshold=0.9):
    corr_matrix = np.corrcoef(X, rowvar=False)
    to_drop = set()
    num_features = corr_matrix.shape[0]

    for i in range(num_features):
        for j in range(i + 1, num_features):
            if abs(corr_matrix[i, j]) > threshold:
                to_drop.add(j)

    keep_indices = [i for i in range(num_features) if i not in to_drop]
    return X[:, keep_indices], keep_indices
X_dropped_normalized, kept_indices = dropout_highly_correlated_features(normalize_features, threshold=0.8)
#Scatter plot for X_dropped_normalized for violent and non-violent
def scatter_plots(X, y_violent, y_non_violent, kept_indices):
    output_dir = "./media/normalized"
    if not path.exists(output_dir):
        makedirs(output_dir)

    for idx, feature_idx in enumerate(kept_indices):
        plt.figure(figsize=(12, 5))

        # Plot for violent crimes
        plt.subplot(1, 2, 1)
        sns.scatterplot(x=X[:, idx], y=y_violent, color='red', s=3)
        plt.title(f'Feature {feature_idx} vs Violent Crimes')
        plt.xlabel(f'Feature {feature_idx}')
        plt.ylabel('Y_violent')

        # Plot for non-violent crimes
        plt.subplot(1, 2, 2)
        sns.scatterplot(x=X[:, idx], y=y_non_violent, color='blue', s=3)
        plt.title(f'Feature {feature_idx} vs Non-Violent Crimes')
        plt.xlabel(f'Feature {feature_idx}')
        plt.ylabel('Y_non_violent')

        plt.tight_layout()
        plt.savefig(path.join(output_dir, f'feature_{feature_idx}_scatter.png'))
        plt.close()
#scatter_plots(X_dropped_normalized, Y_violent, Y_non_violent, kept_indices)

#Compute Kurtosis
def compute_kurtosis(X):
    n = X.shape[0]
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    kurtosis = np.sum(((X - mean) / (std + 1e-8))**4, axis=0) / n -3
    return kurtosis
kurtosis_vector = compute_kurtosis(X_dropped_normalized)
print("Kurtosis of each feature after dropping highly correlated ones:")
# for i in range(len(kurtosis_vector)):
#     print(f"Feature {kept_indices[i]}: Kurtosis = {kurtosis_vector[i]}")
#drop features with high kurtosis
def drop_high_kurtosis_features(X, kurtosis_vector, threshold=10):
    to_drop = [i for i, k in enumerate(kurtosis_vector) if abs(k) > threshold]
    keep_indices = [i for i in range(X.shape[1]) if i not in to_drop]
    return X[:, keep_indices], keep_indices
X_final, final_kept_indices = drop_high_kurtosis_features(X_dropped_normalized, kurtosis_vector, threshold=10)
print(f"Final shape after dropping high kurtosis features: {X_final.shape}")

#naiive regression on X_final
X_final = np.insert(X_final, 0, np.ones((X_final.shape[0],)), axis=1) #add intercept
#violent
beta_hat_v_naive,*_=np.linalg.lstsq(X_final, Y_violent, rcond=None)
Y_violent_pred_naive=  X_final @ beta_hat_v_naive
mape_v_naive = mape(Y_violent, Y_violent_pred_naive)
#log fit
Y_violent_log_naive = np.log(np.clip(Y_violent, 1e-5, None))
beta_hat_v_log_naive,*_=np.linalg.lstsq(X_final, Y_violent_log_naive, rcond=None)
Y_violent_exp_pred_naive=  np.exp(X_final @ beta_hat_v_log_naive)
mape_v_exp_naive = mape(Y_violent, Y_violent_exp_pred_naive)
print(f"Violent Crime MAPE (direct) naive: {mape_v_naive:.2f}")
print(f"Violent Crime MAPE (log fit on y) naive: {mape_v_exp_naive:.2f}")
#non-violent
beta_hat_nv_naive,*_=np.linalg.lstsq(X_final, Y_non_violent, rcond=None)
Y_non_violent_pred_naive=  X_final @ beta_hat_nv_naive
mape_nv_naive = mape(Y_non_violent, Y_non_violent_pred_naive)
#log fit
Y_non_violent_log_naive = np.log(np.clip(Y_non_violent, 1e-5, None))
beta_hat_nv_log_naive,*_=np.linalg.lstsq(X_final, Y_non_violent_log_naive, rcond=None)
Y_non_violent_exp_pred_naive=  np.exp(X_final @ beta_hat_nv_log_naive)
mape_nv_exp_naive = mape(Y_non_violent, Y_non_violent_exp_pred_naive)
print(f"Non Violent Crime MAPE (direct) naive: {mape_nv_naive:.2f}")
print(f"Non Violent Crime MAPE (log fit on y) naive: {mape_nv_exp_naive:.2f}")


#test data
#load test data
testing_data=scio.loadmat("./proj1files/testingdata.mat")
testing_data=testing_data["testingdata"]
X_test=testing_data[:,0:78]
Y_violent_test=testing_data[:,79]
Y_non_violent_test=testing_data[:,80]

#Normalize X_test, make sure it's the same normalization as training.
X_test = normalize_feature(X_test,X_train)

#drop same features as training
X_test_dropped = X_test[:, kept_indices]
X_test_final = X_test_dropped[:, final_kept_indices]
X_test_final = np.insert(X_test_final, 0, np.ones((X_test_final.shape[0],)), axis=1) #add intercept

#Violent
Y_violent_test_pred_naive=  X_test_final @ beta_hat_v_naive
mape_v_test_naive = mape(Y_violent_test, Y_violent_test_pred_naive)
Y_violent_test_exp_pred_naive=  np.exp(X_test_final @ beta_hat_v_log_naive)
mape_v_test_exp_naive = mape(Y_violent_test, Y_violent_test_exp_pred_naive)
print(f"Violent Crime Test MAPE (direct) naive: {mape_v_test_naive:.2f}")
print(f"Violent Crime Test MAPE (log fit on y) naive: {mape_v_test_exp_naive:.2f}")

#non-violent
Y_non_violent_test_pred_naive=  X_test_final @ beta_hat_nv_naive
mape_nv_test_naive = mape(Y_non_violent_test, Y_non_violent_test_pred_naive)
Y_non_violent_test_exp_pred_naive=  np.exp(X_test_final @ beta_hat_nv_log_naive)
mape_nv_test_exp_naive = mape(Y_non_violent_test, Y_non_violent_test_exp_pred_naive)
print(f"Non Violent Crime Test MAPE (direct) naive: {mape_nv_test_naive:.2f}")
print(f"Non Violent Crime Test MAPE (log fit on y) naive: {mape_nv_test_exp_naive:.2f}")

#We now implement L2 regulization
