#%%
# RANDOM FOREST MODEL

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

# Whole dataset with features
data = pd.read_csv('C:/Users/Utente/Documents/Research project/Data&Code/Code_Research_Project/final_results.csv', encoding='ISO-8859-1')

# Single choice questions with features 
df_single_choice = pd.read_csv('C:/Users/Utente/Documents/Research project/Data&Code/Code_Research_Project/final_single_choice.csv', encoding='ISO-8859-1')
df_single_choice = df_single_choice[~(df_single_choice.eq("Missing").all(axis=1))]

# Multiple choice questions with features 
df_multiple_choice = pd.read_csv('C:/Users/Utente/Documents/Research project/Data&Code/Code_Research_Project/final_multiple_choice.csv', encoding='ISO-8859-1')
df_multiple_choice = df_multiple_choice[~(df_multiple_choice.eq("Missing").all(axis=1))]

# Multiple choice questions embeddings
embeddings_multiple_choice = pd.read_csv ("C:/Users/Utente/Documents/Research project/Data&Code/Embeddings/final embeddings/multi_df/clean/jgrosjean-mathesis_sentence-swissbert_multi_df_final_clean.csv")

# Single choice questions embeddings
embeddings_single_choice = pd.read_csv ("C:/Users/Utente/Documents/Research project/Data&Code/Embeddings/final embeddings/single_df/cleaned/jgrosjean-mathesis_sentence-swissbert_final__single_df_clean.csv")

#%%
#SINGLE CHOICE DATASET WITH FEATURES
X_sc = df_single_choice.iloc[:, 12:195].values #features
y_sc = df_single_choice.iloc[:, 4].values #item difficulty

# SINGLE CHOICE DATASET WITH EMBEDDINGS (ONLY USING SIMILARITY AND NOT THE ACTUAL EMBEDDINGS DUE TO WRONG FORMATTING)
X_sc_emb = embeddings_single_choice.iloc[:, 25:43].values 

# Calculating the mean of the item difficulty to use as a naive baseline to compare with my model
y_mean = np.mean(y_sc) # -0.126

# Converting "Missing" in 0
X_sc = pd.DataFrame(X_sc)
X_sc.replace('Missing', 0, inplace=True)

# Splitting train and test set for the features dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_sc, y_sc, test_size=0.3, random_state=42)

# Splitting train and test set for the embeddings dataset
X_train_emb, X_test_emb, y_train_emb, y_test_emb = train_test_split(
    X_sc_emb, y_sc, test_size=0.3, random_state=42)

# Creating a vector of all mean
y_mean_vec = np.full(len(y_test), y_mean)

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42) #features
rf_embeddings = RandomForestRegressor(n_estimators=1000, random_state=42) #embeddings
# Train the model on training data for features dataset
rf.fit(X_train, y_train)

# Train the model on training data for embeddings dataset
rf_embeddings.fit(X_train_emb, y_train_emb)

# Use the forest's predict method on the features test data
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error for features:", mse)  #0.59730

# Use the forest's predict method on the features test data
y_pred_emb = rf_embeddings.predict(X_test_emb)
mse_emb = mean_squared_error(y_test_emb, y_pred_emb)
print("Mean Squared Error for embeddings:", mse_emb)  #0.65

# Use y_mean for comparison
mse_baseline = mean_squared_error(y_test, y_mean_vec)
print("Mean Squared Error for baseline model:", mse_baseline) # 0.749

# Calculate mean absolute percentage error (MAPE)
errors = abs(y_pred - y_test)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.') #10.57% BAD!!!!

# %%
# TUNE HYPERPARAMETERS TO GET A BETTER MODEL
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['log2', None, 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [20, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf_hp = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf_hp, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

rf_random_emb = RandomizedSearchCV(estimator = rf_hp, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model for features 
rf_random.fit(X_train, y_train)

# Fit the random search model for embeddings 
rf_random_emb.fit(X_train_emb, y_train_emb)

# Use the forest's predict method on the features test data
y_pred = rf_random.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error for features with grid search:", mse) #0.4887

# Use the forest's predict method on the emebddings test data
y_pred_emb = rf_random_emb.predict(X_test_emb)
mse = mean_squared_error(y_test_emb, y_pred_emb)
print("Mean Squared Error for embeddings with grid search:", mse) #0.66

# Calculate mean absolute percentage error (MAPE)
errors = abs(y_pred - y_test)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.') #20.19% 

#%%
# The accuracy is still too low, try to reduce the number of features to use as a lot of them seem to do not have impact
# Let's try with PCA
# Standardize Features
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_sc_stan = sc_X.fit_transform(X_sc)

from sklearn.decomposition import PCA
pca = PCA(n_components=29)
principalComponents = pca.fit_transform(X_sc_stan)

# New DataFrame with principal components
pca_X = pd.DataFrame(
    data=principalComponents, 
    columns=[f'PC{i+1}' for i in range(principalComponents.shape[1])]  
)
X_train_pca, X_test_pca, y_train, y_test = train_test_split(
    pca_X, y_sc, test_size=0.3, random_state=42)

# Train the model on training data
rf.fit(X_train_pca, y_train)

# Use the forest's predict method on the test data
y_pred = rf.predict(X_test_pca)
mse = mean_squared_error(y_test, y_pred) #0.581
print("Mean Squared Error:", mse)

# Calculate mean absolute percentage error (MAPE)
errors = abs(y_pred - y_test)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.') #48.63%

#%%
rf_random = RandomizedSearchCV(estimator = rf_hp, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train_pca, y_train)
# Use the forest's predict method on the test data
y_pred = rf_random.predict(X_test_pca)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse) #0.577

# Calculate mean absolute percentage error (MAPE)
errors = abs(y_pred - y_test)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.') #78.99% 

# %%
# MULTIPLE CHOICE DATASET WITH FEATURES
X_mc = df_multiple_choice.iloc[:, 14:285].values #features
y_mc = df_multiple_choice.iloc[:, 4].values #item difficulty

X_mc_emb = embeddings_multiple_choice.iloc[:, 31:55].values 

# Converting "Missing" in 0
X_mc = pd.DataFrame(X_mc)
X_mc.replace('Missing', 0, inplace=True)

# Splitting train and test set 
X_train, X_test, y_train, y_test = train_test_split(
    X_mc, y_mc, test_size=0.3, random_state=42)

#embeddings dataset
X_train_emb, X_test_emb, y_train_emb, y_test_emb = train_test_split(
    X_mc_emb, y_mc, test_size=0.3, random_state=42)

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf_emb = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(X_train, y_train)
rf_emb.fit(X_train_emb, y_train_emb)

# Use the forest's predict method on the test data
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse) #0.996

# Use the forest's predict method on the test data
y_pred_emb = rf_emb.predict(X_test_emb)
mse_emb = mean_squared_error(y_test_emb, y_pred_emb)
print("Mean Squared Error embeddings:", mse_emb) #1.09

#%%
# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(X_mc, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# List of features sorted from most to least important
sorted_importances = [importance[1] for importance in feature_importances]
sorted_features = [importance[0] for importance in feature_importances]
# Cumulative importances
cumulative_importances = np.cumsum(sorted_importances)
# Find number of features for cumulative importance of 95%
# Add 1 because Python is zero-indexed
#print('Number of features for 75 importance:', np.where(cumulative_importances > 0.75)[0][0] + 1) #40
# %%
# TUNE HYPERPARAMETERS TO TRY TO GET A BETTER MODEL
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['log2', None, 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf_hp = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf_hp, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random_emb_mc = RandomizedSearchCV(estimator = rf_hp, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model
rf_random.fit(X_train, y_train)
rf_random_emb_mc.fit(X_train_emb, y_train_emb)

# Use the forest's predict method on the test data
y_pred = rf_random.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse) #0.94

y_pred_emb = rf_random_emb_mc.predict(X_test_emb)
mse = mean_squared_error(y_test_emb, y_pred_emb)
print("Mean Squared Error for embeddibgs:", mse) #1.126

errors = abs(y_pred - y_test)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.') #66.26% 

# %%
# Use PCA for multiple choice
X_mc_stan = sc_X.fit_transform(X_mc)
pca = PCA(n_components=32) #n_components must be between 0 and min(n_samples, n_features)=32, I would have put 40 as they explained 75%
principalComponents = pca.fit_transform(X_mc_stan)

# New DataFrame with principal components
pca_X_mc = pd.DataFrame(
    data=principalComponents, 
    columns=[f'PC{i+1}' for i in range(principalComponents.shape[1])]  
)
X_train_pca, X_test_pca, y_train, y_test = train_test_split(
    pca_X_mc, y_mc, test_size=0.3, random_state=42)

# Train the model on training data
rf_random.fit(X_train_pca, y_train)

# Use the forest's predict method on the test data
y_pred = rf_random.predict(X_test_pca)
mse = mean_squared_error(y_test, y_pred) #1.22
print("Mean Squared Error:", mse)

# Calculate mean absolute percentage error (MAPE)
errors = abs(y_pred - y_test)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.') #99.56%





