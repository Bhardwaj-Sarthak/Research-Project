#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import ast

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


# %%
# SVR FOR SINGLE CHOICE QUESTIONS WITH FEATURES

X_sc = df_single_choice.iloc[:, 12:195].values #features
y_sc = df_single_choice.iloc[:, 4].values #item discrimination

X_sc_emb = embeddings_single_choice.iloc[:, 25:43].values

# FEATURES SCALING 

# Converting "Missing" in 0
X_sc = pd.DataFrame(X_sc)
X_sc.replace('Missing', 0, inplace=True)

# Splitting train and test set 
X_train, X_test, y_train, y_test = train_test_split(
    X_sc, y_sc, test_size=0.3, random_state=42)

# For embeddings
X_train_sc_emb, X_test_sc_emb, y_train_sc_emb, y_test_sc_emb = train_test_split(
    X_sc_emb, y_sc, test_size=0.3, random_state=42)

# Standardize Features
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

# For embeddings
X_train_sc_emb = sc_X.fit_transform(X_train_sc_emb)
X_test_sc_emb = sc_X.fit_transform(X_test_sc_emb)

# MODEL DEFINITION
# Define different kernel functions
kernel_functions = ['linear', 'rbf']

for i, kernel in enumerate(kernel_functions, 1):
    # Create SVM classifier with the specified kernel
    svr_classifier = SVR(kernel=kernel)
    svr_classifier_emb = SVR(kernel=kernel)

    # Train the classifier
    svr_classifier.fit(X_train, y_train)
    svr_classifier_emb.fit(X_train_sc_emb, y_train_sc_emb)

    # Evaluate accuracy
    y_pred = svr_classifier.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(kernel, "Mean Squared Error:", mse)

    y_pred_emb = svr_classifier_emb.predict(X_test_sc_emb)
    mse = mean_squared_error(y_test_sc_emb, y_pred_emb)
    print(kernel, "Mean Squared Error embeddings:", mse)

# linear Mean Squared Error: 1.0998008556188728
# linear Mean Squared Error embeddings: 0.6064687239143942
# rbf Mean Squared Error: 0.5247678068616017
# rbf Mean Squared Error embeddings: 0.7655035897493039

# %%
# SVR FOR MULTIPLE CHOICE QUESTIONS WITH FEATURES

X_mc = df_multiple_choice.iloc[:, 14:285].values #features
y_mc = df_multiple_choice.iloc[:, 4].values #item discrimination

# FEATURES SCALING 

# Converting "Missing" in 0
X_mc = pd.DataFrame(X_mc)
X_mc.replace('Missing', 0, inplace=True)

# Splitting train and test set 
X_train, X_test, y_train, y_test = train_test_split(
    X_mc, y_mc, test_size=0.3, random_state=42)

# Standardize Features
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# MODEL DEFINITION

for i, kernel in enumerate(kernel_functions, 1):
    # Create SVM classifier with the specified kernel
    svr_classifier = SVR(kernel=kernel)

    # Train the classifier
    svr_classifier.fit(X_train, y_train)

    # Evaluate accuracy
    y_pred = svr_classifier.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(kernel, "Mean Squared Error:", mse)

#linear Mean Squared Error: 1.5314999239561826
#rbf Mean Squared Error: 0.8020825552352708

# %%
# SVR FOR MULTIPLE CHOICE QUESTIONS WITH EMBEDDINGS

X_mc_e = embeddings_multiple_choice.iloc[:, 31:55].values  #features
y_mc_e = embeddings_multiple_choice.iloc[:, 4].values #item difficulty

# Converting "Missing" in 0
X_mc_e = pd.DataFrame(X_mc_e)
X_mc_e.replace('Missing', 0, inplace=True)

# Splitting train and test set 
X_train, X_test, y_train, y_test = train_test_split(
    X_mc_e, y_mc_e, test_size=0.3, random_state=42)


# Standardize Features
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# MODEL DEFINITION
kernel_functions = ['linear', 'rbf']
for i, kernel in enumerate(kernel_functions, 1):
    # Create SVM classifier with the specified kernel
    svr_classifier = SVR(kernel=kernel)

    # Train the classifier
    svr_classifier.fit(X_train, y_train)

    # Evaluate accuracy
    y_pred = svr_classifier.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(kernel, "Mean Squared Error:", mse)

#linear Mean Squared Error: 2.3256682423362594
#rbf Mean Squared Error: 1.0898280382980992

# %%
