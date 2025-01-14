#Neural Networks Architetture for Item Difficulty Prediction

#%%
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.regularizers import l2
# %%
# Whole dataset with features
data = pd.read_csv('C:/Users/Utente/Documents/Research project/Data&Code/Code_Research_Project/final_results.csv', encoding='ISO-8859-1')

# Single choice questions with features 
df_single_choice = pd.read_csv('C:/Users/Utente/Documents/Research project/Data&Code/Code_Research_Project/final_single_choice.csv', encoding='ISO-8859-1')
df_single_choice = df_single_choice[~(df_single_choice.eq("Missing").all(axis=1))]

# Multiple choice questions with features 
df_multiple_choice = pd.read_csv('C:/Users/Utente/Documents/Research project/Data&Code/Code_Research_Project/final_multiple_choice.csv', encoding='ISO-8859-1')
df_multiple_choice = df_multiple_choice[~(df_multiple_choice.eq("Missing").all(axis=1))]

#%%
#SINGLE CHOICE DATASET WITH FEATURES
X_sc = df_single_choice.iloc[:, 12:195].values #features
y_sc = df_single_choice.iloc[:, 4].values #item difficulty

# Converting "Missing" in 0
X_sc = pd.DataFrame(X_sc)
X_sc.replace('Missing', 0, inplace=True)

# Splitting train and test set 
X_train, X_test, y_train, y_test = train_test_split(
    X_sc, y_sc, test_size=0.3, random_state=42)

# Standardize Features
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) #use statistics calculated for the training set

# Set the number of neurons/nodes for each layer:
model = Sequential([
    Dense(182, input_shape=(182,), activation='relu', kernel_regularizer=l2(0.01)), # first layer
    Dense(64, activation='relu',kernel_regularizer=l2(0.01)), # second layer
    Dense(32, activation='relu',kernel_regularizer=l2(0.01)), 
    Dense(1) # output layer, one neuron as it is the output node 
])
# Compile the model and calculate its accuracy:
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae']) 

# Train the model
model.fit(X_train, y_train, epochs=500, batch_size=10)

# Evaluating the model on the test data
loss, mae = model.evaluate(X_test, y_test)
print('Test model loss:', loss) #0.828
print('Test model mae:', mae) #0.73
# Compare MAE to the range for y, which is 4.016

# Baseline model, that predicts the mean of the target parameter. Used to compare with more complicated model
baseline_mae = np.mean(np.abs(y_test - np.mean(y_train)))
print("Baseline MAE:", baseline_mae) #0.74


#%%
#EMBEDDINGS SINGLE CHOICE
import embeddings_gen as emb
#%%
import pandas as pd
df=pd.read_csv('C:/Users/Utente/Documents/Research project/Data&Code/Code_Research_Project/final_results.csv', encoding='ISO-8859-1')

'''
response_type: default 'single choice', or 'multiple response'
'''
df= emb.preprocess_text(df,clean_text=True,response_type='single choice')

'''
model_name: default 'paraphrase-multilingual-MiniLM-L12-v2',
                    'xlm-r-distilroberta-base-paraphrase-v1',
                    'distiluse-base-multilingual-cased-v1',
                    'T-Systems-onsite/cross-en-de-roberta-sentence-transformer',
            form the SentenceTransformer library
            or
            fine-tuned transformer models for Swiss-German from the Hugging Face Transformers library:
                    'ZurichNLP/unsup-simcse-xlm-roberta-base',
                    'jgrosjean-mathesis/swissbert-for-sentence-embeddings'
            from the Hugging Face Transformers library
response_type: default 'single choice', or 'multiple response'
'''
final_embeddings= emb.generate_embeddings(df,model_name='xlm-r-distilroberta-base-paraphrase-v1',response_type='single choice')

# %%
# NN MODEL USING EMBEDDINGS FOR SINGLE CHOICE

# Splitting train and test set 
X_emb = final_embeddings
X_train_emb, X_test_emb, y_train, y_test = train_test_split(
    X_emb, y_sc, test_size=0.3, random_state=42)

# Standardize Features
sc_X = StandardScaler()
X_train_emb = sc_X.fit_transform(X_train_emb)
X_test_emb = sc_X.transform(X_test_emb) #use statistics calculated for the training set

# Set the number of neurons/nodes for each layer:
model = Sequential([
    Dense(182, input_shape=(7680,), activation='relu', kernel_regularizer=l2(0.01)), # first layer
    Dense(64, activation='relu',kernel_regularizer=l2(0.01)), # second layer
    Dense(32, activation='relu',kernel_regularizer=l2(0.01)), 
    Dense(1) # output layer, one neuron as it is the output node 
])
# Compile the model and calculate its accuracy:
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae']) 
# Train the model
model.fit(X_train_emb, y_train, epochs=500, batch_size=10)

# Evaluating the model on the test data
loss, mae = model.evaluate(X_test_emb, y_test)
print('Test model loss:', loss) #0.407
print('Test model mae:', mae) #0.5087
# Compare MAE to the range for y, which is 4.016

# Baseline model, that predicts the mean of the target parameter. Used to compare with more complicated model
baseline_mae = np.mean(np.abs(y_test - np.mean(y_train)))
print("Baseline MAE:", baseline_mae) #0.744

# %%
#MULTIPLE CHOICE DATASET WITH FEATURES
X_mc = df_multiple_choice.iloc[:, 14:285].values #features
y_mc = df_multiple_choice.iloc[:, 4].values #item discrimination
#%%
# Converting "Missing" in 0
X_mc = pd.DataFrame(X_mc)
X_mc.replace('Missing', 0, inplace=True)

# Splitting train and test set 
X_train, X_test, y_train, y_test = train_test_split(
    X_mc, y_mc, test_size=0.3, random_state=42)

# Standardize Features
mc_X = StandardScaler()
X_train = mc_X.fit_transform(X_train)
X_test = mc_X.transform(X_test) #use statistics calculated for the training set

# Set the number of neurons/nodes for each layer:
model = Sequential([
    Dense(270, input_shape=(270,), activation='relu', kernel_regularizer=l2(0.01)), # first layer
    Dense(64, activation='relu',kernel_regularizer=l2(0.01)), # second layer
    Dense(32, activation='relu',kernel_regularizer=l2(0.01)), 
    Dense(1) # output layer, one neuron as it is the output node 
])
# Compile the model and calculate its accuracy:
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae']) 

# Train the model
model.fit(X_train, y_train, epochs=500, batch_size=10)

# Evaluating the model on the test data
loss, mae = model.evaluate(X_test, y_test)
print('Test model loss:', loss) #1.14
print('Test model mae:', mae) #0.88
# Compare MAE to the range for y, which is 4.016

# Baseline model, that predicts the mean of the target parameter. Used to compare with more complicated model
baseline_mae = np.mean(np.abs(y_test - np.mean(y_train)))
print("Baseline MAE:", baseline_mae) #0.796

# %%
# NN MODEL USING EMBEDDINGS FOR MULTIPLE CHOICE
df=pd.read_csv('C:/Users/Utente/Documents/Research project/Data&Code/Code_Research_Project/final_results.csv', encoding='ISO-8859-1')
df= emb.preprocess_text(df,clean_text=True, response_type='multiple response')
final_embeddings_mc= emb.generate_embeddings(df,model_name='xlm-r-distilroberta-base-paraphrase-v1',response_type='multiple response')

# Splitting train and test set 
X_emb_mc = final_embeddings_mc
X_train_emb, X_test_emb, y_train, y_test = train_test_split(
    X_emb_mc, y_mc, test_size=0.3, random_state=42)

# Standardize Features
sc_X = StandardScaler()
X_train_emb = sc_X.fit_transform(X_train_emb)
X_test_emb = sc_X.transform(X_test_emb) #use statistics calculated for the training set
#%%
# Set the number of neurons/nodes for each layer:
model = Sequential([
    Dense(182, input_shape=(7680,), activation='relu', kernel_regularizer=l2(0.01)), # first layer
    Dense(64, activation='relu',kernel_regularizer=l2(0.01)), # second layer
    Dense(32, activation='relu',kernel_regularizer=l2(0.01)), 
    Dense(1) # output layer, one neuron as it is the output node 
])
# Compile the model and calculate its accuracy:
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae']) 
# Train the model
model.fit(X_train_emb, y_train, epochs=500, batch_size=10)

# Evaluating the model on the test data
loss, mae = model.evaluate(X_test_emb, y_test)
print('Test model loss:', loss) #2.01
print('Test model mae:', mae) #0.54
# Compare MAE to the range for y, which is 4.016

# Baseline model, that predicts the mean of the target parameter. Used to compare with more complicated model
baseline_mae = np.mean(np.abs(y_test - np.mean(y_train)))
print("Baseline MAE:", baseline_mae) #0.796
# %%
