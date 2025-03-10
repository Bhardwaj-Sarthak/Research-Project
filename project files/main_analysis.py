# change the path to the directory where the project files are stored
# add the data file to the project files directory
# add the name in line 35 of the code
# select the response type
# rest you are good to go

#%%
import sys
sys.path.append(r"C:\Users\Sarthak\Downloads\research project\project files")
from analysis_functions import *
import embeddings_gen as emb
from cen import *
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import ParameterGrid
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import spacy
import functions as f
import feature_eng as fg
import warnings
warnings.filterwarnings("ignore")
import subprocess
try:
    nlp = spacy.load("de_core_news_sm")
except OSError:
    # Download the model if it's not available
    subprocess.run(["python", "-m", "spacy", "download", "de_core_news_sm"], check=True)
    nlp = spacy.load("de_core_news_sm")

#%%
response='single choice' #single choice / multiple response
df=pd.read_csv("Selection_Items_Tuebingen_1.csv", sep=';' ,encoding='ISO-8859-1') #add the name of the data file
df_text = df.fillna('') 
results = df_text.apply(lambda row: fg.process_row(row, df_text), axis=1)
results_df = pd.DataFrame(results.tolist())
df_t = pd.concat([df_text, results_df], axis=1)
df_t.columns = df_t.columns.astype(str)
df_t = df_t[df_t['Item Type'] == response]
df_t.fillna(0,inplace=True)
df_t.reset_index(drop=True, inplace=True)
df_emb= emb.preprocess_text(df,clean_text=True,response_type= response)
#df_t.to_csv("TEXT_FEAT.csv", index=False, encoding='ISO-8859-1')

#%%
print("Item Difficulty range:", 
	  f"min: {df['Item Difficulty'].min():.3f}, max: {df['Item Difficulty'].max():.3f}")
print("Item Discrimination range:", 
	  f"min: {df['Item Discrimination'].min():.3f}, max: {df['Item Discrimination'].max():.3f}")

#%%
final_embeddings_1= emb.generate_embeddings(df_emb,model_name='ZurichNLP/unsup-simcse-xlm-roberta-base',response_type=response)
final_embeddings_2= emb.generate_embeddings(df_emb,model_name='paraphrase-multilingual-MiniLM-L12-v2',response_type=response)
final_embeddings_3= emb.generate_embeddings(df_emb,model_name='xlm-r-distilroberta-base-paraphrase-v1',response_type=response)
final_embeddings_4= emb.generate_embeddings(df_emb,model_name='distiluse-base-multilingual-cased-v1',response_type=response)
final_embeddings_5= emb.generate_embeddings(df_emb,model_name='T-Systems-onsite/cross-en-de-roberta-sentence-transformer',response_type=response)
final_embeddings_6= emb.generate_embeddings(df_emb,model_name='jgrosjean-mathesis/swissbert-for-sentence-embeddings',response_type=response)

#%%
df_comb_1=pd.concat([df_t,pd.DataFrame(final_embeddings_1)],axis=1)
df_comb_2=pd.concat([df_t,pd.DataFrame(final_embeddings_2)],axis=1)
df_comb_3=pd.concat([df_t,pd.DataFrame(final_embeddings_3)],axis=1)
df_comb_4=pd.concat([df_t,pd.DataFrame(final_embeddings_4)],axis=1)
df_comb_5=pd.concat([df_t,pd.DataFrame(final_embeddings_5)],axis=1)
df_comb_6=pd.concat([df_t,pd.DataFrame(final_embeddings_6)],axis=1)

df_comb = [df_comb_1, df_comb_2, df_comb_3, df_comb_4, df_comb_5, df_comb_6]
for df in df_comb:
        df.drop(['InternCode','Title',
        'Content',
        'Question',
        'Correct Response',
        'Response Option 1',
        'Response Option 2',
        'Response Option 3',
        'Response Option 4',
        'Response Option 5',
        'Response Option 6',
        'Response Option 7',
        'Item Type',
        'Item Difficulty',
        'Item Discrimination'],axis=1,inplace=True)
        df.columns = df.columns.astype(str)
        df.fillna(0,inplace=True)
        df.reset_index(drop=True, inplace=True)
emb_comb=[pd.DataFrame(final_embeddings_1),pd.DataFrame(final_embeddings_2),pd.DataFrame(final_embeddings_3),pd.DataFrame(final_embeddings_4),pd.DataFrame(final_embeddings_5),pd.DataFrame(final_embeddings_6)]
#%%
y=df_t['Item Difficulty']
#%%
print('from here the results of the models will be printed in the followinf order \n',
    'ZurichNLP/unsup-simcse-xlm-roberta-base\n',
    f"_ _ "*10,
    "\n",
    f" _ _"*10,
    "\n",
    'paraphrase-multilingual-MiniLM-L12-v2 \n',
    f"_ _ "*10,
    "\n",
    f" _ _"*10,
    "\n",
    'xlm-r-distilroberta-base-paraphrase-v1 \n',
    f"_ _ "*10,
    "\n",
    f" _ _"*10,
    "\n",
    'distiluse-base-multilingual-cased-v1 \n',
    f"_ _ "*10,
    "\n",
    f" _ _"*10,
    "\n",
    'T-Systems-onsite/cross-en-de-roberta-sentence-transformer\n',
    f"_ _ "*10,
    "\n",
    f" _ _"*10,
    "\n",
    'jgrosjean-mathesis/swissbert-for-sentence-embeddings \n' )

print('Results of Lasso for text features and embeddings combined:')
print_lasso(df_comb, y)
print('Results of elastic net for text features and embeddings combined:')
print_elastic_net(df_comb, y)
print('Results of Lasso for text features:')
pritn_lasso_text(df_t,y)
print('Results of Lasso for text embeddings:')
print_lasso(emb_comb, y)
print('Results of elastic net for text embeddings:')
print_elastic_net(emb_comb, y)
print('Results of random forest for text embeddings:')
print_random_forest(emb_comb, y)
print('Results of support vector regression for text embeddings:')
print_SVR(emb_comb, y)
print('Results of support vector machine for text embeddings:')
print_SVM(emb_comb, y)
print('Results of random forest for classification for text embeddings:')
print_Random_Forest_Classification(emb_comb, y)



# stop here if response is multiple response
if response == 'multiple response':
    print('Response type is multiple response. CEN is not applicable due to less number of data points')
    sys.exit()

Y = torch.tensor(df_t[['Item Difficulty', 'Item Discrimination']].values, dtype=torch.float32)
print('Results of CEN for text embeddings:')
X1 = torch.tensor(final_embeddings_1[:, :768]).float()  #content
X2 = torch.tensor(final_embeddings_1[:, 768:1536]).float()  #Qustion
X3 = torch.tensor(final_embeddings_1[:, 1536:2304]).float() #correct response
X4 = torch.tensor(final_embeddings_1[:, 2304:3072]).float() #response 1
X5 = torch.tensor(final_embeddings_1[:, 3072:3840]).float() #response 2
X6 = torch.tensor(final_embeddings_1[:, 3840:4608]).float() #response 3
X7 = torch.tensor(final_embeddings_1[:, 4608:5376]).float() #response 4
train_cen(X1, X2, X3, X4, X5, X6, X7, Y)
print('ZurichNLP/unsup-simcse-xlm-roberta-base\n', (f"_ _ "*10))
#%%
X1 = final_embeddings_2[:, :384] #content
X1 = torch.tensor(X1).float()
X2 = final_embeddings_2[:, 384:768] #Qustion
X2 = torch.tensor(X2).float()
X3 = final_embeddings_2[:, 768:1152] #correct response
X3 = torch.tensor(X3).float()
X4 = final_embeddings_2[:, 1152:1536] #response 1
X4 = torch.tensor(X4).float()
X5 = final_embeddings_2[:, 1536:1920] #response 2
X5 = torch.tensor(X5).float()
X6 = final_embeddings_2[:, 1920:2304] #response 3
X6 = torch.tensor(X6).float()
X7 = final_embeddings_2[:, 2304:2688] #response 4
X7 = torch.tensor(X7).float()

train_cen(X1, X2, X3, X4, X5, X6, X7, Y)
print('\n paraphrase-multilingual-MiniLM-L12-v2 \n',
    f"_ _ "*10,)
X1 = torch.tensor(final_embeddings_3[:, :768]).float()  #content
X2 = torch.tensor(final_embeddings_3[:, 768:1536]).float()  #Qustion
X3 = torch.tensor(final_embeddings_3[:, 1536:2304]).float() #correct response
X4 = torch.tensor(final_embeddings_3[:, 2304:3072]).float() #response 1
X5 = torch.tensor(final_embeddings_3[:, 3072:3840]).float() #response 2
X6 = torch.tensor(final_embeddings_3[:, 3840:4608]).float() #response 3
X7 = torch.tensor(final_embeddings_3[:, 4608:5376]).float() #response 4

train_cen(X1, X2, X3, X4, X5, X6, X7, Y)
X1 = final_embeddings_4[:, :512] #content
X1 = torch.tensor(X1).float()
X2 = final_embeddings_4[:, 512:1024] #Qustion
X2 = torch.tensor(X2).float()
X3 = final_embeddings_4[:, 1024:1536] #correct response
X3 = torch.tensor(X3).float()
X4 = final_embeddings_4[:, 1536:2048] #response 1
X4 = torch.tensor(X4).float()
X5 = final_embeddings_4[:, 2048:2560] #response 2
X5 = torch.tensor(X5).float()
X6 = final_embeddings_4[:, 2560:3072] #response 3
X6 = torch.tensor(X6).float()
X7 = final_embeddings_4[:, 3072:3584] #response 4
X7 = torch.tensor(X7).float()

train_cen(X1, X2, X3, X4, X5, X6, X7, Y)
print('\n distiluse-base-multilingual-cased-v1 \n',
      '_ _ '*10)
X1 = torch.tensor(final_embeddings_5[:, :768]).float()  #content
X2 = torch.tensor(final_embeddings_5[:, 768:1536]).float()  #Qustion
X3 = torch.tensor(final_embeddings_5[:, 1536:2304]).float() #correct response
X4 = torch.tensor(final_embeddings_5[:, 2304:3072]).float() #response 1
X5 = torch.tensor(final_embeddings_5[:, 3072:3840]).float() #response 2
X6 = torch.tensor(final_embeddings_5[:, 3840:4608]).float() #response 3
X7 = torch.tensor(final_embeddings_5[:, 4608:5376]).float() #response 4

train_cen(X1, X2, X3, X4, X5, X6, X7, Y)
print('\n T-Systems-onsite/cross-en-de-roberta-sentence-transformer \n',
        '_ _ '*10)
X1 = torch.tensor(final_embeddings_6[:, :768]).float()  #content
X2 = torch.tensor(final_embeddings_6[:, 768:1536]).float()  #Qustion
X3 = torch.tensor(final_embeddings_6[:, 1536:2304]).float() #correct response
X4 = torch.tensor(final_embeddings_6[:, 2304:3072]).float() #response 1
X5 = torch.tensor(final_embeddings_6[:, 3072:3840]).float() #response 2
X6 = torch.tensor(final_embeddings_6[:, 3840:4608]).float() #response 3
X7 = torch.tensor(final_embeddings_6[:, 4608:5376]).float() #response 4

train_cen(X1, X2, X3, X4, X5, X6, X7, Y)
print('\n jgrosjean-mathesis/swissbert-for-sentence-embeddings \n',
        '____'*10)
print('________________________END_______________________')
# %%