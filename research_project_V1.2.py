#%%
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import seaborn as sns
import torch
import matplotlib.pyplot as plt
from sentence_transformers import  util
import statistics

#%%
df = pd.read_csv("C:/Users/Sarthak/Downloads/research project/Selection_Items_Tuebingen_1.csv", sep=';', encoding='ISO-8859-1')
df.fillna("N/A", inplace=True)
df.head(5)
#%%
single_df=df[df["Item Type"]=="single choice"]
multi_df=df[df["Item Type"]=="multiple response"]
other_df=df[df["Item Type"]=="order"]


#%%
###################################################################
###################  Nicely Working  ##############################
###################################################################
# model names
# sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 ~ done for single & multi
# xlm-r-distilroberta-base-paraphrase-v1 ~ done for single
# sentence-transformers/distiluse-base-multilingual-cased-v1 ~ done for single
# T-Systems-onsite/cross-en-de-roberta-sentence-transformer ~ done for single

model_name= 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
model = SentenceTransformer(model_name)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model= model.to(device)
single_df.reset_index(drop=True, inplace=True)
#embeddings
para_embd=[]
corr_embd=[]
quest_embd=[]
response_1_embd=[]
response_2_embd=[]
response_3_embd=[]
response_4_embd=[]
response_5_embd=[]
response_6_embd=[]
response_7_embd=[]
#similarity scores
para_question=[]
question_response_correct=[]
para_response_correct=[]
para_response_1=[]
para_response_2=[]
para_response_3=[]
para_response_4=[]
para_response_5=[]
para_response_6=[]
para_response_7=[]
quest_response_1=[]
quest_response_2=[]
quest_response_3=[]
quest_response_4=[]
quest_response_5=[]
quest_response_6=[]
quest_response_7=[]

for i in range(len(single_df)):
    paragraph_embedding = model.encode(single_df['Content'][i])
    response_correct_embedding = model.encode(single_df['Correct Response'][i])
    question_embedding = model.encode(single_df['Question'][i])
    response_1_embedding = model.encode(single_df['Response Option 1'][i])
    response_2_embedding = model.encode(single_df['Response Option 2'][i])
    response_3_embedding = model.encode(single_df['Response Option 3'][i])
    response_4_embedding = model.encode(single_df['Response Option 4'][i])
    response_5_embedding = model.encode(single_df['Response Option 5'][i])
    response_6_embedding = model.encode(single_df['Response Option 6'][i])
    response_7_embedding = model.encode(single_df['Response Option 7'][i])
    
    para_question_sim = util.cos_sim(paragraph_embedding, question_embedding)
    para_response_correct_sim = util.cos_sim(paragraph_embedding, response_correct_embedding)
    question_response_correct_sim = util.cos_sim(question_embedding, response_correct_embedding)
    
    question_response_1_sim = util.cos_sim(question_embedding, response_1_embedding)
    question_response_2_sim = util.cos_sim(question_embedding, response_2_embedding)
    question_response_3_sim = util.cos_sim(question_embedding, response_3_embedding)
    question_response_4_sim = util.cos_sim(question_embedding, response_4_embedding)
    question_response_5_sim = util.cos_sim(question_embedding, response_5_embedding)
    question_response_6_sim = util.cos_sim(question_embedding, response_6_embedding)
    question_response_7_sim = util.cos_sim(question_embedding, response_7_embedding)
    
    para_response_1_sim = util.cos_sim(paragraph_embedding, response_1_embedding)
    para_response_2_sim = util.cos_sim(paragraph_embedding, response_2_embedding)
    para_response_3_sim = util.cos_sim(paragraph_embedding, response_3_embedding)
    para_response_4_sim = util.cos_sim(paragraph_embedding, response_4_embedding)
    para_response_5_sim = util.cos_sim(paragraph_embedding, response_5_embedding)
    para_response_6_sim = util.cos_sim(paragraph_embedding, response_6_embedding)
    para_response_7_sim = util.cos_sim(paragraph_embedding, response_7_embedding)
    
    para_embd.append(paragraph_embedding)
    corr_embd.append(response_correct_embedding)
    quest_embd.append(question_embedding)
    response_1_embd.append(response_1_embedding)
    response_2_embd.append(response_2_embedding)
    response_3_embd.append(response_3_embedding)
    response_4_embd.append(response_4_embedding)
    response_5_embd.append(response_5_embedding)
    response_6_embd.append(response_6_embedding)
    response_7_embd.append(response_7_embedding)
    
    
    para_question.append(para_question_sim)
    question_response_correct.append(question_response_correct_sim)
    para_response_correct.append(para_response_correct_sim)
    
    para_response_1.append(para_response_1_sim)
    para_response_2.append(para_response_2_sim)
    para_response_3.append(para_response_3_sim)
    para_response_4.append(para_response_4_sim)
    para_response_5.append(para_response_5_sim)
    para_response_6.append(para_response_6_sim)
    para_response_7.append(para_response_7_sim)
    
    quest_response_1.append(question_response_1_sim)
    quest_response_2.append(question_response_2_sim)
    quest_response_3.append(question_response_3_sim)
    quest_response_4.append(question_response_4_sim)
    quest_response_5.append(question_response_5_sim)
    quest_response_6.append(question_response_6_sim)
    quest_response_7.append(question_response_7_sim)

print("done")
#%% make a df and export to csv

single_df['Para Embedding'] = para_embd
single_df['Correct Response Embedding'] = corr_embd
single_df['Question Embedding'] = quest_embd
single_df['Response Option 1 Embedding'] = response_1_embd
single_df['Response Option 2 Embedding'] = response_2_embd
single_df['Response Option 3 Embedding'] = response_3_embd
single_df['Response Option 4 Embedding'] = response_4_embd
single_df['Response Option 5 Embedding'] = response_5_embd
single_df['Response Option 6 Embedding'] = response_6_embd
single_df['Response Option 7 Embedding'] = response_7_embd

single_df['Para-Question Similarity'] = para_question
single_df['Question-Correct Response Similarity'] = question_response_correct
single_df['Para-Correct Response Similarity'] = para_response_correct

single_df['Para-Response Option 1 Similarity'] = para_response_1
single_df['Para-Response Option 2 Similarity'] = para_response_2
single_df['Para-Response Option 3 Similarity'] = para_response_3
single_df['Para-Response Option 4 Similarity'] = para_response_4
single_df['Para-Response Option 5 Similarity'] = para_response_5
single_df['Para-Response Option 6 Similarity'] = para_response_6
single_df['Para-Response Option 7 Similarity'] = para_response_7

single_df['Question-Response Option 1 Similarity'] = quest_response_1
single_df['Question-Response Option 2 Similarity'] = quest_response_2
single_df['Question-Response Option 3 Similarity'] = quest_response_3
single_df['Question-Response Option 4 Similarity'] = quest_response_4
single_df['Question-Response Option 5 Similarity'] = quest_response_5
single_df['Question-Response Option 6 Similarity'] = quest_response_6
single_df['Question-Response Option 7 Similarity'] = quest_response_7
model_name_safe = model_name.replace('/', '_')
single_df.to_csv(f'{model_name_safe}_single_df.csv', index=False)

#%% Convert the similarity scores to floats for some nice plots
para_response_float = [float(sim) for sim in para_response_correct] 
para_question_float = [float(sim) for sim in para_question] 
question_response_correct_float = [float(sim) for sim in question_response_correct] 
question_response_correct1_float = [float(sim) for sim in quest_response_1] 
question_response_correct2_float = [float(sim) for sim in quest_response_2] 
question_response_correct3_float = [float(sim) for sim in quest_response_3] 
question_response_correct4_float = [float(sim) for sim in quest_response_4] 
question_response_correct5_float = [float(sim) for sim in quest_response_5] 
question_response_correct6_float = [float(sim) for sim in quest_response_6] 
question_response_correct7_float = [float(sim) for sim in quest_response_7]
#Plots by GPT
data = {
    'Correct Response Paragraph' : para_response_float,
    'Question Paragraph' : para_question_float,
    'Correct Response Question': question_response_correct_float,
    'Response 1': question_response_correct1_float,
    'Response 2': question_response_correct2_float,
    'Response 3': question_response_correct3_float,
    'Response 4': question_response_correct4_float,
    'Response 5': question_response_correct5_float,
    'Response 6': question_response_correct6_float,
    'Response 7': question_response_correct7_float,
}

# Create box plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.boxplot(data.values(), patch_artist=True, boxprops=dict(facecolor='lightblue', color='blue'))

# Add labels
ax.set_xticks(range(1, len(data) + 1))
ax.set_xticklabels(data.keys(), rotation=90)
ax.set_ylabel('Cosine Similarity')
ax.set_title('Comparison of Similarity Scores: Correct vs Other Responses')
ax.axhline(statistics.mean(data['Correct Response Paragraph']), color='red', linestyle='--', linewidth=0.8,label='mean correct response paragraph similarity')
ax.axhline(statistics.mean(data['Correct Response Question']), color='blue', linestyle='--', linewidth=0.8,label='mean correct response question similarity')
ax.legend(loc='upper right', fontsize=10)
plt.tight_layout()
plt.show()
fig, ax = plt.subplots(figsize=(12, 6))

# Plot correct response similarity
ax.scatter(range(len(para_question_float)), para_question_float, label='para_question', color='black', marker='X')
ax.scatter(range(len(para_response_float)), para_response_float, label='para_response', color='black', marker='X')
ax.plot(range(len(question_response_correct_float)), question_response_correct_float, label='Correct Response', color='blue', marker='o')

# Plot other responses' similarities
ax.plot(range(len(quest_response_1)), question_response_correct1_float, label='Response 1', alpha=0.2,color='green', linestyle='--')
ax.plot(range(len(quest_response_2)), question_response_correct2_float, label='Response 2', alpha=0.2,color='purple', linestyle='--')
ax.plot(range(len(quest_response_3)), question_response_correct3_float, label='Response 3', alpha=0.2,color='brown', linestyle='--')
ax.plot(range(len(quest_response_4)), question_response_correct4_float, label='Response 4', alpha=0.2,color='yellow', linestyle='--')
ax.plot(range(len(quest_response_5)), question_response_correct5_float, label='Response 5', alpha=0.2,color='cyan', linestyle='--')
ax.plot(range(len(quest_response_6)), question_response_correct6_float, label='Response 6', alpha=0.2,color='grey', linestyle='--')
ax.plot(range(len(quest_response_7)), question_response_correct7_float, label='Response 7', alpha=0.2,color='red', linestyle='--')

# Highlight correct response
ax.axhline(0, color='black', linestyle='--', linewidth=0.8, label='Zero Similarity')

# Add labels, legend, and title
ax.set_xlabel('Index', fontsize=12)
ax.set_ylabel('Cosine Similarity', fontsize=12)
ax.set_title('Correct vs Other Responses: Similarity Trends', fontsize=14)
ax.legend(loc='best', fontsize=10)

plt.tight_layout()
plt.show()

#%%
# Calculate the difference between the two variables
similarity_difference = [abs(pq) - abs(qr) for pq, qr in zip(para_question_float, question_response_correct_float)]
fig, ax = plt.subplots()
ax.plot(range(len(similarity_difference)), similarity_difference, label='Difference (Para vs Question - Question vs Response Correct)')
ax.set_xlabel('Index')
ax.set_ylabel('Difference in Cosine Similarity')
ax.set_title('Difference in Similarity Scores')
ax.legend()
# Show the plot
plt.show()

plt.hist(similarity_difference,bins=50)

###############################################################
###############################################################
###############################################################
#%% Multi choice pre processing 
multi_df.reset_index(drop=True, inplace=True)
multi_df[['correct response 1','correct response 2','correct response 3' ]] = multi_df['Correct Response'].str.split(';',expand=True,n=2)
multi_df['correct response 1'] = multi_df['correct response 1'].str.strip()
multi_df['correct response 2'] = multi_df['correct response 2'].str.strip()
multi_df['correct response 3'] = multi_df['correct response 3'].str.strip()
multi_df.fillna('N/A',inplace=True)
#%%
# model names
# sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 ~ done for single & multi
# xlm-r-distilroberta-base-paraphrase-v1 ~ done for single & multi
# sentence-transformers/distiluse-base-multilingual-cased-v1 ~ done for single & multi
# T-Systems-onsite/cross-en-de-roberta-sentence-transformer ~ done for single & multi

model_name= 'T-Systems-onsite/cross-en-de-roberta-sentence-transformer'
model = SentenceTransformer(model_name)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model= model.to(device)
single_df.reset_index(drop=True, inplace=True)
#embeddings
para_embd=[]
corr_embd=[]
corr_embd_1=[]
corr_embd_2=[]
corr_embd_3=[]
quest_embd=[]
response_1_embd=[]
response_2_embd=[]
response_3_embd=[]
response_4_embd=[]
response_5_embd=[]
response_6_embd=[]
response_7_embd=[]
#similarity scores
para_question=[]
question_response_correct=[]
question_response_correct_1=[]
question_response_correct_2=[]
question_response_correct_3=[]
para_response_correct=[]
para_response_correct_1=[]
para_response_correct_2=[]
para_response_correct_3=[]
para_response_1=[]
para_response_2=[]
para_response_3=[]
para_response_4=[]
para_response_5=[]
para_response_6=[]
para_response_7=[]
quest_response_1=[]
quest_response_2=[]
quest_response_3=[]
quest_response_4=[]
quest_response_5=[]
quest_response_6=[]
quest_response_7=[]

for i in range(len(multi_df)):
    paragraph_embedding = model.encode(multi_df['Content'][i])
    response_correct_embedding = model.encode(multi_df['Correct Response'][i])
    response_correct_embedding_1 = model.encode(multi_df['correct response 1'][i])
    response_correct_embedding_2 = model.encode(multi_df['correct response 2'][i])
    response_correct_embedding_3 = model.encode(multi_df['correct response 3'][i])
    question_embedding = model.encode(multi_df['Question'][i])
    response_1_embedding = model.encode(multi_df['Response Option 1'][i])
    response_2_embedding = model.encode(multi_df['Response Option 2'][i])
    response_3_embedding = model.encode(multi_df['Response Option 3'][i])
    response_4_embedding = model.encode(multi_df['Response Option 4'][i])
    response_5_embedding = model.encode(multi_df['Response Option 5'][i])
    response_6_embedding = model.encode(multi_df['Response Option 6'][i])
    response_7_embedding = model.encode(multi_df['Response Option 7'][i])
    
    para_question_sim = util.cos_sim(paragraph_embedding, question_embedding)
    para_response_correct_sim = util.cos_sim(paragraph_embedding, response_correct_embedding)
    para_response_correct_1_sim = util.cos_sim(paragraph_embedding, response_correct_embedding_1)
    para_response_correct_2_sim = util.cos_sim(paragraph_embedding, response_correct_embedding_2)
    para_response_correct_3_sim = util.cos_sim(paragraph_embedding, response_correct_embedding_3)
    question_response_correct_sim = util.cos_sim(question_embedding, response_correct_embedding)
    question_response_correct_1_sim = util.cos_sim(question_embedding, response_correct_embedding_1)
    question_response_correct_2_sim = util.cos_sim(question_embedding, response_correct_embedding_2)
    question_response_correct_3_sim = util.cos_sim(question_embedding, response_correct_embedding_3)
    
    correct_response_1_2_sim = util.cos_sim(response_correct_embedding_1, response_correct_embedding_2)
    correct_response_1_3_sim = util.cos_sim(response_correct_embedding_1, response_correct_embedding_3)
    correct_response_2_3_sim = util.cos_sim(response_correct_embedding_2, response_correct_embedding_3)
    
    question_response_1_sim = util.cos_sim(question_embedding, response_1_embedding)
    question_response_2_sim = util.cos_sim(question_embedding, response_2_embedding)
    question_response_3_sim = util.cos_sim(question_embedding, response_3_embedding)
    question_response_4_sim = util.cos_sim(question_embedding, response_4_embedding)
    question_response_5_sim = util.cos_sim(question_embedding, response_5_embedding)
    question_response_6_sim = util.cos_sim(question_embedding, response_6_embedding)
    question_response_7_sim = util.cos_sim(question_embedding, response_7_embedding)
    
    para_response_1_sim = util.cos_sim(paragraph_embedding, response_1_embedding)
    para_response_2_sim = util.cos_sim(paragraph_embedding, response_2_embedding)
    para_response_3_sim = util.cos_sim(paragraph_embedding, response_3_embedding)
    para_response_4_sim = util.cos_sim(paragraph_embedding, response_4_embedding)
    para_response_5_sim = util.cos_sim(paragraph_embedding, response_5_embedding)
    para_response_6_sim = util.cos_sim(paragraph_embedding, response_6_embedding)
    para_response_7_sim = util.cos_sim(paragraph_embedding, response_7_embedding)
    
    para_embd.append(paragraph_embedding)
    corr_embd.append(response_correct_embedding)
    corr_embd_1.append(response_correct_embedding_1)
    corr_embd_2.append(response_correct_embedding_2)
    corr_embd_3.append(response_correct_embedding_3)
    quest_embd.append(question_embedding)
    response_1_embd.append(response_1_embedding)
    response_2_embd.append(response_2_embedding)
    response_3_embd.append(response_3_embedding)
    response_4_embd.append(response_4_embedding)
    response_5_embd.append(response_5_embedding)
    response_6_embd.append(response_6_embedding)
    response_7_embd.append(response_7_embedding)
    
    para_question.append(para_question_sim)
    question_response_correct.append(question_response_correct_sim)
    question_response_correct_1.append(question_response_correct_1_sim)
    question_response_correct_2.append(question_response_correct_2_sim)
    question_response_correct_3.append(question_response_correct_3_sim)
    para_response_correct.append(para_response_correct_sim)
    para_response_correct_1.append(para_response_correct_1_sim)
    para_response_correct_2.append(para_response_correct_2_sim)
    para_response_correct_3.append(para_response_correct_3_sim)
    
    para_response_1.append(para_response_1_sim)
    para_response_2.append(para_response_2_sim)
    para_response_3.append(para_response_3_sim)
    para_response_4.append(para_response_4_sim)
    para_response_5.append(para_response_5_sim)
    para_response_6.append(para_response_6_sim)
    para_response_7.append(para_response_7_sim)
    
    quest_response_1.append(question_response_1_sim)
    quest_response_2.append(question_response_2_sim)
    quest_response_3.append(question_response_3_sim)
    quest_response_4.append(question_response_4_sim)
    quest_response_5.append(question_response_5_sim)
    quest_response_6.append(question_response_6_sim)
    quest_response_7.append(question_response_7_sim)

print("done")
#%% make a df and export to csv

multi_df['Para Embedding'] = para_embd
multi_df['Correct Response Embedding'] = corr_embd
multi_df['Correct Response 1 Embedding'] =corr_embd_1
multi_df['Correct Response 2 Embedding'] =corr_embd_2
multi_df['Correct Response 3 Embedding'] =corr_embd_3
multi_df['Question Embedding'] = quest_embd
multi_df['Response Option 1 Embedding'] = response_1_embd
multi_df['Response Option 2 Embedding'] = response_2_embd
multi_df['Response Option 3 Embedding'] = response_3_embd
multi_df['Response Option 4 Embedding'] = response_4_embd
multi_df['Response Option 5 Embedding'] = response_5_embd
multi_df['Response Option 6 Embedding'] = response_6_embd
multi_df['Response Option 7 Embedding'] = response_7_embd
multi_df['Para-Question Similarity'] = para_question
multi_df['Question-Correct Response Similarity'] = question_response_correct
multi_df['Question-Correct Response 1 Similarity'] =question_response_correct_1
multi_df['Question-Correct Response 2 Similarity'] =question_response_correct_2
multi_df['Question-Correct Response 3 Similarity'] =question_response_correct_3
multi_df['Para-Correct Response Similarity'] = para_response_correct
multi_df['Para-Correct Response 1 Similarity'] =para_response_correct_1
multi_df['Para-Correct Response 2 Similarity'] =para_response_correct_2
multi_df['Para-Correct Response 3 Similarity'] =para_response_correct_3
multi_df['Para-Response Option 1 Similarity'] = para_response_1
multi_df['Para-Response Option 2 Similarity'] = para_response_2
multi_df['Para-Response Option 3 Similarity'] = para_response_3
multi_df['Para-Response Option 4 Similarity'] = para_response_4
multi_df['Para-Response Option 5 Similarity'] = para_response_5
multi_df['Para-Response Option 6 Similarity'] = para_response_6
multi_df['Para-Response Option 7 Similarity'] = para_response_7
multi_df['Question-Response Option 1 Similarity'] = quest_response_1
multi_df['Question-Response Option 2 Similarity'] = quest_response_2
multi_df['Question-Response Option 3 Similarity'] = quest_response_3
multi_df['Question-Response Option 4 Similarity'] = quest_response_4
multi_df['Question-Response Option 5 Similarity'] = quest_response_5
multi_df['Question-Response Option 6 Similarity'] = quest_response_6
multi_df['Question-Response Option 7 Similarity'] = quest_response_7
model_name_safe = model_name.replace('/', '_')
multi_df.to_csv(f'{model_name_safe}_multi_df.csv', index=False)


#%%
plt.scatter(y=(single_df['Item Discrimination']
               ),x=single_df['Title'],c=single_df['Content'].astype('category').cat.codes,cmap='viridis'
            )
plt.xticks(rotation=90)
# %%
model = SentenceTransformer("distiluse-base-multilingual-cased-v2")#paraphrase-multilingual-MiniLM-L12-v2")
#%%
sentence = df['Content']
question = df['Question']
# Sentences are encoded by calling model.encode()

embedding = model.encode(sentence)
embedding2 = model.encode(question)
# %%

# %%
cos_sim = []

for i in range(0, len(embedding)):
    for j in range(i, len(embedding2)):
        vec1 = torch.tensor(embedding[i], dtype=torch.float32).to(device)
        vec2 = torch.tensor(embedding2[j], dtype=torch.float32).to(device)
        
        # cosine similarity
        sim = torch.cosine_similarity(vec1, vec2, dim=0)
        cos_sim.append((sim.item(), i,j))

# mean of cosine similarities
mean_sim = sum(sim[0] for sim in cos_sim) / len(cos_sim)
print("Mean cosine similarity:", mean_sim)

# %%

similarity_values = [sim[0] for sim in cos_sim]
pair_labels = [f"{sim[1]}-{sim[2]}" for sim in cos_sim]

# Plotting
plt.figure(figsize=(12, 6))
plt.bar(range(len(similarity_values)), similarity_values, tick_label=pair_labels)
plt.xlabel("Embedding Pairs (i-j)")
plt.ylabel("Cosine Similarity")
plt.title("Cosine Similarity between Each Pair of Embeddings")
plt.xticks(rotation=90)
plt.show()
# %%
# Extract the similarity values and pair indices
#similarity_values = [sim[0] for sim in cos_sim]
pair_indices = [(sim[1], sim[2]) for sim in cos_sim]

# Scatter plot to show clustering
plt.figure(figsize=(10, 6))
x_values = range(len(similarity_values))  
plt.scatter(x_values, similarity_values, alpha=0.6, c=similarity_values, cmap='viridis')
plt.colorbar(label="Cosine Similarity")
plt.xlabel("Embedding Pairs (Sequential Index)")
plt.ylabel("Cosine Similarity")
plt.title("Scatter Plot Showing Cosine Similarity Clusters")
plt.show()

# %%
# Histogram and KDE plot for cosine similarity distribution
plt.figure(figsize=(10, 6))
sns.histplot(similarity_values, bins=30, kde=True, color="skyblue")
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.title("Distribution of Cosine Similarity")
plt.show()

# %%
