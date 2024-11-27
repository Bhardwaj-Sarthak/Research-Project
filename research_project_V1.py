#%%
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import seaborn as sns
import torch
import matplotlib.pyplot as plt
from sentence_transformers import  util

#%%
df = pd.read_csv("C:/Users/Sarthak/Downloads/research project/Selection_Items_Tuebingen_1.csv", sep=';', encoding='ISO-8859-1')
df.head(20)
# %%
#df['Item Type'].value_counts()
df['Question'].value_counts()
#df['Title'].value_counts()
#df['InternCode'].value_counts()
#%%
single_df=df[df["Item Type"]=="single choice"]
multi_df=df[df["Item Type"]=="multiple choice"]
other_df=df[df["Item Type"]=="order"]
#%%
###################################################################
###################  Nicely Working  ##############################
###################################################################
'''
add incorrect responses similarity score calculation an then plot x = index of the question, y = similarity scores (question-content ; content- correct response ; content- incorrect responses)
'''
# sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
# xlm-r-distilroberta-base-paraphrase-v1
# sentence-transformers/distiluse-base-multilingual-cased-v1
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model= model.to(device)
single_df.reset_index(drop=True, inplace=True)
para_embd=[]
corr_embd=[]
quest_embd=[]
para_question=[]
question_response_correct=[]
for i in range(len(single_df)):
    paragraph_embedding = model.encode(single_df['Content'][i])
    response_correct_embedding = model.encode(single_df['Correct Response'][i])
    question_embedding = model.encode(single_df['Question'][i])
    para_embd.append(paragraph_embedding)
    corr_embd.append(response_correct_embedding)
    quest_embd.append(question_embedding)
    para_question_sim = util.cos_sim(paragraph_embedding, question_embedding)
    para_question.append(para_question_sim)
    question_response_correct_sim = util.cos_sim(question_embedding, response_correct_embedding)
    question_response_correct.append(question_response_correct_sim)

print(para_question)
print(question_response_correct)
#%%
# Convert the similarity scores to floats 
para_question_float = [float(sim) for sim in para_question] 
question_response_correct_float = [float(sim) for sim in question_response_correct] 
fig, ax = plt.subplots() 
ax.scatter(range(len(para_question)), para_question_float, label='Para vs Question', marker='o') 
ax.scatter(range(len(question_response_correct)), question_response_correct_float, label='Question vs Response Correct', marker='x') 
ax.set_xlabel('Index') 
ax.set_ylabel('Cosine Similarity') 
ax.set_title('Similarity Scores') 
ax.legend() 
plt.show()
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
