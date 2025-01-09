#%%
import embeddings_gen as emb
# %%
import pandas as pd
df=pd.read_csv(r"C:\Users\Sarthak\Downloads\research project\Selection_Items_Tuebingen_1.csv", sep=';', encoding='ISO-8859-1')
# %%
'''
response_type: default 'single choice', or 'multiple response'
'''
df= emb.preprocess_text(df,clean_text=True,response_type='single choice')
# %%
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
