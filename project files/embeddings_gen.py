import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
from transformers import AutoModel, AutoTokenizer
import re

def preprocess_text(df,clean_text=True,response_type='single choice'):
    def remove_special_characters(text):
        if not pd.isnull(text):
            text = re.sub(r'[^A-Za-zäöüÄÖÜß\s]', '', text)
        return text
    if response_type == "multiple response":
        df = df[df["Item Type"] == "multiple response"]
        df.reset_index(drop=True, inplace=True)
        split_responses = df['Correct Response'].str.split(';', expand=True, n=2)
        split_responses.columns = ['correct response 1', 'correct response 2', 'correct response 3']
        
        # Ensure all columns are present
        for col in ['correct response 1', 'correct response 2', 'correct response 3']:
            if col not in split_responses.columns:
                split_responses[col] = None
        
        df = pd.concat([df, split_responses], axis=1)
        
        df['correct response 1'] = df['correct response 1'].str.strip()
        df['correct response 2'] = df['correct response 2'].str.strip()
        df['correct response 3'] = df['correct response 3'].str.strip()
        if clean_text==True:
            df['Content'] = df['Content'].apply(remove_special_characters)
            df['Question'] = df['Question'].apply(remove_special_characters)
            df['Correct Response'] = df['Correct Response'].apply(remove_special_characters)
            df['correct response 1'] = df['correct response 1'].apply(remove_special_characters)
            df['correct response 2'] = df['correct response 2'].apply(remove_special_characters)
            df['correct response 3'] = df['correct response 3'].apply(remove_special_characters)
            df['Response Option 1'] = df['Response Option 1'].apply(remove_special_characters)
            df['Response Option 2'] = df['Response Option 2'].apply(remove_special_characters)
            df['Response Option 3'] = df['Response Option 3'].apply(remove_special_characters)
            df['Response Option 4'] = df['Response Option 4'].apply(remove_special_characters)
            df['Response Option 5'] = df['Response Option 5'].apply(remove_special_characters)
            df['Response Option 6'] = df['Response Option 6'].apply(remove_special_characters)
            df['Response Option 7'] = df['Response Option 7'].apply(remove_special_characters)
            df.fillna(np.nan, inplace=True) 
        else:
            pass
    elif response_type == "single choice":
        df = df[df["Item Type"] == "single choice"]
        if clean_text==True:
            df['Content'] = df['Content'].apply(remove_special_characters)
            df['Question'] = df['Question'].apply(remove_special_characters)
            df['Correct Response'] = df['Correct Response'].apply(remove_special_characters)
            df['Response Option 1'] = df['Response Option 1'].apply(remove_special_characters)
            df['Response Option 2'] = df['Response Option 2'].apply(remove_special_characters)
            df['Response Option 3'] = df['Response Option 3'].apply(remove_special_characters)
            df['Response Option 4'] = df['Response Option 4'].apply(remove_special_characters)
            df['Response Option 5'] = df['Response Option 5'].apply(remove_special_characters)
            df['Response Option 6'] = df['Response Option 6'].apply(remove_special_characters)
            df['Response Option 7'] = df['Response Option 7'].apply(remove_special_characters)
            df.fillna(np.nan, inplace=True)
        else:
            pass
    else:
        df=df
        df.fillna(np.nan, inplace=True)
        print("No response type given assumning only single choice questions")
        print("No response type given assuming only single choice questions")
        
    return df



def generate_embeddings(df, model_name, response_type='single choice'):
    '''
        Takes in dataframe with text columns ['InternCode', 'Title', 'Content', 'Item Discrimination',
       'Item Difficulty', 'Question', 'Correct Response', 'Item Type',
       'Response Option 1', 'Response Option 2', 'Response Option 3',
       'Response Option 4', 'Response Option 5', 'Response Option 6',
       'Response Option 7'] and generates embeddings for the text columns using the SentenceTransformer or Transformer models.
    
    Parameters:
        df: dataframe with text columns
        model_name: 
            str, 
            default 'paraphrase-multilingual-MiniLM-L12-v2',
                'xlm-r-distilroberta-base-paraphrase-v1',
                'distiluse-base-multilingual-cased-v1',
                'T-Systems-onsite/cross-en-de-roberta-sentence-transformer',
            form the SentenceTransformer library
            or
            fine-tuned transformer models for Swiss-German from the Hugging Face Transformers library:
            'ZurichNLP/unsup-simcse-xlm-roberta-base',
            'jgrosjean-mathesis/swissbert-for-sentence-embeddings'
            from the Hugging Face Transformers library
            
    Returns:
        a dataset (each cell has value as float) with the embeddings for the for all text columns, each embedding feature is a column in the dataset
    
    '''
    sentence_transformer_models = [
        'paraphrase-multilingual-MiniLM-L12-v2',
        'xlm-r-distilroberta-base-paraphrase-v1',
        'distiluse-base-multilingual-cased-v1',
        'T-Systems-onsite/cross-en-de-roberta-sentence-transformer'
    ]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model_name in sentence_transformer_models:
        model = SentenceTransformer(model_name).to(device)        
        def embed_text(text):
            return model.encode(text, convert_to_tensor=True)
    elif model_name == 'ZurichNLP/unsup-simcse-xlm-roberta-base':
        model = AutoModel.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
        def embed_text(text):
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            return model(**inputs).last_hidden_state.mean(dim=1)

    elif model_name =="jgrosjean-mathesis/swissbert-for-sentence-embeddings":
        model = AutoModel.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.set_default_language("de_CH")
        def embed_text(text):
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
            return model(**inputs).last_hidden_state.mean(dim=1)
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    content_embeddings = []
    corr_rep_embeddings = []
    question_embeddings = []
    resp_opt1_embeddings = []
    resp_opt2_embeddings = []
    resp_opt3_embeddings = []
    resp_opt4_embeddings = []
    resp_opt5_embeddings = []
    resp_opt6_embeddings = []
    resp_opt7_embeddings = []

    for i in range(len(df)):
        if 'Content' in df.columns:
            if pd.notna(df['Content'].iloc[i]):
                content_embeddings.append(embed_text(df['Content'].iloc[i]).cpu().detach().numpy())
            else:
                content_embeddings.append(np.zeros(embed_text("").cpu().detach().numpy().shape))
        else:
            raise Warning("No 'Content' column in the dataframe")

        if 'Correct Response' in df.columns:
            if pd.notna(df['Correct Response'].iloc[i]):
                corr_rep_embeddings.append(embed_text(df['Correct Response'].iloc[i]).cpu().detach().numpy())
            else:
                corr_rep_embeddings.append(np.zeros(embed_text("").cpu().detach().numpy().shape))
        else:
            raise Warning("No 'Correct Response' column in the dataframe")

        if 'Question' in df.columns:
            if pd.notna(df['Question'].iloc[i]):
                question_embeddings.append(embed_text(df['Question'].iloc[i]).cpu().detach().numpy())
            else:
                question_embeddings.append(np.zeros(embed_text("").cpu().detach().numpy().shape))
        else:
            raise Warning("No 'Question' column in the dataframe")

        if 'Response Option 1' in df.columns:
            if pd.notna(df['Response Option 1'].iloc[i]):
                resp_opt1_embeddings.append(embed_text(df['Response Option 1'].iloc[i]).cpu().detach().numpy())
            else:
                resp_opt1_embeddings.append(np.zeros(embed_text("").cpu().detach().numpy().shape))
        else:
            raise Warning("No 'Response Option 1' column in the dataframe")

        if 'Response Option 2' in df.columns:
            if pd.notna(df['Response Option 2'].iloc[i]):
                resp_opt2_embeddings.append(embed_text(df['Response Option 2'].iloc[i]).cpu().detach().numpy())
            else:
                resp_opt2_embeddings.append(np.zeros(embed_text("").cpu().detach().numpy().shape))
        else:
            raise Warning("No 'Response Option 2' column in the dataframe")

        if 'Response Option 3' in df.columns:
            if pd.notna(df['Response Option 3'].iloc[i]):
                resp_opt3_embeddings.append(embed_text(df['Response Option 3'].iloc[i]).cpu().detach().numpy())
            else:
                resp_opt3_embeddings.append(np.zeros(embed_text("").cpu().detach().numpy().shape))
        else:
            raise Warning("No 'Response Option 3' column in the dataframe")

        if 'Response Option 4' in df.columns:
            if pd.notna(df['Response Option 4'].iloc[i]):
                resp_opt4_embeddings.append(embed_text(df['Response Option 4'].iloc[i]).cpu().detach().numpy())
            else:
                resp_opt4_embeddings.append(np.zeros(embed_text("").cpu().detach().numpy().shape))
        else:
            raise Warning("No 'Response Option 4' column in the dataframe")

        if 'Response Option 5' in df.columns:
            if pd.notna(df['Response Option 5'].iloc[i]):
                resp_opt5_embeddings.append(embed_text(df['Response Option 5'].iloc[i]).cpu().detach().numpy())
            else:
                resp_opt5_embeddings.append(np.zeros(embed_text("").cpu().detach().numpy().shape))
            
        else:
            raise Warning("No 'Response Option 5' column in the dataframe")

        if 'Response Option 6' in df.columns:
            if pd.notna(df['Response Option 6'].iloc[i]):
                resp_opt6_embeddings.append(embed_text(df['Response Option 6'].iloc[i]).cpu().detach().numpy())
            else:
                resp_opt6_embeddings.append(np.zeros(embed_text("").cpu().detach().numpy().shape))
            
        else:
            raise Warning("No 'Response Option 6' column in the dataframe")

        if 'Response Option 7' in df.columns:
           if pd.notna(df['Response Option 7'].iloc[i]):
               resp_opt7_embeddings.append(embed_text(df['Response Option 7'].iloc[i]).cpu().detach().numpy())
           else:
           
               resp_opt7_embeddings.append(np.zeros(embed_text("").cpu().detach().numpy().shape))
            
        else:
            raise Warning("No 'Response Option 7' column in the dataframe")
    
    content_embeddings = np.array(content_embeddings).reshape(-1, content_embeddings[0].shape[-1])
    corr_rep_embeddings = np.array(corr_rep_embeddings).reshape(-1, corr_rep_embeddings[0].shape[-1])
    question_embeddings = np.array(question_embeddings).reshape(-1, question_embeddings[0].shape[-1])
    resp_opt1_embeddings = np.array(resp_opt1_embeddings).reshape(-1, resp_opt1_embeddings[0].shape[-1])
    resp_opt2_embeddings = np.array(resp_opt2_embeddings).reshape(-1, resp_opt2_embeddings[0].shape[-1])
    resp_opt3_embeddings = np.array(resp_opt3_embeddings).reshape(-1, resp_opt3_embeddings[0].shape[-1])
    resp_opt4_embeddings = np.array(resp_opt4_embeddings).reshape(-1, resp_opt4_embeddings[0].shape[-1])
    resp_opt5_embeddings = np.array(resp_opt5_embeddings).reshape(-1, resp_opt5_embeddings[0].shape[-1])
    resp_opt6_embeddings = np.array(resp_opt6_embeddings).reshape(-1, resp_opt6_embeddings[0].shape[-1])
    resp_opt7_embeddings = np.array(resp_opt7_embeddings).reshape(-1, resp_opt7_embeddings[0].shape[-1])
    if response_type== 'single choice':
        embeddings = np.concatenate((content_embeddings, corr_rep_embeddings, question_embeddings,
                                     resp_opt1_embeddings, resp_opt2_embeddings, resp_opt3_embeddings,
                                     resp_opt4_embeddings), axis=1)
        print("the shape of embeddings is",embeddings.shape)
    elif response_type== 'multiple response':
        embeddings = np.concatenate((
            content_embeddings, corr_rep_embeddings, question_embeddings,
            resp_opt1_embeddings, resp_opt2_embeddings, resp_opt3_embeddings,
            resp_opt4_embeddings, resp_opt5_embeddings, resp_opt6_embeddings,
            resp_opt7_embeddings
            ), axis=1)
        print("the shape of embeddings is",embeddings.shape)
    else:
        embeddings = np.concatenate((
            content_embeddings, corr_rep_embeddings, question_embeddings,
            resp_opt1_embeddings, resp_opt2_embeddings, resp_opt3_embeddings,
            resp_opt4_embeddings, resp_opt5_embeddings, resp_opt6_embeddings,
            resp_opt7_embeddings
            ), axis=1)
        print("the shape of embeddings is",embeddings.shape)
        raise Warning("No response type given assuming multiple columns with 0 value might be present")
  
    return embeddings


#cosine_sim_content_corr_rep = util.pytorch_cos_sim(torch.tensor(content_embeddings), torch.tensor(corr_rep_embeddings)).cpu().detach().numpy()
#    cosine_sim_content_question = util.pytorch_cos_sim(torch.tensor(content_embeddings), torch.tensor(question_embeddings)).cpu().detach().numpy()
#    cosine_sim_content_resp_opt1 = util.pytorch_cos_sim(torch.tensor(content_embeddings), torch.tensor(resp_opt1_embeddings)).cpu().detach().numpy()
#    cosine_sim_content_resp_opt2 = util.pytorch_cos_sim(torch.tensor(content_embeddings), torch.tensor(resp_opt2_embeddings)).cpu().detach().numpy()
#    cosine_sim_content_resp_opt3 = util.pytorch_cos_sim(torch.tensor(content_embeddings), torch.tensor(resp_opt3_embeddings)).cpu().detach().numpy()
#    cosine_sim_content_resp_opt4 = util.pytorch_cos_sim(torch.tensor(content_embeddings), torch.tensor(resp_opt4_embeddings)).cpu().detach().numpy()
#    cosine_sim_content_resp_opt5 = util.pytorch_cos_sim(torch.tensor(content_embeddings), torch.tensor(resp_opt5_embeddings)).cpu().detach().numpy()
#    cosine_sim_content_resp_opt6 = util.pytorch_cos_sim(torch.tensor(content_embeddings), torch.tensor(resp_opt6_embeddings)).cpu().detach().numpy()
#    cosine_sim_content_resp_opt7 = util.pytorch_cos_sim(torch.tensor(content_embeddings), torch.tensor(resp_opt7_embeddings)).cpu().detach().numpy()
#    
#    cosine_sim_corr_rep_question = util.pytorch_cos_sim(torch.tensor(corr_rep_embeddings), torch.tensor(question_embeddings)).cpu().detach().numpy()
#    cosine_sim_corr_rep_resp_opt1 = util.pytorch_cos_sim(torch.tensor(corr_rep_embeddings), torch.tensor(resp_opt1_embeddings)).cpu().detach().numpy()
#    cosine_sim_corr_rep_resp_opt2 = util.pytorch_cos_sim(torch.tensor(corr_rep_embeddings), torch.tensor(resp_opt2_embeddings)).cpu().detach().numpy()
#    cosine_sim_corr_rep_resp_opt3 = util.pytorch_cos_sim(torch.tensor(corr_rep_embeddings), torch.tensor(resp_opt3_embeddings)).cpu().detach().numpy()
#    cosine_sim_corr_rep_resp_opt4 = util.pytorch_cos_sim(torch.tensor(corr_rep_embeddings), torch.tensor(resp_opt4_embeddings)).cpu().detach().numpy()
#    cosine_sim_corr_rep_resp_opt5 = util.pytorch_cos_sim(torch.tensor(corr_rep_embeddings), torch.tensor(resp_opt5_embeddings)).cpu().detach().numpy()
#    cosine_sim_corr_rep_resp_opt6 = util.pytorch_cos_sim(torch.tensor(corr_rep_embeddings), torch.tensor(resp_opt6_embeddings)).cpu().detach().numpy()
#    cosine_sim_corr_rep_resp_opt7 = util.pytorch_cos_sim(torch.tensor(corr_rep_embeddings), torch.tensor(resp_opt7_embeddings)).cpu().detach().numpy()
#    
#    cosine_sim_question_resp_opt1 = util.pytorch_cos_sim(torch.tensor(question_embeddings), torch.tensor(resp_opt1_embeddings)).cpu().detach().numpy()
#    cosine_sim_question_resp_opt2 = util.pytorch_cos_sim(torch.tensor(question_embeddings), torch.tensor(resp_opt2_embeddings)).cpu().detach().numpy()
#    cosine_sim_question_resp_opt3 = util.pytorch_cos_sim(torch.tensor(question_embeddings), torch.tensor(resp_opt3_embeddings)).cpu().detach().numpy()
#    cosine_sim_question_resp_opt4 = util.pytorch_cos_sim(torch.tensor(question_embeddings), torch.tensor(resp_opt4_embeddings)).cpu().detach().numpy()
#    cosine_sim_question_resp_opt5 = util.pytorch_cos_sim(torch.tensor(question_embeddings), torch.tensor(resp_opt5_embeddings)).cpu().detach().numpy()
#    cosine_sim_question_resp_opt6 = util.pytorch_cos_sim(torch.tensor(question_embeddings), torch.tensor(resp_opt6_embeddings)).cpu().detach().numpy()
#    cosine_sim_question_resp_opt7 = util.pytorch_cos_sim(torch.tensor(question_embeddings), torch.tensor(resp_opt7_embeddings)).cpu().detach().numpy()
#    
#    # concatenate the cosine similarity values to the embeddings
#    
#    sim = np.concatenate((cosine_sim_content_corr_rep,
#                        cosine_sim_content_question,
#                        cosine_sim_content_resp_opt1,
#                        cosine_sim_content_resp_opt2,
#                        cosine_sim_content_resp_opt3,
#                        cosine_sim_content_resp_opt4,
#                        cosine_sim_content_resp_opt5,
#                        cosine_sim_content_resp_opt6,
#                        cosine_sim_content_resp_opt7,
#                        cosine_sim_corr_rep_question,
#                        cosine_sim_corr_rep_resp_opt1,
#                        cosine_sim_corr_rep_resp_opt2,
#                        cosine_sim_corr_rep_resp_opt3,
#                        cosine_sim_corr_rep_resp_opt4,
#                        cosine_sim_corr_rep_resp_opt5,
#                        cosine_sim_corr_rep_resp_opt6,
#                        cosine_sim_corr_rep_resp_opt7,
#                        cosine_sim_question_resp_opt1,
#                        cosine_sim_question_resp_opt2,
#                        cosine_sim_question_resp_opt3,
#                        cosine_sim_question_resp_opt4,
#                        cosine_sim_question_resp_opt5,
#                        cosine_sim_question_resp_opt6,
#                        cosine_sim_question_resp_opt7), axis=1)