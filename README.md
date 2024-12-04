# Research-Project
This project includes our research on "Modelling Reading Comprehension with Machine Learning and Psychometrics"
## Data Disc

- *InternCode*: An internal code for linking each item to additional information. In this project, this serves as an ID for each data point.
- *Title*: The title of the text.
- *Text*: The content of the text (in German).
- *Item Discrimination*: The item discrimination parameter, also known as the factor loading. This represents the correlation between a latent factor (the underlying trait being measured) and the response to this item.
- *Item Difficulty*: The item difficulty parameter. Positive values indicate a more difficult item, while negative values denote an easier one.
- *Question*: The question (in German) posed to the participants. (Please note that this file does not yet include all response options; I’ve requested an updated version, so please disregard this for now.)
- *Correct Response*: The correct response option(s).
- *Item Type*: This specifies whether the item is a single-choice question (one correct answer), a multiple-choice question (one or more correct answers), or an ordering task, where response options must be sequenced correctly. Responses are counted as correct only if all selected answers are accurate.

# Timeline
- 13/11-20/11: Receiving data + Basic concepts literature review
- 20/11-4/12: In depth literature review about embeddings and models
- 4/12-18/12: Implementation on python of different embeddings techniques
- 18/12-29/01: Implementation on python of different models to estimate  ~ item difficulty (eventually also item discrimination)
- 29/01-7/02: Finalizing the project and final presentation
- 7/02-31/03: Writing final report

# Important Notes
## Jenny 

--
## Sarthak
- Explore Word2vec
- Explore NLTK
- Interesting Papers to read
  -- https://arxiv.org/pdf/2401.02709
-The following categories are the most relevant:

  --Semantic Textual Similarity (STS):
    ---Most directly aligns with our goal of measuring semantic distance between text pairs.
  
  --Retrieval:
    ---Useful for tasks like ranking multiple responses for a given question or retrieving the most relevant question for a paragraph.
  
  --Reranking:
    ---multiple candidate responses to a question and want to rank them by relevance.
  
  --Pair Classification:
    ---frame the task as a binary or multi-class classification problem (e.g., correct vs. incorrect response).

# German Text Embedding Models


Best performing (MTEB) 
- sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- xlm-r-distilroberta-base-paraphrase-v1
- sentence-transformers/distiluse-base-multilingual-cased-v1
*test them more thoroughly*

pre-trained models that are well-suited for handling German text.

### 1. **Multilingual BERT (mBERT)**
Multilingual BERT is trained on 104 languages, including German. It can be used for a variety of NLP tasks and has good performance across multiple languages.

- **Pros**: Supports multiple languages, widely used and tested.
- **Cons**: May not be as fine-tuned for German as other dedicated models.

### 2. **XLM-RoBERTa**
XLM-RoBERTa is a transformer model trained on 100 languages, including German. It generally outperforms mBERT in many tasks due to its larger training dataset and improved architecture.

- **Pros**: High performance on multilingual tasks, including German.
- **Cons**: Requires more computational resources.

### 3. **German BERT (bert-base-german-cased)**
This is a BERT model specifically trained on German text. It may offer better performance on German-specific tasks compared to multilingual models.

- **Pros**: Fine-tuned for the German language, good for German-specific tasks.
- **Cons**: Limited to German language only.

### 4. **GermEval Models**
These models are specifically fine-tuned for German tasks and datasets, such as GermEval. They might provide more accurate embeddings for German texts.

- **Pros**: Specifically tailored for German language and tasks.
- **Cons**: Limited to German.

### 5. **SBERT (Sentence-BERT)**
SBERT is a modification of BERT that is fine-tuned to produce more semantically meaningful sentence embeddings. There are multilingual versions of SBERT, such as `xlm-r-distilroberta-base-paraphrase-v1`, which can be used for German.

- **Pros**: Produces sentence embeddings directly, efficient for measuring distances between texts.
- **Cons**: Requires more fine-tuning for specific tasks.

A ranked list of open-source models for generating text embeddings for German text. The models are ranked from best to worst in terms of average performance, with a focus on semantic tasks such as similarity, clustering, and classification.

## Model Rankings

~ hugging face language model leaderboard link - https://huggingface.co/spaces/mteb/leaderboard

| Rank | Model Name                                         | Framework            | Type                          | Notes                                                                                                  |
|------|----------------------------------------------------|----------------------|-------------------------------|--------------------------------------------------------------------------------------------------------|
| 1    | **paraphrase-multilingual-MiniLM-L12-v2**          | Sentence Transformers| Transformer                   | Excellent for multilingual tasks, optimized for sentence embeddings, very efficient and accurate.      |
| 2    | **distiluse-base-multilingual-cased-v2**           | Sentence Transformers| Transformer                   | High-quality multilingual embeddings, also works well for German, good balance of speed and accuracy.  |
| 3    | **dbmdz/bert-base-german-cased**                   | Hugging Face         | Transformer (BERT)            | German-specific BERT model, strong performance on German language tasks, captures German nuances well. |
| 4    | **de_trf_bertbasegerman_cased_lg** (spaCy)         | spaCy                | Transformer                   | Based on `dbmdz/bert-base-german-cased`, integrated with spaCy pipeline, high-quality German embeddings.|
| 5    | **xlm-roberta-large**                              | Hugging Face         | Multilingual Transformer      | Strong multilingual performance, including German, but resource-intensive due to model size.           |
| 6    | **Flair Stacked Embeddings (de-forward, de-backward, BERT)** | Flair         | Recurrent + Transformer       | Good for German-specific tasks, captures sentence-level context well but slower than transformer models. |
| 7    | **bert-base-multilingual-cased**                   | Hugging Face         | Multilingual Transformer (BERT)| General-purpose multilingual BERT, decent performance for German but not optimized for embeddings.     |
| 8    | **LASER**                                          | Facebook             | LSTM-based multilingual       | Language-agnostic, works well for multilingual/cross-lingual, not as nuanced as transformer models.    |
| 9    | **cc.de.300.vec** (FastText)                       | FastText             | Word embeddings               | Lightweight, fast, but lacks contextuality, performs worse on complex German semantics.                |
| 10   | **Word2Vec/Doc2Vec (Gensim)**                      | Gensim               | Word/Document embeddings      | Customizable but generally lower performance, lacks contextual information, requires large corpus.     |

## Explanation of the Ranking (made by GPT)

- **Top Performers**: 
  - `paraphrase-multilingual-MiniLM-L12-v2` and `distiluse-base-multilingual-cased-v2` are optimized for generating sentence embeddings and perform consistently well across multiple languages, including German. They are efficient and handle complex language nuances effectively.
  
- **German-Specific Models**:
  - `dbmdz/bert-base-german-cased` is trained specifically on German text and captures German language nuances better than general multilingual models.
  - `de_trf_bertbasegerman_cased_lg` from spaCy uses the same model and integrates well with the spaCy pipeline, making it a good choice for applications within the spaCy ecosystem.
  
- **General Multilingual Models**:
  - `xlm-roberta-large` provides strong multilingual embeddings, although it is more resource-intensive and may not be as efficient for real-time applications.
  - `bert-base-multilingual-cased` is decent but less specialized for sentence embeddings and German-specific nuances.

- **Flair Stacked Embeddings**:
  - Flair embeddings are powerful for German text, especially with stacked forward/backward embeddings, but are slower than transformer-based models.

- **Lower Contextual Models**:
  - `LASER` is language-agnostic and effective for cross-lingual tasks, but generally does not perform as well as more contextualized transformer models for German.
  - FastText (`cc.de.300.vec`) provides quick and simple embeddings but lacks contextual understanding, which is often crucial for German language tasks.
  - Gensim Word2Vec/Doc2Vec can be useful if you have a large German corpus and specific customization needs, but in general, these models underperform compared to more modern contextual embeddings.



