# Research-Project
This project includes our research on "Modelling Reading Comprehension with Machine Learning and Psychometrics"
# Timeline
18/11 paper reading
# Important Notes
## Jenny 

--
## Sarthak
- Explore Word2vec
- Explore NLTK
Rank	Model Name	Framework	Type	Notes
1	paraphrase-multilingual-MiniLM-L12-v2	Sentence Transformers	Transformer	Excellent for multilingual tasks, optimized for sentence embeddings, very efficient and accurate.
2	distiluse-base-multilingual-cased-v2	Sentence Transformers	Transformer	High-quality multilingual embeddings, also works well for German, good balance of speed and accuracy.
3	dbmdz/bert-base-german-cased	Hugging Face	Transformer (BERT)	German-specific BERT model, strong performance on German language tasks, captures German nuances well.
4	de_trf_bertbasegerman_cased_lg (spaCy)	spaCy	Transformer	Based on dbmdz/bert-base-german-cased, integrated with spaCy pipeline, high-quality German embeddings.
5	xlm-roberta-large	Hugging Face	Multilingual Transformer	Strong multilingual performance, including German, but resource-intensive due to model size.
6	Flair Stacked Embeddings (de-forward, de-backward, BERT)	Flair	Recurrent + Transformer	Good for German-specific tasks, captures sentence-level context well but slower than transformer models.
7	bert-base-multilingual-cased	Hugging Face	Multilingual Transformer (BERT)	General-purpose multilingual BERT, decent performance for German but not optimized for embeddings.
8	LASER	Facebook	LSTM-based multilingual	Language-agnostic, works well for multilingual/cross-lingual, not as nuanced as transformer models.
9	cc.de.300.vec (FastText)	FastText	Word embeddings	Lightweight, fast, but lacks contextuality, performs worse on complex German semantics.
10	Word2Vec/Doc2Vec (Gensim)	Gensim	Word/Document embeddings	Customizable but generally lower performance, lacks contextual information, requires large corpus.
