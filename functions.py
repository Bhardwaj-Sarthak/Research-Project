#All functions to features extraction
#%%
import pandas as pd
import numpy as np
import spacy
import nltk
import pyphen
import evaluate
import torch
import german_compound_splitter

nlp = spacy.load("de_core_news_sm")
#%%
#Features based on the morphological analyzer of spicy
#subjunctive sentence are more difficult to understand than indicative sentences
#some German cases are more difficult to understand than others, e.g., the genitive is often named as difficult to understand therefore often replaced by the dative

def is_subjunctive(tokens):
    """
    tokens: list of spacy.Token objects

    returns 1 if a part of the sentence is subjunctive, 0 if not.
    """
    for token in tokens:
        if "Mood=Sub" in token.morph:
            return 1
    return 0


def ratio_case(tokens):
    """
    tokens: list of spacy.Token objects

    returns ratio of nouns in all four cases.
    """
    num_nouns = 0
    num_nom = 0
    num_gen = 0
    num_dat = 0
    num_acc = 0
    for token in tokens:
        if token.pos_ == "NOUN":
            num_nouns += 1
            if "Case=Nom" in token.morph:
                num_nom += 1
            elif "Case=Gen" in token.morph:
                num_gen += 1
            elif "Case=Dat" in token.morph:
                num_dat += 1
            elif "Case=Acc" in token.morph:
                num_acc += 1
    if num_nouns == 0:
        return 0, 0, 0, 0
    F_ratio_nom = round(num_nom/num_nouns,6)
    F_ratio_gen = round(num_gen/num_nouns,6)
    F_ratio_dat = round(num_dat/num_nouns,6)
    F_ratio_acc = round(num_acc/num_nouns,6)

    return F_ratio_nom, F_ratio_gen, F_ratio_dat, F_ratio_acc


#Feature based on Dependency Tree Distance
#Words which are discontinuously connected in a sentence are more difficult to understand because the reader need to memorize more elements of the sentence to combine the meaning.

def distance_between_words(tokens):
    """
    tokens: list of spacy.Token objects
    calculate the average and the maximum distance between nodes in the dependency tree
    
    return average and max distance value
    """
    max_distance = 0
    list_distances = list()
    for token in tokens:
        distance = abs(token.i-token.head.i)
        list_distances.append(distance)
        if distance > max_distance:
            max_distance = distance
    if len(list_distances) > 0:
        return round(sum(list_distances)/len(list_distances),6), round(max_distance/len(tokens),6)
    else:
        return 0, 0
    
def distance_between_verb_particles(tokens):
    """
    tokens: list of spacy.Token objects
    calculate the average and the maximum distance between verbs and particle verbs in the dependency tree
    
    return average and max distance value
    """
    max_distance = 0
    list_distances = list()
    for token in tokens:
        if token.tag_ == "PTKVZ":
            distance = abs(token.i-token.head.i)
            list_distances.append(distance)
            if distance > max_distance:
                max_distance = distance
    if len(list_distances) > 0:
        return round((sum(list_distances)/len(list_distances))/len(tokens),6), round(max_distance/len(tokens),6)
    else:
        return 0, 0


#Features based on Verb-Noun-Ratio
#the more verbs in a sentece, the better to understand the sentence
#the more nouns in a sentence, the less to understand the sentence
#the more verbs per nouns in a sentence, the better to understand the sentence

def verb_noun_ratio(tokens):
    """
    tokens: list of Spacy.Token objects
    calcualtes the ratio from verbs to nouns. 
    """
    n_nouns = 0
    n_verbs = 0
    for token in tokens:
        if token.pos_ == "NOUN":
            n_nouns += 1
        elif token.pos_ == "VERB":
            n_verbs += 1
    if n_nouns == 0:
        return 0
    else:
        return round(n_verbs/n_nouns,6)




#%%
#Eucledian distance and cosine similarity as defined by stepahnek 
def euclidean_distance_from_tokens(tokensA, tokensB):
    """
    Calculate the Euclidean distance between two texts based on their tokens.

    Input:
        tokensA (list): List of tokens from text A.
        tokensB (list): List of tokens from text B.

    Returns:
        float: The Euclidean distance between the token vectors of the two texts.
    """
    # Create the union of all unique tokens from both texts
    all_tokens = sorted(set(tokensA).union(set(tokensB)))
    
    # Define the vectors tA and tB
    tA = np.array([1 if token in tokensA else 0 for token in all_tokens])
    tB = np.array([1 if token in tokensB else 0 for token in all_tokens])
    
    # Compute the Euclidean distance
    distance_euc = np.sqrt(np.sum((tA - tB) ** 2))
    
    # Compute cosine similarity
    # Calculate the dot product
    dot_product = np.dot(tA, tB)
    
    # Calculate the magnitude (norm) of each vector
    magnitude_tA = np.sqrt(np.sum(tA ** 2))
    magnitude_tB = np.sqrt(np.sum(tB ** 2))
    
    # Avoid division by zero
    if magnitude_tA == 0 or magnitude_tB == 0:
        return 0.0
    
    # Calculate cosine similarity
    cosine_sim = dot_product / (magnitude_tA * magnitude_tB)
    return distance_euc, cosine_sim


import spacy

# Load the German language model
nlp = spacy.load("de_core_news_sm")
# Initialize the Pyphen object for German language
pyphen_german = pyphen.Pyphen(lang='de_CH')

def text_length_statistics(tokens):
    if not isinstance(tokens, spacy.tokens.doc.Doc):
        raise ValueError("Input must be a spaCy Doc object")

    # Filter out punctuation from tokens
    words = [token for token in tokens if not token.is_punct]
    
    # Extract sentences (excluding punctuation)
    sentences = list(tokens.sents)
    
    # Sentence lengths (ignoring punctuation)
    sentence_lengths = [len([word for word in sentence if not word.is_punct]) for sentence in sentences]

    # Total number of words (excluding punctuation)
    total_words = len(words)

    # Total number of characters (excluding punctuation)
    total_characters = sum(len(token.text) for token in words)

    # Average word length
    avg_word_length = total_characters / total_words if total_words > 0 else 0

    # Average sentence length (ignoring punctuation)
    avg_sentence_length = total_words / len(sentences) if sentences else 0

    # Maximum sentence length
    max_sentence_length = max(sentence_lengths) if sentence_lengths else 0

    # Count syllables for each word (token)
    total_syllables = sum(len(pyphen_german.inserted(token.text).split('-')) for token in words)

    # Average syllable length
    avg_syll_length = total_syllables / total_words if total_words > 0 else 0

    return {
        "Total Words": total_words,
        "Total Characters": total_characters,
        "Average Word Length": avg_word_length,
        "Average Sentence Length": avg_sentence_length,
        "Maximum Sentence Length": max_sentence_length,
        "Average syllables lenght" : avg_syll_length
    }
     
#%%
# Using wortsalat to get the text statistics
#The Wiener Sachtextformel (WSTF) is a readability measure specifically designed for German texts. 
# It accounts for: The proportion of long words, Sentence length, The complexity of sentences.
import wortsalat
from wortsalat import analyze_wortsalat
#from wortsalat.analyze_wortsalat import analyze_wortsalat

def text_statistics(text):
    """
    Extract Wiener-Sachtextformel score

    Input: The entire text to analyze.
    """
    if not text.strip():  # Check if the text is empty or contains only whitespace
        return None  # Or return a default value, e.g., 0

    wiener_sachtextformel = 0
    try:
        wiener_sachtextformel = wortsalat.analyze_wortsalat.calculate_wiener_sachtextformel(text)
    except ZeroDivisionError:
        pass  # Handle the case where n_words is 0

    return wiener_sachtextformel
   
#%%
# Fox Index, as defined in Stepanek
#using phypen that also has german switzerland as language as no specif pakcgag for syllabus was found

def count_syllables_german(word):
    """Return the syllable count for a German word using pyphen."""
    if not isinstance(word, str):  # Convert spaCy Token to string if needed
        word = word.text
    syllables = pyphen_german.inserted(word).split('-')
    return len(syllables)


def fog_index_german(tokens):
    """
    Calculate the FOG and SMOG index for a vector of tokenized German text.
    
    Parameters:
        tokens (spacy.tokens.Doc): Tokenized text as a spaCy Doc object.
    
    Returns:
        tuple: FOG index and SMOG index of the text.
    """
    # Split the tokens into sentences
    sentences = list(tokens.sents)
    
    # Number of sentences
    num_sentences = len(sentences)
    
    # Total number of words (tokens)
    num_words = len(tokens)
    
    # Count words with 3 or more syllables (convert each Token to string)
    nwords_with_3_or_more_syllables = sum(1 for word in tokens if count_syllables_german(word.text) >= 3)
    
    # Compute average number of words per sentence (w)
    w = num_words / num_sentences if num_sentences > 0 else 0
    
    # Compute the FOG and SMOG index using the formula
    if num_words > 0:
        fog = 0.4 * (w + (100 * nwords_with_3_or_more_syllables) / num_words)
        smog = (
            1.043 * np.sqrt(nwords_with_3_or_more_syllables * (30 / num_sentences)) + 3.129
            if num_sentences > 0
            else 0
        )
    else:
        fog = 0.0
        smog = 0.0
    
    return fog, smog



# %%
