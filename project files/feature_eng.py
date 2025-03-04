import spacy
import pandas as pd
import functions as f
nlp = spacy.load("de_core_news_sm")

# Processing function
def process_row(row, data):
    """
    Process a single row to calculate all features for the text, title, and questions.
    """
    results = {}
    excluded_columns = ["InternCode", "Title", "Item Type", "Item Discrimination", "Item Difficulty"]
    response_columns = [col for col in data.columns if col.startswith("Response Option")]

    for col in data.columns:
        if col not in excluded_columns:
            text = row[col]
            if not text or not text.strip():  # Skip if the text is None or empty
                continue
            tokens = nlp(text)

            # Compute all features
            ratio_case = f.ratio_case(tokens)
            distance_between_words = f.distance_between_words(tokens)
            distance_between_verb_particles = f.distance_between_verb_particles(tokens)
            text_length_stats = f.text_length_statistics(tokens)
            fog_index = f.fog_index_german(tokens)

            features = {
                f"{col}_Subjunctive_Sentence": f.is_subjunctive(tokens),
                f"{col}_Noun_Ration_Nominative": ratio_case[0],
                f"{col}_Noun_Ration_Genetive": ratio_case[1],
                f"{col}_Noun_Ration_Dative": ratio_case[2],
                f"{col}_Noun_Ration_Accusative": ratio_case[3],
                f"{col}_Average_Dependency_Tree_Distance": distance_between_words[0],
                f"{col}_Max_Dependency_Tree_Distance": distance_between_words[1],
                f"{col}_Average_Verbs_Particles_Distance": distance_between_verb_particles[0],
                f"{col}_Max_Verbs_Particles_Distance": distance_between_verb_particles[1],
                f"{col}_Ratio_Verbs_to_Nouns": f.verb_noun_ratio(tokens),
                f"{col}_Total_Words": text_length_stats["Total Words"],
                f"{col}_Total_Characters": text_length_stats["Total Characters"],
                f"{col}_Average_Word_Length": text_length_stats["Average Word Length"],
                f"{col}_Average_Sentence_Length": text_length_stats["Average Sentence Length"],
                f"{col}_Maximum_Sentence_Length": text_length_stats["Maximum Sentence Length"],
                f"{col}_Average_Syllables_Length": text_length_stats["Average syllables lenght"],
                #f"{col}_Wiener_Cachtextformel": f.text_statistics(text),
                f"{col}_FOG_Index": fog_index[0],
                f"{col}_SMOG_Index": fog_index[1]
            }
            results.update(features)

           # Compute cosine similarity and Euclidean distance for each response option
            for response_col in response_columns:
                response_text = row[response_col]
                if response_text.strip():  # Only compute if response text is not empty
                    cosine_similarity, euclidean_distance = f.euclidean_distance_from_tokens(row['Content'], response_text)
                else:
                    cosine_similarity, euclidean_distance = None, None

                # Add both values as separate columns
                results[f"{col}_Cosine_Similarity_with_{response_col}"] = cosine_similarity
                results[f"{col}_Euclidean_Distance_with_{response_col}"] = euclidean_distance


    return results


