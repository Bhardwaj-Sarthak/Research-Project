#%%
import pandas as pd
data = pd.read_csv('C:/Users/Utente/Documents/Research project/Data&Code/Selection_Items_Tuebingen.csv', encoding='ISO-8859-1', sep =';')
data.head()
#print(data.shape) #118,15

#%%
# Data preprocessing 

#counting nans for each column
nan_values_count = data.isnull().sum()
#print(nan_values_count)

#counting unique response type
response_type = data["Item Type"].unique()
#print('The item type are:', response_type)

#create three different dataset based on the item type 
single_choice=data[data["Item Type"]=="single choice"]
#single_choice.head()

#counting Nan
nan_values_count = single_choice.isnull().sum()
#print(nan_values_count)
#option 5,6,7 all NaN, drop the columns
single_choice.drop(['Response Option 5', 'Response Option 6', 'Response Option 7'], axis=1)

multiple_choice=data[data["Item Type"]=="multiple response"]
#multiple_choice.head()
#print(multiple_choice.shape)

#counting Nan
nan_values_count = multiple_choice.isnull().sum()
#print(nan_values_count)

#option 7 all NaN, drop the columns
multiple_choice.drop([ 'Response Option 7'], axis=1)

order=data[data["Item Type"]=="order"]
#order.head()
#print(order.shape)
#counting Nan
nan_values_count = order.isnull().sum()
#print(nan_values_count)

#creating dataset with all uniques texts (24 unique texts)
df_unique = data.drop_duplicates(subset='Title') 
#df_unique.head()

#%%
import spacy
import functions as f
import feature_eng as fg
nlp = spacy.load("de_core_news_sm")

# Apply processing for each row
df_text = data.fillna("")  # Replace NaN with empty string
results = df_text.apply(lambda row: fg.process_row(row, df_text), axis=1)

# Convert results to DataFrame
results_df = pd.DataFrame(results.tolist())

# Combine results with the original dataset
final_df = pd.concat([df_text, results_df], axis=1)

print(final_df)


# %%
# Fill NaN values with a specific value
df_filled = final_df.fillna('Missing')
#print(df_filled)

# %%
# Save the final DataFrame to a CSV file
df_filled.to_csv("final_results.csv", index=False, encoding='ISO-8859-1')

# %%
