from transformers import BertModel, BertTokenizer
from transformers import RobertaTokenizer, RobertaModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers import CamembertModel, CamembertTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from geopy.distance import geodesic
import numpy as np
import pycountry_convert as pc
from scipy.stats import linregress
# from transformers import LlamaModel, LlamaConfig
from apikey import HF_API_TOKEN, OPENAI_API_KEY
from torch import cuda
import openai
from langchain.embeddings import OpenAIEmbeddings
# from transformers import LlamaForCausalLM, LlamaTokenizer

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'; print(device)

list_of_nb_cities = [1, 3, 5]

list_of_models = {
    'bert': {
        'name': 'bert-base-uncased',
        'tokenizer': BertTokenizer.from_pretrained('bert-base-uncased'),
        'model': BertModel.from_pretrained('bert-base-uncased')
    },
    'bert-base-multilingual-uncased':{
        'name': 'bert-base-multilingual-uncased',
        'tokenizer': AutoTokenizer.from_pretrained('bert-base-multilingual-uncased'),
        'model': BertModel.from_pretrained('bert-base-multilingual-uncased')
    },
    'roberta': {
        'name': 'roberta-base',
        'tokenizer': AutoTokenizer.from_pretrained('roberta-base'),
        'model': RobertaModel.from_pretrained('roberta-base')
    },
    # 'geolm': {
    #     'name': 'zekun-li/geolm-base-cased',
    #     'tokenizer': AutoTokenizer.from_pretrained('zekun-li/geolm-base-cased'),
    #     'model': RobertaModel.from_pretrained('zekun-li/geolm-base-cased'),
    #     'mask': "<mask>"
    # },
    'xlm-roberta-base': {
        'name': 'xlm-roberta-base',
        'tokenizer': AutoTokenizer.from_pretrained('xlm-roberta-base'),
        'model': RobertaModel.from_pretrained('xlm-roberta-base'),
        'mask': "<mask>"
    },
    'llama2': {
        'name': 'meta-llama/Llama-2-7b-chat-hf',
        'tokenizer': AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf"),
        'model': AutoModel.from_pretrained("meta-llama/Llama-2-7b-chat-hf"),
    },
    'llama3': {
        'name': 'meta-llama/Meta-Llama-3-8B-Instruct',
        'type': 'llm',
        'local': './models/llama3/Meta-Llama-3-8B-Instruct',
        'tokenizer': AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct"),
        'model': AutoModel.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct"),
     },
    'mistral': {
        'name': 'mistralai/Mistral-7B-Instruct-v0.1',
        'tokenizer': AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.1'),
        'model': AutoModel.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    },
    'mistral-v03': {
        'name': 'mistralai/Mistral-7B-Instruct-v0.3',
        'type': 'llm',
        'local': './models/mistralai/Mistral-7B-Instruct-v0.3',
        'tokenizer': AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3'),
        'model': AutoModel.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    },
    'openai/ada': {
        'name': "text-embedding-ada-002",
        'tokenizer': 'cl100k_base',
        'model': OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    },
}

def haversine_distance(coord1, coord2):
    """
    Calculate the Haversine distance between two points on the earth.
    """
    return geodesic(coord1, coord2).kilometers

def get_n_closest_cities(df, city, n):
    """
    Retrieve the n closest cities based on their embeddings and calculate geographic distance.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'city', 'embedding', 'country', 'continent', and 'Coordinates' columns
    city (str): The name of the city to find closest cities for
    n (int): Number of closest cities to retrieve

    Returns:
    pd.DataFrame: DataFrame with the n closest cities, their similarity scores, and geographic distances
    """
    # Ensure the city is in the dataframe
    if city not in df['city'].values:
        raise ValueError(f"City '{city}' not found in the DataFrame.")

    # Get the embedding for the specified city
    city_row = df[df['city'] == city]
    city_embedding = city_row['embedding'].values[0].reshape(1, -1)
    city_coords = city_row['Coordinates'].values[0]
    
    # Compute cosine similarities
    embeddings = np.vstack(df['embedding'].values)
    similarities = cosine_similarity(city_embedding, embeddings).flatten()

    # Add similarities to the dataframe
    df['similarity'] = similarities

    # Calculate geographic distance: commented to reduce compute time
    # df['geo_distance'] = df['Coordinates'].apply(lambda coords: haversine_distance(city_coords, coords))
    df['geo_distance'] = np.NaN

    # Sort the dataframe by similarity in descending order and exclude the city itself
    closest_cities = df[df['city'] != city].sort_values(by='similarity', ascending=False).head(n)

    return closest_cities[['city', 'country', 'continent', 'similarity', 'geo_distance']]

def get_all_closest_cities(df, n):
    """
    Apply the get_n_closest_cities function to the whole DataFrame to find the n closest cities for each city.

    Parameters:
    df (pd.DataFrame): DataFrame containing 'city', 'embedding', 'country', 'continent', and 'Coordinates' columns
    n (int): Number of closest cities to retrieve for each city

    Returns:
    pd.DataFrame: DataFrame with columns 'city', 'country', 'continent', 'closest_city', 'closest_country', 'closest_continent', 'similarity', 'geo_distance'
    """
    all_closest_cities = []

    for city in df['city']:
        closest_cities_df = get_n_closest_cities(df, city, n)
        original_city_info = df[df['city'] == city][['city', 'country', 'continent', 'Coordinates', 'Population']].iloc[0]
        for index, row in closest_cities_df.iterrows():
            all_closest_cities.append({
                'city': original_city_info['city'],
                'country': original_city_info['country'],
                'continent': original_city_info['continent'],
                'Coordinates': original_city_info['Coordinates'],
                'Population': original_city_info['Population'],
                'closest_city': row['city'],
                'closest_country': row['country'],
                'closest_continent': row['continent'],
                'similarity': row['similarity'],
                'geo_distance': row['geo_distance']
            })

    return pd.DataFrame(all_closest_cities)


def country_to_continent(country_alpha2):
    #country_alpha2 = pc.country_name_to_country_alpha2(country_name)
    try:
        country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
        country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
    except:
        country_continent_name = "unknown"
    return country_continent_name

def tensor_to_array(embedding):
    return embedding.numpy()

def compute_geo_distance(df):
    coordinates = df["Coordinates"].tolist()
    num_coordinates = len(coordinates)

    # Create an empty distance matrix
    distance_matrix = np.zeros((num_coordinates, num_coordinates))

    # Calculate distances and populate the distance matrix
    for i in range(num_coordinates):
        for j in range(i + 1, num_coordinates):
            coord1 = coordinates[i]
            coord2 = coordinates[j]
            distance = geodesic(coord1, coord2).kilometers
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance  # Since the distance matrix is symmetric
    return distance_matrix

def calculate_same_country_percentage_by_continent(df):
    results = df.groupby('continent').apply(lambda x: (x['country'] == x['closest_country']).mean() * 100)
    results_df = results.reset_index().rename(columns={0: 'same_country_percentage'})
    return results_df

df_all_models = pd.DataFrame()
if __name__ == "__main__":
    df = pd.read_csv("data/geonames-all-cities-with-a-population-1000.csv", sep=";")
    df["city"] = df["ASCII Name"]
    df["city"] = df["city"].astype(str)
    df['Alternate Names'] = df['Alternate Names'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
    # Work only on city with Population is 100k of inhabitants
    df = df[df['Population'] > 100000].reset_index(drop=True)
    df["country"] = df["Country name EN"]
    df["continent"] = df["Country Code"].apply(country_to_continent)
    df_global = df.copy()
    
    for i, model_name in enumerate(list_of_models):
        print(f"Dealing with {model_name}")
        if model_name != "openai/ada":
            tokenizer = list_of_models[model_name]["tokenizer"]
            model = list_of_models[model_name]["model"]

            def word_embedding(input_text):
                input_ids = tokenizer.encode(input_text, return_tensors="pt")
                # input_ids: [101, 2182, 2003, 2070, 3793, 2000, 4372, 16044, 102]
                with torch.no_grad():
                    last_hidden_states = model(input_ids).last_hidden_state
                # last_hidden_states = last_hidden_states.mean(1)
                # return last_hidden_states[:,1,:][0] # return embedding of the word
                return last_hidden_states.mean(dim=1)[0] # we mean for word chunked into subtoken (out of model vocabulary) and [CLS] & [SEP]
        else:
            openai.api_key = OPENAI_API_KEY
            model = list_of_models[model_name]["model"]
            def word_embedding(input_text):
                return np.array(model.embed_documents([input_text])[0])
        
        df["last_hidden_states"] = df["city"].apply(word_embedding)
        if model_name != "openai/ada":
                df["embedding"] = df["last_hidden_states"].apply(tensor_to_array)
        else:
                df["embedding"] = df["last_hidden_states"]

        list_same_country_percentage_by_continent = []
        for n in list_of_nb_cities:
            result_df = get_all_closest_cities(df, n)
            percentage_df = calculate_same_country_percentage_by_continent(result_df)
            percentage_df = percentage_df.rename(columns={'same_country_percentage': f"{model_name}_{str(n)}"})
            list_same_country_percentage_by_continent.append(percentage_df.set_index('continent'))
            result_df['same_country'] = result_df['country'] == result_df['closest_country']
            result_df['same_continent'] = result_df['continent'] == result_df['closest_continent']
            result_df.to_csv(f"./output/closest_cities_{model_name.split('/')[0]}_{n}")
            percentage_df = result_df.groupby(['city', 'country', 'continent', 'Coordinates', 'Population']).agg(
                same_country_pct=('same_country', 'mean'),
                same_continent_pct=('same_continent', 'mean')
            ).reset_index()
            percentage_df['same_country_pct'] = percentage_df['same_country_pct'] * 100
            percentage_df['same_continent_pct'] = percentage_df['same_continent_pct'] * 100
            percentage_df.to_csv(f"./output/closest_cities_percentage_{model_name.split('/')[0]}_{n}.csv")


        # Combine all the percentage dataframes into a single dataframe
        same_country_percentage_by_continent = pd.concat(list_same_country_percentage_by_continent, axis=1)
        continent_order = [ 'North America', 'South America', 'Europe', 'Africa', 'Asia', 'Oceania']
        same_country_percentage_by_continent.loc[continent_order]

        # df_all_models = pd.concat([df_all_models, same_country_percentage_by_continent], axis=0)
        if i == 0:
            df_all_models = same_country_percentage_by_continent
        else:
            df_all_models = pd.merge(df_all_models, same_country_percentage_by_continent, on="continent")
        df_all_models.to_csv('./output/closest_cities_embedding.csv')
        
    print("end")
