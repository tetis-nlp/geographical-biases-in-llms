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


def plot_correlation_scatter(geo_distances, semantic_distances, model_name, continent):
    plt.figure(figsize=(8, 6))
    plt.scatter(geo_distances.flatten(), semantic_distances.flatten(), color='green', alpha=0.5)
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(geo_distances.flatten(), semantic_distances.flatten())
    line = slope * geo_distances.flatten() + intercept
    plt.plot(geo_distances.flatten(), line, color='red', label=f'Regression (R2={r_value**2:.2f})')
    plt.text(0.05, 0.95, f'R2 = {r_value**2:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    # plt.title(f"Correlation between geo and semantic distance with {model_name} for {continent}")
    plt.xlabel("Geo Distance")
    plt.ylabel("Semantic Distance")
    plt.grid(True)
    plt.savefig(f"./figs/correlation_dgeo_dsem_correlation_{model_name.replace('/', '_')}_{continent}.png", bbox_inches='tight')
    #plt.show()
    return r_value**2

if __name__ == "__main__":
    df = pd.read_csv("data/geonames-all-cities-with-a-population-1000.csv", sep=";")
    df["city"] = df["ASCII Name"]
    df["city"] = df["city"].astype(str)
    df['Alternate Names'] = df['Alternate Names'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
    # Work only on city with Population is 100k of inhabitants
    # df = df[df['Population'] > 1000000].reset_index(drop=True)
    df["continent"] = df["Country Code"].apply(country_to_continent)
    df_global = df.copy()
    
    df_model_r_squared = pd.DataFrame()
    for model_name in list_of_models:
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
        
        continent_r_squared = []
        for continent in df["continent"].unique():
            df = df[df["continent"] == continent]
            df = df.nlargest(50, 'Population')
            df["last_hidden_states"] = df["city"].apply(word_embedding)
            if model_name != "openai/ada":
                df["embedding"] = df["last_hidden_states"].apply(tensor_to_array)
            else:
                df["embedding"] = df["last_hidden_states"]
            df.to_csv(f"./output/correlation_embedding_{model_name.split('/')[0]}_{continent}.csv")
            embedding_array = np.stack(df["embedding"].values)
            semantic_distance_matrix = 1 - cosine_similarity(embedding_array, embedding_array)
            geo_distance_matrix = compute_geo_distance(df)
            r_squared = plot_correlation_scatter(geo_distance_matrix, semantic_distance_matrix, model_name, continent)
            continent_r_squared.append({continent: r_squared})
            df = df_global.copy()
        df_continent = pd.DataFrame(continent_r_squared)
        df_continent["model"] = model_name
        df_model_r_squared = pd.concat([df_model_r_squared, df_continent], ignore_index=True)

    df_model_r_squared.reset_index(drop = True)
    df_model_r_squared.set_index('model', inplace=True)
    df_model_r_squared = df_model_r_squared.groupby('model').sum()
    df_model_r_squared.to_csv('./output/model_r_squared.csv')
    df_model_r_squared = df_model_r_squared.round(2)
    print(df_model_r_squared)
    print("end")
