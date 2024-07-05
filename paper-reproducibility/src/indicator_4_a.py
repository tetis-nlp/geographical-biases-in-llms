import my_keys
from geopy.geocoders import Nominatim
import pandas as pd
import pycountry_convert as pc
import torch
from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf #For GeoBERT
from langchain.embeddings import OpenAIEmbeddings
from geopy.distance import geodesic
from scipy import stats
from mpl_toolkits.basemap import Basemap
import matplotlib.patches as mpatches
import geopandas as gpd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from transformers import LlamaModel, LlamaConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from apikey import HF_API_TOKEN
from torch import bfloat16
from transformers import BitsAndBytesConfig
from transformers import CamembertModel, CamembertTokenizer

# Load data: all cities all around the world with population greater than 1Million
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

def normalized_distance(data_array, min_bound=0, max_bound=1):
    """
    Normalize matrix betwen [min_bound, max_bound]. Default: between [0,1]
    """
    min_value = np.min(data_array)
    max_value = np.max(data_array)

    return min_bound + ((data_array - min_value) * (max_bound - min_bound)) / (max_value - min_value)

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

def compute_geo_distance_to_city(df):
    city = "Paris"
    city_coordinates = "48.85341, 2.3488"
    coordinates = df["Coordinates"].tolist()
    num_coordinates = len(coordinates)

    # Create an empty distance matrix
    distance_matrix = np.zeros(num_coordinates)

    # Calculate distances and populate the distance matrix
    for i in range(num_coordinates):
        coord1 = coordinates[i]
        distance = geodesic(coord1, city_coordinates).kilometers
        distance_matrix[i] = distance
    return distance_matrix

list_of_models = {
    'bert': {
        'name': 'bert-base-uncased',
        'tokenizer': BertTokenizer.from_pretrained('bert-base-uncased'),
        'model': BertModel.from_pretrained('bert-base-uncased')
    },
    'camembert':{
        'name': 'camembert-base',
        'tokenizer': AutoTokenizer.from_pretrained('camembert-base'),
        'model': CamembertModel.from_pretrained('camembert-base')
    },
    'bert-base-multilingual-uncased':{
        'name': 'bert-base-multilingual-uncased',
        'tokenizer': AutoTokenizer.from_pretrained('bert-base-multilingual-uncased'),
        'model': BertModel.from_pretrained('bert-base-multilingual-uncased')
    },
    'roberta': {
        'name': 'roberta-base',
        'tokenizer': RobertaTokenizer.from_pretrained('roberta-base'),
        'model': RobertaModel.from_pretrained('roberta-base')
    },
    'geolm': {
        'name': 'zekun-li/geolm-base-cased',
        'tokenizer': AutoTokenizer.from_pretrained('zekun-li/geolm-base-cased'),
        'model': RobertaModel.from_pretrained('zekun-li/geolm-base-cased'),
        'mask': "<mask>"
    },
    'xlm-roberta-base': {
        'name': 'xlm-roberta-base',
        'tokenizer': AutoTokenizer.from_pretrained('xlm-roberta-base'),
        'model': RobertaModel.from_pretrained('xlm-roberta-base'),
        'mask': "<mask>"
    },
    'llama2': {
        'name': 'meta-llama/Llama-2-7b-hf',
        'tokenizer': AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', use_auth_token=HF_API_TOKEN),
        'model': LlamaModel(LlamaConfig())
    },
    'mistral': {
        'name': 'mistralai/Mistral-7B-v0.1',
        'tokenizer': AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1'),
        'model': AutoModel.from_pretrained("mistralai/Mistral-7B-v0.1")
    },
    'openai/ada': {
        'name': "text-embedding-ada-002",
        'tokenizer': 'cl100k_base',
        'model': OpenAIEmbeddings(openai_api_key=my_keys.openai_api_key)
    },
}

continents_basemap = {
    "World": {"llcrnrlat": -90, "urcrnrlat": 90, "llcrnrlon": -180, "urcrnrlon": 180},
    "Africa": {"llcrnrlat": -35, "urcrnrlat": 38, "llcrnrlon": -25, "urcrnrlon": 60},
    "Antarctica": {"llcrnrlat": -90, "urcrnrlat": -60, "llcrnrlon": -180, "urcrnrlon": 180},
    "Asia": {"llcrnrlat": -10, "urcrnrlat": 70, "llcrnrlon": 3, "urcrnrlon": 160},
    "Europe": {"llcrnrlat": 30, "urcrnrlat": 75, "llcrnrlon": -25, "urcrnrlon": 50},
    "North America": {"llcrnrlat": 0, "urcrnrlat": 85, "llcrnrlon": -180, "urcrnrlon": -9}, # real america
    "Oceania": {"llcrnrlat": -50, "urcrnrlat": 0, "llcrnrlon": 100, "urcrnrlon": 180},
    "South America": {"llcrnrlat": -60, "urcrnrlat": 15, "llcrnrlon": -90, "urcrnrlon": -30},
}

# Want to take into account subtokens ?
allow_subtokens = False

df = pd.read_csv("data/geonames-all-cities-with-a-population-1000.csv", sep=";")
df["continent"] = df["Country Code"].apply(country_to_continent)
df["city"] = df["ASCII Name"]
df_global = df

# Extract the 3 more populated cities per country (229 countries)
df = df.groupby('Country name EN').apply(lambda x: x.nlargest(3, 'Population')).reset_index(drop=True)

# compute normalized geo distance
geo_distance_matrix = compute_geo_distance(df)
geo_distance_matrix_norm = normalized_distance(geo_distance_matrix)

# Compute semantic distance
boxplot_data_semantic_prox = []
boxplot_data_semantic_normalized_distance = []
df_aggregated_ratio_by_country_by_models = []
for i, model_name in enumerate(list_of_models):
    print(f"Model: {model_name}")
    # if model_name in ['bert', 'roberta']:
    if model_name != "openai/ada":
        tokenizer = list_of_models[model_name]["tokenizer"]
        model = list_of_models[model_name]["model"]

        def word_embedding(input_text):
            # if not allow_subtokens:
            #     if input_text not in tokenizer.vocab:
            #         return torch.tensor([float('nan')])

            input_ids = tokenizer.encode(input_text, return_tensors="pt")
            # input_ids: [101, 2182, 2003, 2070, 3793, 2000, 4372, 16044, 102]
            with torch.no_grad():
                last_hidden_states = model(input_ids).last_hidden_state
            # last_hidden_states = last_hidden_states.mean(1)
            # return last_hidden_states[:,1,:][0] # return embedding of the word
            return last_hidden_states.mean(dim=1)[0] # we mean for word chunked into subtoken (out of model vocabulary) and [CLS] & [SEP]
        
        def bert_cosine_similarity_word_pairwise(token1, token2):
            embedding = word_embedding(token1)
            embedding2 = word_embedding(token2)
            return cosine_similarity(embedding, embedding2)
        df["last_hidden_states"] = df["city"].apply(word_embedding)
        df["embedding"] = df["last_hidden_states"].apply(tensor_to_array)
        embedding_array = np.stack(df["embedding"].values)
    elif model_name == "openai/ada":
        model = list_of_models[model_name]["model"]
        embedding_array = np.array(model.embed_documents(df["city"].values))
    # else:
    #     # https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1
    #     bnb_config = BitsAndBytesConfig(
    #         load_in_4bit=True,  # 4-bit quantization
    #         bnb_4bit_quant_type='nf4',  # Normalized float 4
    #         bnb_4bit_use_double_quant=True,  # Second quantization after the first
    #         bnb_4bit_compute_dtype=bfloat16  # Computation type
    #     )

    #     model = AutoModelForCausalLM.from_pretrained(
    #         list_of_models[model_name]["name"],
    #         trust_remote_code=True,
    #         quantization_config=bnb_config,
    #         device_map='auto',
    #         token=HF_API_TOKEN
    #     )
    #     model.eval()

    #     tokenizer = AutoTokenizer.from_pretrained(list_of_models[model_name]["name"], token=HF_API_TOKEN)

    semantic_distance_matrix = 1 - cosine_similarity(embedding_array, embedding_array)
    # semantic_distance_matrix_norm = normalized_distance(semantic_distance_matrix)
    # no need to normalize becaus values are already in [0 - 1]
    semantic_distance_matrix_norm = semantic_distance_matrix
    distance_ratio = (1 + semantic_distance_matrix_norm) / (1 + geo_distance_matrix_norm)
    distance_ratio_norm = distance_ratio
    relative_diff = (semantic_distance_matrix_norm - geo_distance_matrix_norm) / (semantic_distance_matrix_norm + geo_distance_matrix_norm)
    mean_semantique = semantic_distance_matrix_norm
    harmonic_div = (semantic_distance_matrix_norm - geo_distance_matrix_norm) / geo_distance_matrix_norm
    df_results = pd.DataFrame(distance_ratio_norm, columns=df["city"].values, index=df["city"].values)
    df_results_semantic_distance_matrix = pd.DataFrame(semantic_distance_matrix, columns=df["city"].values, index=df["city"].values)
    df_results_relative = pd.DataFrame(relative_diff, columns=df["city"].values, index=df["city"].values)
    df_results_mean_semantique = pd.DataFrame(mean_semantique, columns=df["city"].values, index=df["city"].values)
    df_results_mean_semantique.to_csv(f"./output/aggregated_df_results_mean_semantique_{model_name.replace('/', '_')}.csv")
    df_results_harmonic_div = pd.DataFrame(harmonic_div, columns=df["city"].values, index=df["city"].values)

    boxplot_data_semantic_prox.append(df_results_semantic_distance_matrix.mean())
    boxplot_data_semantic_normalized_distance.append(df_results_mean_semantique.mean())

    world_init = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    # Compute for each continent
    for continent in continents_basemap:
        world = world_init
        # world = gpd.read_file("./data/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp")
        if continent != "World":
            world = world[world["continent"] == continent]

        df_aggregated_ratio_by_country_list = []
        for country in df['Country name EN'].unique():
            df_country = df_results.join(df[df["Country name EN"] == country].set_index("city"), how="inner")
            df_country_relative = df_results_relative.join(df[df["Country name EN"] == country].set_index("city"), how="inner")
            df_country_mean_semantique = df_results_mean_semantique.join(df[df["Country name EN"] == country].set_index("city"), how="inner")
            df_country_harmonic_div = df_results_harmonic_div.join(df[df["Country name EN"] == country].set_index("city"), how="inner")
            columns_to_drop = [item for item in df.columns if item != 'city']
            df_country = df_country.drop(columns=columns_to_drop)
            df_country_relative = df_country_relative.drop(columns=columns_to_drop)
            df_country_mean_semantique = df_country_mean_semantique.drop(columns=columns_to_drop)
            df_country_harmonic_div = df_country_harmonic_div.drop(columns=columns_to_drop)
            aggregated_country = {
                'country': country,
                'GDI': df_country.mean().mean(),
                'relative_diff': df_country_relative.mean().mean(),
                'mean_semantique': df_country_mean_semantique.mean().mean(),
                "harmonic_div": df_country_harmonic_div.mean().mean(),
            }
            df_aggregated_ratio_by_country_list.append(aggregated_country)
        df_aggregated_ratio_by_country = pd.DataFrame(df_aggregated_ratio_by_country_list)

        # compute relative difference between GDI and mean semantique
        diff = np.abs((df_aggregated_ratio_by_country['GDI'] - df_aggregated_ratio_by_country['mean_semantique']) / 
                      (df_aggregated_ratio_by_country['GDI'] + df_aggregated_ratio_by_country['mean_semantique']))
        df_aggregated_ratio_by_country["diff_relative_GDI_mean"] = diff

        # Define the mapping of country names
        country_mapping = {
            'Russian Federation': 'Russia',
            'United States': 'United States of America',
            'Libyan Arab Jamahiriya': 'Libya',
            'South Sudan, The Republic of': 'S. Sudan',
            'Sudan, The Republic of': 'Sudan',
            'Central African Republic': 'Central African Rep.',
            'Congo, Democratic Republic of the': 'Dem. Rep. Congo',
            'Tanzania, United Republic of': 'Tanzania',
            'Iran, Islamic Rep. of': 'Iran',
            'Czech Republic': 'Czechia',
            'Venezuela, Bolivarian Rep. of': 'Venezuela'
        }
        # Apply the mapping to the 'country' column in your DataFrame
        df_aggregated_ratio_by_country['country'] = df_aggregated_ratio_by_country['country'].replace(country_mapping)



        world = world.merge(df_aggregated_ratio_by_country, how='left', left_on='name', right_on='country')
        world.to_csv(f"./output/aggregated_{continent}_{model_name.replace('/', '_')}.csv")

        if continent == "World":
            df_aggregated_ratio_by_country_by_models.append(df_aggregated_ratio_by_country)

        # Plot the map
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        if continent == "Europe" or continent == "Oceania":
            m = Basemap(**continents_basemap[continent], resolution='c', ax=ax)
            # m.drawcoastlines()b.
            # m.drawcountries()
        # world.boundary.plot(ax=ax) # display boundaries or not
        world.plot(column='GDI', ax=ax, legend=True,
                legend_kwds={'label': "GDI",
                            'orientation': "horizontal"})
        ax.set_title(f"Mean GDI by Country for {model_name} model on {continent}", fontsize=20)
        # plt.show()
        plt.savefig(f"./figs/world_aggregated_GDI_{continent}_aggregated_{model_name.replace('/', '_')}.png", bbox_inches='tight')
        plt.close()


# box plot
df_boxplot = pd.DataFrame({'Bert': boxplot_data_semantic_prox[0].to_numpy(), 'Roberta': boxplot_data_semantic_prox[1].to_numpy(), 'openai/ada': boxplot_data_semantic_prox[2].to_numpy()})
fig, ax = plt.subplots()
sns.set(style="darkgrid")
sns.boxplot(data=df_boxplot, palette="pastel")
ax.set_xlabel('Models')
ax.set_ylabel('Mean semantic distance')
plt.savefig(f"./figs/boxplot_semantic_boxplot_data_semantic_prox.png")
plt.close()
print(f"\n\n The 20 most distance cities for BERT:\n {boxplot_data_semantic_prox[0].nlargest(n=20)} \n")
print(f"The 20 most distance cities for RoBERTa:\n {boxplot_data_semantic_prox[1].nlargest(n=20)} \n")
print(f"The 20 most distance cities for openAI/ada\n: {boxplot_data_semantic_prox[2].nlargest(n=20)} \n")

df_boxplot = pd.DataFrame({'Bert': boxplot_data_semantic_normalized_distance[0], 'Roberta': boxplot_data_semantic_normalized_distance[1], 'openai/ada': boxplot_data_semantic_normalized_distance[2]})
fig, ax = plt.subplots()
sns.set(style="darkgrid")
sns.boxplot(data=df_boxplot, palette="pastel")
ax.set_xlabel('Models')
ax.set_ylabel('Mean normalized semantic distance')
plt.savefig(f"./figs/boxplot_semantic_boxplot_data_semantic_normalized_distance.png")
plt.close()
