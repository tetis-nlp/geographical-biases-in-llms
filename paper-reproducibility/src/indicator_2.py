from transformers import BertModel, BertTokenizer
from transformers import RobertaTokenizer, RobertaModel
from transformers import AutoTokenizer
from transformers import CamembertModel, CamembertTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from apikey import HF_API_TOKEN



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
list_of_models = {
    'bert': {
        'name': 'bert-base-uncased',
        'type': "SLM",
        'tokenizer': BertTokenizer.from_pretrained('bert-base-uncased'),
        # 'model': BertModel.from_pretrained('bert-base-uncased')
    },
    'bert-base-multilingual-uncased':{
        'name': 'bert-base-multilingual-uncased',
        'type': "SLM",
        'tokenizer': AutoTokenizer.from_pretrained('bert-base-multilingual-uncased'),
        # 'model': BertModel.from_pretrained('bert-base-multilingual-uncased')
    },
    'roberta': {
        'name': 'roberta-base',
        'type': "SLM",
        'tokenizer': AutoTokenizer.from_pretrained('roberta-base'),
        # 'model': RobertaModel.from_pretrained('roberta-base')
    },
    'geolm': {# no correct prediction
        'name': 'zekun-li/geolm-base-cased',
        'type': "SLM",
        'tokenizer': AutoTokenizer.from_pretrained('zekun-li/geolm-base-cased'),
        # 'model': RobertaModel.from_pretrained('zekun-li/geolm-base-cased'),
        'mask': "<mask>"
    },
    'xlm-roberta-base': {
        'name': 'xlm-roberta-base',
        'type': "SLM",
        'tokenizer': AutoTokenizer.from_pretrained('xlm-roberta-base'),
        # 'model': RobertaModel.from_pretrained('xlm-roberta-base'),
        'mask': "<mask>"
    },
    'llama2': {
        'name': 'meta-llama/Llama-2-7b-hf',
        'type': 'llm',
        'tokenizer': AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', use_auth_token=HF_API_TOKEN),
        # 'model': RobertaModel.from_pretrained('roberta-base')
    },
    'llama3': {
        'name': 'meta-llama/Meta-Llama-3-8B-Instruct',
        'type': 'llm',
        'local': './models/llama3/Meta-Llama-3-8B-Instruct',
        'tokenizer': AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct', use_auth_token=HF_API_TOKEN),
        # 'tokenizer': AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', use_auth_token=HF_API_TOKEN),
        # 'model': LlamaModel(LlamaConfig())
     },
    'mistral': {
        'name': 'mistralai/Mistral-7B-v0.1',
        'type': 'llm',
        'tokenizer': AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1'),
        # 'model': RobertaModel.from_pretrained('roberta-base')
    },
    'mistral-v03': {
        'name': 'mistralai/Mistral-7B-Instruct-v0.3',
        'type': 'llm',
        'local': './models/mistralai/Mistral-7B-Instruct-v0.3',
        'tokenizer': AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3'),
    #     # 'tokenizer': AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1'),
    #     # 'model': AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
    },
    'chatgpt':{
        'name': 'gpt-3.5-turbo-0301',
        'type': "LLM_remote_api"
    },
}

def decompose_to_subtokens(word, tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')):
    # Tokenize the word
    sub_tokens = tokenizer.tokenize(word)

    return sub_tokens

def check_token_in_vocab(token_to_check, tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')):
    # Check if the token is in the vocabulary
    # "Ġ": is the special character for Roberta Tokenizer saying it's the start of a word
    if isinstance(token_to_check, list): # for column with alternate names
        # work only on BERT and RoBERTa
        return any(str.lower(alternate_names) in tokenizer.get_vocab() for alternate_names in token_to_check) or any(str.lower("Ġ" + alternate_names) in tokenizer.get_vocab() for alternate_names in token_to_check)
    if str(type(tokenizer)) == "<class 'transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast'>":
        # for llama2 and mistral, the tokenizer do take into account the Maj
        return token_to_check in tokenizer.vocab
    # for Roberta, because Roberta is case sensistive
    elif str(type(tokenizer)) ==  "<class 'transformers.models.roberta.tokenization_roberta_fast.RobertaTokenizerFast'>" or str(type(tokenizer)) ==  "<class 'transformers.models.xlm_roberta.tokenization_xlm_roberta_fast.XLMRobertaTokenizerFast'>": 
        return token_to_check in tokenizer.get_vocab() or str('Ġ' + token_to_check) in tokenizer.get_vocab()
    else:
        return str.lower(token_to_check) in tokenizer.vocab or str.lower("Ġ" + token_to_check) in tokenizer.vocab

def retrieve_closest_tokens(target_word, top_k=30, tokenizer=BertTokenizer.from_pretrained('bert-base-uncased'), model=BertModel.from_pretrained('bert-base-uncased')):
    # Encode the target word
    input_ids = tokenizer.encode(target_word, return_tensors="pt")

    # Get the embeddings for the target word
    with torch.no_grad():
        outputs = model(input_ids)
        word_embedding = outputs.last_hidden_state.mean(dim=1) # we mean for word chunked into subtoken (out of model vocabulary) and [CLS] & [SEP]

    # Load embeddings for all tokens in the vocabulary
    all_embeddings = model.embeddings.word_embeddings.weight.data

    # Calculate cosine similarities
    similarities = cosine_similarity(word_embedding, all_embeddings)

    # Get top-10 closest tokens
    top_indices = similarities[0].argsort()[-top_k:]
    closest_tokens = [tokenizer.convert_ids_to_tokens([idx])[0] for idx in top_indices][::-1]

    # print
    print(f"Info on target word: {target_word}")
    # Check if the token is in the vocabulary
    token_in_vocab = check_token_in_vocab(target_word)
    print(f"\t Is it in model vocabulary?: {token_in_vocab}")
    if token_in_vocab is False:
        print(f"\t Subtokens: {decompose_to_subtokens(target_word)}")
    # Print the top-10 closest tokens
    print(f"\t The {top_k} closest word: {closest_tokens}")
    print("\n")

def plot_city_percentage_and_count(df, world_global, model_name, alternative_names=False):
    # Plot percentage of city > 100k in vocabulary
    country_city_counts = df.groupby('Country name EN')['city'].count()
    country_in_vocabulary_counts = df[df['subtokens']].groupby('Country name EN')['city'].count()
    country_ratios = country_in_vocabulary_counts / country_city_counts * 100
    # Using alternative_name ?
    if alternative_names:
        alternate = "_with_alter_names"
    else:
        alternate = ""

    world = world_global.copy()
    world = world.merge(country_ratios, how='left', left_on='name', right_index=True)
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    # ax.set_title(f"The percentage of cities with a population of over 100,000 that are in {model_name}'s vocabulary", fontsize=20)
    world.boundary.plot(ax=ax) # display boundaries or not
    world.plot(column='city', ax=ax, cmap='RdYlGn', legend=True, legend_kwds={'label': "Percentage", 'orientation': "horizontal"})
    plt.savefig(f"./figs/in_model_vocabulary_percent_in_vocab_{model_name}{alternate}_rdYlGn.png", bbox_inches='tight')
    plt.close()
    
    # Nb of city >100k
    world = world_global.copy()
    world = world.merge(country_city_counts, how='left', left_on='name', right_index=True)
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    world.plot(column='city', cmap='coolwarm', ax=ax, legend=True, legend_kwds={'label': "Number"})
    ax.set_title(f"Number of cities with over 100,000 inhabitants per country", fontsize=20)
    plt.savefig(f"./figs/in_model_vocabulary_nb_city_100k.png", bbox_inches='tight')
    plt.close()

def plot_cities_with_borders(df, world, model_name):
    # Extract Latitude and Longitude from Coordinates
    df[['Latitude', 'Longitude']] = df['Coordinates'].str.extract(r'([-]?\d+\.\d+),\s*([-]?\d+\.\d+)').astype(float)

    # Scatter plot of cities
    plt.figure(figsize=(15, 10))

    # Scatter plot for True values (blue)
    plt.scatter(df[df['subtokens']]['Longitude'], df[df['subtokens']]['Latitude'], s=df[df['subtokens']]['Population']/100000, alpha=0.5, color='blue', label='True')

    # Scatter plot for False values (red)
    plt.scatter(df[~df['subtokens']]['Longitude'], df[~df['subtokens']]['Latitude'], s=df[~df['subtokens']]['Population']/100000, alpha=0.5, color='red', label='False')

    # Plot country borders
    world.boundary.plot(ax=plt.gca(), color='black')
    # Add legend
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label=f'Cities in {model_name} vocabulary: no'),
                        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label=f'Cities in {model_name} vocabulary: yes')]
    plt.legend(handles=legend_elements)
    # Set labels
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(f"./figs/in_model_vocabulary_scatter_plot_{model_name}.png", bbox_inches='tight')
    plt.close()
    #plt.show()

import pycountry_convert as pc
def country_to_continent(country_alpha2):
    #country_alpha2 = pc.country_name_to_country_alpha2(country_name)
    try:
        country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
        country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
    except:
        country_continent_name = "unknown"
    return country_continent_name



if __name__ == "__main__":
    world_global = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    df = pd.read_csv("data/geonames-all-cities-with-a-population-1000.csv", sep=";")
    df["city"] = df["ASCII Name"]
    df["city"] = df["city"].astype(str)
    df['Country name EN'] = df['Country name EN'].replace(country_mapping)
    df['Alternate Names'] = df['Alternate Names'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
    # Work only on city with Population is 100k of inhabitants
    df = df[df['Population'] > 100000].reset_index(drop=True)
    df_global = df.copy()
    
    vocab_size = []
    percentage_by_continent = []
    for model_name in list_of_models:
        df = df_global.copy()
        print(f"Dealing with {model_name}")
        if list_of_models[model_name]["type"] != "LLM_remote_api":
            tokenizer = list_of_models[model_name]["tokenizer"]
            # model = list_of_models[model_name]["model"]
            print(f"\t Size of the vocab: {len(tokenizer.vocab)}")
            vocab_size.append({model_name: len(tokenizer.vocab)})
            df["subtokens"] = df["city"].apply(check_token_in_vocab, tokenizer=tokenizer)
        else:
            import tiktoken
            tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
            def in_vocab(city):
                result = False
                try:
                    if(len(tokenizer.encode(city)) == 1): # no subtokens
                        result = True
                except:
                    pass
                return result
            df["subtokens"] = df["city"].apply(in_vocab)

        plot_cities_with_borders(df, world_global, model_name)
        plot_city_percentage_and_count(df, world_global, model_name)
        # df["subtokens"] = df.apply(lambda row: check_token_in_vocab(row["city"], tokenizer) or check_token_in_vocab(row["Alternate Names"], tokenizer) if isinstance(row["Alternate Names"], list) else check_token_in_vocab(row["city"], tokenizer), axis=1)
        # plot_city_percentage_and_count(df, world_global, model_name, alternative_names=True)

        # Generate a table
        df["continent"] = df["Country Code"].apply(country_to_continent)
        df.to_csv(f"./output/vocab_{model_name}.csv")
        country_city_counts = df.groupby('continent')['city'].count()
        country_in_vocabulary_counts = df[df['subtokens']].groupby('continent')['city'].count()
        country_ratios = country_in_vocabulary_counts / country_city_counts * 100
        model_stat ={
            model_name: country_ratios.to_dict(),
        }
        percentage_by_continent.append(model_stat)
        # print(f"closest token from Ouagadoug: {retrieve_closest_tokens('ouagadougou', 10, tokenizer, model)}")
    
    percentage_by_continent.append({"Nb of cities > 100k": country_city_counts.to_dict()})
    print(percentage_by_continent)
    df_results = pd.DataFrame()
    for entry in percentage_by_continent:
        model_name, model_data = list(entry.items())[0]
        df_model = pd.DataFrame([model_data])
        df_model.index = [model_name]
        df_results = pd.concat([df_results, df_model])
    df_results = df_results.astype(float)
    df_results = df_results.fillna(0)
    df_results.iloc[-1] = df_results.iloc[-1].astype(int)
    df_results = df_results.round(2)
    df_results.to_csv(f"./output/percentage_in_vocab.csv")
    print(df_results)

    dff = pd.read_csv(f"./output/percentage_in_vocab.csv")
    dff['model'] = dff["Unnamed: 0"]
    dff = dff[['model', 'North America', 'South America', 'Europe', 'Africa', 'Asia', 'Oceania']]
    model_order = list(list_of_models.keys())
    dff.set_index("model", inplace=True)
    dff = dff.loc[model_order]

    df_vocab_size = pd.DataFrame(vocab_size)
    df_vocab_size = df_vocab_size.stack()
    print(df_vocab_size.T)

    # retrieve_closest_tokens("paris", tokenizer, model)
    # retrieve_closest_tokens("london", tokenizer, model)
    # retrieve_closest_tokens("ouagadougou", tokenizer, model)
    # retrieve_closest_tokens("berlin", tokenizer, model)
