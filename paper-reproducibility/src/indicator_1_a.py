from transformers import BertModel, BertTokenizer
from transformers import RobertaTokenizer, RobertaModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import CamembertModel, CamembertTokenizer
from transformers import pipeline, BitsAndBytesConfig
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from geopy.distance import geodesic
import numpy as np
import pycountry_convert as pc
from scipy.stats import linregress
from transformers import LlamaModel, LlamaConfig
from apikey import HF_API_TOKEN, OPENAI_API_KEY
from torch import bfloat16
import openai
import pycountry_convert as pc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# /!\ contourning huggingface accessibility from BigVM
# don't check certificate
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Create a custom requests session that disables SSL verification
class CustomSession(requests.Session):
    def __init__(self):
        super().__init__()
        self.verify = False


list_of_models = {
    'bert': {
        'name': 'bert-base-uncased',
        # 'tokenizer': BertTokenizer.from_pretrained('bert-base-uncased'),
        # 'model': BertModel.from_pretrained('bert-base-uncased'),
        'mask': "[MASK]",
        'type': 'slm',
        'local': './models/bert/bert-base-uncased',
    },
    'bert-base-multilingual-uncased':{
        'name': 'bert-base-multilingual-uncased',
        # 'tokenizer': AutoTokenizer.from_pretrained('bert-base-multilingual-uncased'),
        # 'model': BertModel.from_pretrained('bert-base-multilingual-uncased'),
        'mask': "[MASK]",
        'type': 'slm',
        'local': './models/bert-base-multilingual-uncased/bert-base-multilingual-uncased',
    },
    'roberta': {
        'name': 'roberta-base',
        # 'tokenizer': AutoTokenizer.from_pretrained('roberta-base'),
        # 'model': RobertaModel.from_pretrained('roberta-base'),
        'mask': "<mask>",
        'type': 'slm',
        'local': './models/FacebookAI/roberta-base',
    },
    'xlm-roberta-base': {
        'name': 'xlm-roberta-base',
        # 'tokenizer': AutoTokenizer.from_pretrained('xlm-roberta-base'),
        # 'model': RobertaModel.from_pretrained('xlm-roberta-base'),
        'mask': "<mask>",
        'type': 'slm',
        'local': './models/FacebookAI/xlm-roberta-base',
    },
    # 'geolm': {
    #     'name': 'zekun-li/geolm-base-cased',
    #     # 'tokenizer': AutoTokenizer.from_pretrained('zekun-li/geolm-base-cased'),
    #     # 'model': RobertaModel.from_pretrained('zekun-li/geolm-base-cased'),
    #     'mask': "[MASK]",
    #     'type': 'slm',
    #     'local': './models/geolm/geolm-base-cased',
    # },
    'mistral': {
        'name': 'mistralai/Mistral-7B-Instruct-v0.1',
        'local': './models/mistral/Mistral-7B-Instruct-v0.1',
        'type': 'llm',
    #     # 'tokenizer': AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1'),
    #     # 'model': AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
    }, 
    'mistral-v03': {
        'name': 'mistralai/Mistral-7B-Instruct-v0.3',
        'type': 'llm',
        'local': './models/mistralai/Mistral-7B-Instruct-v0.3',
    #     # 'tokenizer': AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1'),
    #     # 'model': AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
    },
    'llama2': {
        'name': 'meta-llama/Llama-2-7b-chat-hf',
        'type': 'llm',
        'local': './models/llama2/Llama-2-7b-chat-hf',
        # 'tokenizer': AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', use_auth_token=HF_API_TOKEN),
        # 'model': LlamaModel(LlamaConfig())
     },
    'llama3': {
        'name': 'meta-llama/Meta-Llama-3-8B-Instruct',
        'type': 'llm',
        'local': './models/llama3/Meta-Llama-3-8B-Instruct',
        # 'tokenizer': AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', use_auth_token=HF_API_TOKEN),
        # 'model': LlamaModel(LlamaConfig())
     },
    # 'qwen': {
    #     'name': 'Qwen/Qwen1.5-7B-Chat',
    #     'type': 'llm',
    #     'local': './models/qwen/Qwen-7B',
    #     # 'tokenizer': AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', use_auth_token=HF_API_TOKEN),
    #     # 'model': LlamaModel(LlamaConfig())
    #  },
    'chatgpt':{
        'name': 'gpt-3.5-turbo-0301',
        'type': 'rllm',
    },
}

country_mapping = {
#    'united states': 'united states of america',
    'south sudan, the Republic of': 'south sudan',
    'sudan, the Republic of': 'sudan',
#    'central african republic': 'central african rep.',
    'congo, democratic republic of the': 'democratic republic of the congo',
#    'czech republic': 'czechia',
    'western sahara' : 'w. Sahara',
    'cote d\'ivoire': 'côte d\'ivoire',
#    'equatorial guinea': 'eq. guinea',
#    'south sudan': 's. sudan',
#    'democratic republic of the congo': 'dem. rep. congo',
    'republic of congo': 'congo',
}
def country_to_continent(country_alpha2):
    #country_alpha2 = pc.country_name_to_country_alpha2(country_name)
    try:
        country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
        country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
    except:
        country_continent_name = "unknown"
    return country_continent_name

if __name__ == "__main__":
    df = pd.read_csv("data/kaggle_world_captials_gps.csv")
    df['CountryName'] = df['CountryName'].str.lower()
    df["continent"] = df["CountryCode"].apply(country_to_continent) # as other the 2 first expe
    df = df[(df["continent"] != "unknown") & (df["continent"] != "antartica")]
    df_global = df.copy()
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world["name"] = world["name"].str.lower()
    world_global = world.copy()
    df_accuracy_model = pd.DataFrame()
    for model_name in list_of_models:
        print(f"Dealing with {model_name}")
        # if model_name != "llama2" and model_name != "mistral" and model_name != "chatgpt":
        if list_of_models[model_name]["type"] == "slm":
            fill_mask = pipeline(task="fill-mask", model=list_of_models[model_name]["name"])

            def self_masking(city):
                # Define the masked sentence
                if model_name != "camembert": # english
                    masked_sentence = f'{city} is capital of {list_of_models[model_name]["mask"]}.'
                else: # french
                    masked_sentence = f'{city} est la capitale de {list_of_models[model_name]["mask"]}.'
                # Use the pipeline to predict the masked token
                predictions = fill_mask(masked_sentence)
                # Get the predicted token
                predicted_token = predictions[0]['token_str']
                return predicted_token.lstrip()
        #elif model_name == "mistral":
        elif list_of_models[model_name]["type"] == "llm":
            #old fashion 
            # bnb_config = BitsAndBytesConfig(
            #     load_in_4bit=True,  # 4-bit quantization
            #     bnb_4bit_quant_type='nf4',  # Normalized float 4
            #     bnb_4bit_use_double_quant=True,  # Second quantization after the first
            #     bnb_4bit_compute_dtype=bfloat16  # Computation type
            # )
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)

            model = AutoModelForCausalLM.from_pretrained(
                # list_of_models[model_name]["local"],
                # local_files_only=True,
                list_of_models[model_name]["name"],
                trust_remote_code=True,
                # quantization_config=bnb_config,
                quantization_config=quantization_config,
                device_map='auto',
                # use_safetensors=False,
                token=HF_API_TOKEN
            )
            model.eval()

            tokenizer = AutoTokenizer.from_pretrained(list_of_models[model_name]["name"], token=HF_API_TOKEN, trust_remote_code=True)
            # tokenizer = AutoTokenizer.from_pretrained(list_of_models[model_name]["local"], local_files_only=True)
            def self_masking(city):
                messages = [
                    {"role": "user", "content": "Name the country corresponding to its capital: Paris. Only give the country."},
                    {"role": "assistant", "content": "France"},
                    {"role": "user", "content": f"Name the country corresponding to its capital: {city}. Only give the country."}]
                encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
                # prompt = f"Name the country corresponding to {city}. Only give the country."
                model_inputs = encodeds.to('cuda')

                generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
                decoded = tokenizer.batch_decode(generated_ids)
                if model_name == "qwen": #could not apply a standardized chat template
                    try:
                        country = decoded[0].split("<|im_start|>assistant\n")[2].split("吗")[0] # 吗: Chinese question tag
                    except:
                        try :
                            country = decoded[0].split("<|im_start|>assistant")[2].split("吗")[0].replace("\n", "")
                        except:
                            country = decoded[0]
                elif model_name == "llama3":
                    try:
                        country = decoded[0].split("<|end_header_id|>")[4].split("<|eot_id|>")[0].replace("\n", "")
                    except:
                        country = decoded[0]
                else:
                    try:
                        country = decoded[0].split("[/INST] ")[-1].replace("</s>", "").lstrip()
                    except:
                        try:
                            country = decoded[0].split("[/INST]")[-1].lstrip()
                        except:
                            country = decoded[0]
                return country
        else: #ChatGPT
            openai.api_key = OPENAI_API_KEY
            def self_masking(city):
                messages = [
                    {"role": "user", "content": "Name the country corresponding to its capital: Paris. Only give the country."},
                    {"role": "assistant", "content": "France"},
                    {"role": "user", "content": f"Name the country corresponding to its capital: {city}. Only give the country."}]
                try:
                    reponse = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo-0301",
                        messages=messages
                    )
                    return reponse['choices'][0]['message']['content'].replace(".","")
                except:
                    return "time_out"
        try:
            df[f"predicted_{model_name}"] = df["CapitalName"].apply(self_masking).str.lower()
            df[f"correct_{model_name}"] = df['CountryName'] == df[f"predicted_{model_name}"]
            df.to_csv(f"./output/self_masking_{model_name}")

            # Calculate the accuracy by continent
            # accuracy_by_continent = df.groupby('ContinentName')[f"correct_{model_name}"].mean() * 100
            accuracy_by_continent = df.groupby('continent')[f"correct_{model_name}"].mean() * 100

            # Create a DataFrame to display the results
            accuracy_df = pd.DataFrame({'ContinentName': accuracy_by_continent.index, 'Accuracy': accuracy_by_continent.values})
            accuracy_df.loc[len(accuracy_df)] = {'ContinentName': "World", 'Accuracy': df[f"correct_{model_name}"].mean() * 100}
            accuracy_df["model"] = model_name
            print(accuracy_df)
            df_accuracy_model = pd.concat([df_accuracy_model, accuracy_df], ignore_index=True)
            

            # Merge the world shapefile with your DataFrame on country names
            df['CountryName'] = df['CountryName'].replace(country_mapping)
            world = world_global.copy()
            world = world.merge(df, how='left', left_on='name', right_on='CountryName')

            # Plot the map
            fig, ax = plt.subplots(1, 1, figsize=(15, 10))
            world.boundary.plot(ax=ax, linewidth=0.5, color='black')
            # pastel
            colors = ['#FF9999', '#66B3FF']  # Light red and light blue
            cmap = plt.cm.get_cmap('Pastel1', 2) 
            world.plot(column=f"correct_{model_name}", cmap=cmap, ax=ax, legend=True)
            legend = ax.get_legend()
            for label in legend.get_texts():
                label.set_fontsize(25)
            # plt.show()
            # plt.title(f"Correct Predictions of Countries Given their Capitals with {model_name}")
            plt.savefig(f"./figs/self_masking_{model_name}.png", bbox_inches='tight')
            plt.close()
            df = df_global.copy()
        except:
            print(f"error with: {model_name}")


    # df_accuracy_model.set_index('model', inplace=True)
    # df_accuracy_model = df_accuracy_model.groupby('model').sum()
    df_accuracy_model = df_accuracy_model.round(2)
    df_accuracy_model.to_csv('./output/self_masking_model_without_pivot.csv')
    df_accuracy_model = df_accuracy_model.pivot(index='model', columns='ContinentName', values='Accuracy')
    df_accuracy_model.to_csv('./output/self_masking_model.csv')

    dff = pd.read_csv('./output/self_masking_model.csv')
    dff = dff[['model', 'North America', 'South America', 'Europe', 'Africa', 'Asia', 'Oceania', 'World']]
    model_order = list(list_of_models.keys())
    dff.set_index("model", inplace=True)
    dff = dff.loc[model_order]
    dff.to_csv('./output/self_masking_model_formated.csv')
    print(dff)
    df_nb_countries = pd.DataFrame(df_global.groupby('continent')["CountryName"].count()).transpose()
    df_nb_countries = df_nb_countries[['North America', 'South America', 'Europe', 'Africa', 'Asia', 'Oceania']]
    df_nb_countries["World"] = df_global["CountryName"].count()
    print(df_nb_countries)

    print("end")
            


        



