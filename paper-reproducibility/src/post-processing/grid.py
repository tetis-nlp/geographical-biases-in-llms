import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon
import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx

# Color bar choice
cmap = "RdYlGn"

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
model_order = list(list_of_models.keys())

list_of_expe =["Capital", "1M_inhabitants", "city_in_vocab", "closest_embedding_country", "closest_embedding_continent"]

for expe in list_of_expe:
    for model_name in model_order:
        # Load the data
        # model_name = "bert"
        if expe == "Capital":
            df = pd.read_csv(f"./output/self_masking_{model_name}")
        elif expe == "city_in_vocab":
            if model_name == "chatgpt":
                break
            else:
                df = pd.read_csv(f"./output/vocab_{model_name}.csv")
                df[f'correct_{model_name}'] = df["subtokens"]
        elif expe == "closest_embedding_country":
            if model_name == "chatgpt":
                break
            else:
                df = pd.read_csv(f"./output/closest_cities_{model_name.split('/')[0]}_5")
                df[f'correct_{model_name}'] = df["same_country"]
        elif expe == "closest_embedding_continent":
            if model_name == "chatgpt":
                break
            else:
                df = pd.read_csv(f"./output/closest_cities_{model_name.split('/')[0]}_5")
                df[f'correct_{model_name}'] = df["same_continent"]
        else:
            df = pd.read_csv(f"./output/self_masking_city_which_country{model_name}")
        try:
            df['longitude'] = df['CapitalLongitude']
            df['latitude'] = df["CapitalLatitude"]
        except:
            df[['latitude', 'longitude']] = df['Coordinates'].str.split(', ', expand=True)
            df['latitude'] = df['latitude'].astype(float)
            df['longitude'] = df['longitude'].astype(float)

        # Convert the DataFrame to a GeoDataFrame
        geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

        # Define the bounding box for the grid
        minx, miny, maxx, maxy = gdf.total_bounds

        # Define the grid size (adjust as necessary)
        grid_size = 5.0  # Adjust the size as needed

        # Create the grid cells
        x_coords = np.arange(minx, maxx + grid_size, grid_size)
        y_coords = np.arange(miny, maxy + grid_size, grid_size)
        polygons = [Polygon([(x, y), (x + grid_size, y), (x + grid_size, y + grid_size), (x, y + grid_size)]) 
                    for x in x_coords for y in y_coords]

        grid = gpd.GeoDataFrame({'geometry': polygons}, crs="EPSG:4326")

        # Assign points to grid cells
        gdf = gdf.sjoin(grid, how='left', predicate='within')

        # Filter out rows with NaN values in 'index_right'
        gdf = gdf.dropna(subset=['index_right'])

        # Convert 'index_right' to integer for indexing
        gdf['index_right'] = gdf['index_right'].astype(int)

        # Initialize the grid statistics
        grid['correct'] = 0
        grid['total'] = 0

        # Calculate the percentage of correct answers in each grid cell
        for idx, row in gdf.iterrows():
            grid_idx = row['index_right']
            grid.at[grid_idx, 'total'] += 1
            if row[f'correct_{model_name}']:
                grid.at[grid_idx, 'correct'] += 1

        # Handle division by zero
        grid['percent_correct'] = np.where(grid['total'] > 0, 100 * grid['correct'] / grid['total'], np.nan)

        # Convert to Web Mercator for contextily
        grid = grid.to_crs(epsg=3857)
        gdf = gdf.to_crs(epsg=3857)

        # Plot the results
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        # grid.boundary.plot(ax=ax, linewidth=1, edgecolor='black')
        grid_plot = grid.plot(column='percent_correct', ax=ax, legend=False, cmap=cmap, edgecolor='black', alpha=0.5)

        # Add basemap
        ctx.add_basemap(ax, crs=grid.crs, source=ctx.providers.CartoDB.Positron)

        # Plot the points on top
        # gdf.plot(ax=ax, color='red', markersize=5, alpha=0.5)

        fig.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust the bottom margin

        # Add horizontal colorbar
        # cax = fig.add_axes([0.2, 0.1, 0.6, 0.02])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=grid['percent_correct'].min(), vmax=grid['percent_correct'].max()))
        # sm._A = []
        # cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
        cbar = fig.colorbar(sm, orientation='horizontal')
        # cbar.set_label('Percentage of Correct Answers')

        # plt.title('Percentage of Correct Answers by Grid Cell')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.savefig(f"./figs/grid_{expe}_{model_name}.png")