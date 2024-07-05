# Process to reproduce the paper

To reproduce the figures and tables from the manuscript, please follow this procedure.

All these experiments have been run on an Ubuntu server with NVIDIA V100 (32GB RAM), but they could be run on a GPU with at least 24GB. If you don't have a GPU, you may want to use Google Colab as provided by the landing page of this GitHub repository.

Please note that the total estimated execution time, if the indicators are run sequentially, is approximately 3 to 4 days.

## 1. Get the data
To run all the scripts, it is necessary to download two geodatasets and place them into the `data/` directory:

1. data/kaggle_world_captials_gps.csv: https://www.kaggle.com/datasets/nikitagrec/world-capitals-gps
2. data/geonames-all-cities-with-a-population-1000.csv: https://public.opendatasoft.com/explore/dataset/geonames-all-cities-with-a-population-1000/table/?flg=fr-fr&disjunctive.cou_name_en&sort=name

## 2. Configure our runtime environment
### Python environment
```{bash}
conda env create -f environment.yml
```

### API keys
The scripts need 2 API keys from OpenAI and from HuggingFace.
Please generate them and copy-paste them into the file `apikey.py`. A template is provided below:

```{python}
OPENAI_API_KEY = "...secret..."
HF_API_TOKEN = "...secret..."
```

## 3. Run the codes
### 1. Activate the conda envrionment

```{bash}
conda activate geographical-biases-in-llms 
```

### 2. Run the scripts corresponding to the 4 indicators
The indicators can be run in parallel or in any order. However, the post-processing scripts have to be run after the 4 indicators have finished.

```{bash}
# 1rst indicator
python paper-reproducibility/src/indicator_1_a.py
python paper-reproducibility/src/indicator_1_b.py

# 2nd indicator
python paper-reproducibility/src/indicator_2.py

# 3rd indicator
python paper-reproducibility/src/indicator_3.py

# 4th indicator
python paper-reproducibility/src/indicator_4_a.py
python paper-reproducibility/src/indicator_4_b.py
```

Then run the post-processing scripts:

```{bash}
python paper-reproducibility/src/post-processing/grid.py
python paper-reproducibility/src/post-processing/litterature_reviews.py
```

### 3. Corresponding scripts to reproduce tables and figures from the paper 

#### 3.1 Figures to reproduce

| Figure   | Script                                 |
|--------------------|----------------------------------------|
| Figure 1 (a) and (b) | Indicator_1_a.py                      |
| Figure 1 (c) and (d) | Indicator_1_a.py post_processing/grid.py |
| Figure 2           | Indicator_1_b.py post_processing/grid.py |
| Figure 3 (a) and (b) | Indicator_2.py                        |
| Figure 3 (c) and (d) | Indicator_2.py post_processing/grid.py |
| Figure 4           | Indicator_3.py                         |
| Figure 5           | Indicator_4.py                         |
| Figure 6 (all)     | Indicator_4.py post_processing/grid.py  |

#### 3.2 Tables to reproduce

| Table              | Script                                 |
|--------------------|----------------------------------------|
| Table 1            | Indicator_2.py                         |
| Table 2            | Indicator_1_a.py                       |
| Table 3            | Indicator_1_b.py                       |
| Table 4            | Indicator_2.py                         |
| Table 5            | Indicator_3.py                         |
| Table 6            | Indicator_4.py                         |
| Table 7            | post_processing/litterature_reviews.py |
