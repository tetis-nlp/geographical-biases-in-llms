# Geographical Biases in Large Language Models (LLMs)

This tutorial aims to identify geographical biases propagated by LLMs. For this purpose, 4 indicators are proposed.

1. Spatial disparities in geographical knowledge. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tetis-nlp/geographical-biases-in-llms/blob/master/.src/0000.ipynb)
2. Spatial information coverage in training datasets. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tetis-nlp/geographical-biases-in-llms/blob/master/.src/0001.ipynb)
3. Correlation between geographic distance and semantic distance. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tetis-nlp/geographical-biases-in-llms/blob/master/.src/0002.ipynb)
4. Anomaly between geographical distance and semantic distance. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tetis-nlp/geographical-biases-in-llms/blob/master/.src/0003.ipynb)

<p align="center">
  <img src="readme.ressources/World_aggregated_bert.png" alt="Semantic Distances" width="400"/><br/>
  <strong>Fig. 1:</strong> Average semantic distances (using BERT) between the three most populous cities in a country compared to other cities worldwide.
</p>

<p align="center">
  <img src="readme.ressources/percentage_correct_country_predictions_for_100kinhab_cities.png" alt="Semantic Distances" width="400"/><br/>
  <strong>Fig. 2:</strong> Percentage of correct country predictions given cities name with more than 100K inhabitants by spatial aggregation in 5° by 5° pixels?
</p>


-----
## Authors

<img align="left" src="https://www.umr-tetis.fr/images/logo-header-tetis.png">


|           |
|----------------------|
| Rémy Decoupes        |
| Maguelonne Teisseire |
| Mathieu Roche        |

**Acknowledgement**:

This study was partially funded by EU grant 874850 MOOD and is catalogued as MOOD099. The contents of
this publication are the sole responsibility of the authors and do not necessarily reflect the views of the European
Commission

<a href="https://mood-h2020.eu/"><img src="https://mood-h2020.eu/wp-content/uploads/2020/10/logo_Mood_texte-dessous_CMJN_vecto-300x136.jpg" alt="mood"/></a> 


---
## Citing this work

If you find this work helpful or refer to it in your research, please consider citing:

+ *Evaluation of Geographical Distortions in Language Models: A Crucial Step Towards Equitable Representations, Rémy Decoupes, Roberto Interdonato, Mathieu Roche, Maguelonne Teisseire, Sarah Valentin*. 
[![arXiv](https://img.shields.io/badge/arXiv-2404.17401-b31b1b.svg)](https://arxiv.org/abs/2404.17401)

## This tutorial has been presented in

- **PAKDD'24**: See [slides](slides/PAKDD2024_Tutorial_LM_Spatial.pdf). [![SWH](https://archive.softwareheritage.org/badge/swh:1:rev:7a5b8799435c1bc858fde76347347a7ef44f0053/)](https://archive.softwareheritage.org/swh:1:rev:7a5b8799435c1bc858fde76347347a7ef44f0053;origin=https://github.com/tetis-nlp/geographical-biases-in-llms;visit=swh:1:snp:6dad1d279a108a591059f4a006d379285bb2a575)


|   |   |   |   |
|---|---|---|---|
| <a href="https://www.agroparistech.fr/"><img src="https://ecampus.paris-saclay.fr/pluginfile.php/422294/coursecat/description/logo_sansbaseline.png" alt="AgroParisTech"/></a> | <a href="https://www.cirad.fr/"><img src="https://en.agreenium.fr/sites/default/files/styles/large/public/CIRAD.jpg" alt="CIRAD" /></a> | <a href="https://www.cnrs.fr"><img src="https://upload.wikimedia.org/wikipedia/fr/thumb/7/72/Logo_Centre_national_de_la_recherche_scientifique_%282023-%29.svg/langfr-130px-Logo_Centre_national_de_la_recherche_scientifique_%282023-%29.svg.png" alt="CNRS"/></a> | <a href="https://www.inrae.fr"><img src="https://www.inrae.fr/themes/custom/inrae_socle/logo.svg" alt="INRAE" /></a> |
