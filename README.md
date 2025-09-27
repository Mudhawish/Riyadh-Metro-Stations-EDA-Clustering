{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\froman\fcharset0 Times-Roman;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c0;}
\margl1440\margr1440\vieww10700\viewh15080\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs24 \cf0 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 # \uc0\u55357 \u56967  Riyadh Metro Stations \'97 EDA & Clustering\
\
An exploratory data analysis and clustering project using **Riyadh Metro station data** (~88 stations).  \
The project covers **data cleaning, Q&A-style EDA, clustering with K-Means, model evaluation, and interactive Plotly visualizations** to better understand metro coverage and station groupings.\
\
---\
\
## \uc0\u55357 \u56513  Project Structure\
\
```text\
\uc0\u9500 \u9472 \u9472  data/\
\uc0\u9474    \u9500 \u9472 \u9472  riyadh_metro_raw.csv            # Raw dataset (before cleaning)\
\uc0\u9474    \u9492 \u9472 \u9472  riyadh_metro_clean.csv          # Cleaned dataset (after preprocessing)\
\uc0\u9500 \u9472 \u9472  notebooks/\
\uc0\u9474    \u9500 \u9472 \u9472  01_clean_merge_addcols.ipynb    # Data cleaning, merging & feature engineering\
\uc0\u9474    \u9492 \u9472 \u9472  02_eda_and_model.ipynb          # EDA (Q&A) & clustering models\
\uc0\u9500 \u9472 \u9472  outputs/\
\uc0\u9474    \u9492 \u9472 \u9472  riyadh_metro_with_clusters.csv  # (Optional) Data with cluster assignments\
\uc0\u9500 \u9472 \u9472  requirements.txt                    # Python dependencies\
\uc0\u9492 \u9472 \u9472  README.md                           # Project documentation\
```\
\
---\
\
## \uc0\u55356 \u57263  Objectives\
\
- **Clean & merge** raw datasets into a consistent, usable format\
- **Explore** metro stations with a Q&A-style EDA (questions \uc0\u8594  code \u8594  insights)\
- Analyze **ratings distribution**, top/bottom-rated stations, and overall coverage\
- **Cluster** stations geographically using K-Means\
- **Visualize** results with Plotly density heatmaps and interactive cluster maps\
\
---\
\
## \uc0\u55357 \u56522  Dataset\
\
- **Source**: Manually cleaned & merged Riyadh Metro Stations Dataset\
- **Stations**: ~88\
- **Type**: Structural (not time-series)\
\
| Column              | Description                            |\
| ------------------- | -------------------------------------- |\
| `Station_Name`      | Name of the station                    |\
| `Latitude`          | Latitude coordinate                    |\
| `Longitude`         | Longitude coordinate                   |\
| `Rating`            | User rating (score)                    |\
| `Number_of_Ratings` | Number of reviews per station          |\
| `Metro line name`   | Line identifier (name)                 |\
| `Metro line number` | Line number                            |\
| `Station type`      | Elevated / Underground                 |\
| `Type_of_Utility`   | Station classification                 |\
| `Match_Method`      | Notes on data source matching          |\
\
---\
\
## \uc0\u55357 \u57056 \u65039  Installation\
\
Clone the repository:\
```sh\
git clone https://github.com/<YOUR-USERNAME>/riyadh-metro-stations-eda-clustering.git\
cd riyadh-metro-stations-eda-clustering\
```\
\
Create a virtual environment and install dependencies:\
```sh\
python -m venv .venv\
source .venv/bin/activate   # On Windows: .venv\\Scripts\\activate\
pip install -r requirements.txt\
```\
\
Launch Jupyter Notebook:\
```sh\
jupyter notebook\
```\
\
---\
\
## \uc0\u55357 \u56520  Key Insights\
\
- Central and northern Riyadh show the **highest station density**\
- **Station ratings** vary significantly across metro lines and areas\
- **Top-rated stations** are concentrated in key hubs and business districts\
- **K-Means clustering** identified natural geographic zones of station coverage\
- **Plotly heatmaps** highlight city-wide coverage patterns and gaps\
\
---\
\
## \uc0\u9881 \u65039  Requirements\
\
- pandas  \
- numpy  \
- scikit-learn  \
- plotly  \
- matplotlib  \
\
---\
\
## \uc0\u55357 \u56516  License\
\
This project is licensed under the **MIT License**. You are free to use, modify, and distribute it with attribution.\
\
---\
\
## \uc0\u10024  Acknowledgments\
\
Developed as part of a data science learning journey, focusing on practical EDA and clustering techniques with real-world Riyadh Metro data.\
}