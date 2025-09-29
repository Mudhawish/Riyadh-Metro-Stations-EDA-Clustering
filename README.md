# 🚇 Riyadh Metro Stations — EDA & Clustering

An end-to-end data science project using **Riyadh Metro station data** (~88 stations).  
This repository includes **data cleaning, Q&A-style exploratory analysis, clustering with K-Means, model evaluation, and an interactive Streamlit dashboard with Plotly visualizations** to better understand metro coverage, station groupings, and user ratings.

---
## 🌟 Highlights

- **End-to-end Workflow**: Combines raw data acquisition, exploratory analysis, machine learning clustering, and interactive dashboard development all in one repository.
- **Practical Clustering on Real Geographic Data**: Demonstrates practical and modern clustering methods applied to real-world geographic data, offering actionable insights.
- **Interactive Visualization**: Features an interactive dashboard for deep exploration of station-level metrics, including density maps, cluster assignments, and station ratings.
- **Clear & Reproducible Structure**: Organized, transparent, and reproducible project structure to facilitate understanding, collaboration, and extension by others.

---
## 📁 Project Structure

```text
├── data/
│   ├── riyadh_metro_raw_1.csv          # Raw dataset (before cleaning)
│   ├── riyadh_metro_raw_2.csv
│   └── riyadh_metro_clean.csv          # Cleaned dataset (after preprocessing)
│
├── notebooks/
│   ├── 01_clean_merge_addcols.ipynb    # Data cleaning, merging & feature engineering
│   └── 02_eda_and_model.ipynb          # EDA (Q&A) & clustering models
│
├── riyadh_metro_app.py                 # Streamlit dashboard
│
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Ignore unnecessary files
└── README.md                           # Project documentation
```

---

## 🎯 Objectives

- **Data Preparation**: Clean and merge raw datasets into a consistent, analysis-ready format.
- **Exploratory Data Analysis (EDA)**: Investigate metro stations using a Q&A-style approach (questions → code → insights).
- **Ratings & Coverage Analysis**: Examine ratings distribution, identify top and bottom-rated stations, and uncover metro coverage patterns.
- **Geographic Clustering**: Apply K-Means clustering to group stations geographically, with evaluation using silhouette scores.
- **Interactive Visualization**: Visualize insights with interactive Plotly heatmaps, scatter maps, and cluster plots.
- **Dashboard Deployment**: Share findings through a Streamlit dashboard for live, user-friendly exploration of data and clustering results.

---

## 📊 Dataset

- **Source**: Kaggle and opendatasoft
- **Stations**: ~88
- **Type**: Structural (not time-series)

| Column              | Description                            |
| ------------------- | -------------------------------------- |
| `Station_Name`      | Name of the station                    |
| `Latitude`          | Latitude coordinate                    |
| `Longitude`         | Longitude coordinate                   |
| `Rating`            | User rating (score)                    |
| `Number_of_Ratings` | Number of reviews per station          |
| `Metro line name`   | Line identifier (name)                 |
| `Metro line number` | Line number                            |
| `Station type`      | Elevated / Underground                 |
| `Type_of_Utility`   | Station classification                 |
| `Match_Method`      | Notes on data source matching          |

---

## 🛠️ Installation

Clone the repository:
```sh
git clone https://github.com/<YOUR-USERNAME>/riyadh-metro-stations-eda-clustering.git
cd riyadh-metro-stations-eda-clustering
```

Create a virtual environment and install dependencies:
```sh
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Launch Jupyter Notebook:
```sh
jupyter notebook
```
---

## 🚀 Run the Dashboard

The interactive dashboard is built using **Streamlit**.  
It allows you to explore station data, visualize density maps, and view clustering results interactively.

To launch the dashboard locally:

```sh
pip install -r requirements.txt
streamlit run riyadh_metro_app.py
```

---

## 📈 Key Insights

- **High Station Density**: Central and northern Riyadh have the highest concentration of metro stations.
- **Rating Variation**: Station ratings differ significantly across metro lines and geographic areas.
- **Top-rated Station Clusters**: The highest-rated stations are concentrated around business districts and city hubs.
- **Clustering Results**: K-Means clustering revealed natural geographic groupings in metro coverage, validated by silhouette scores.
- **Visualization Insights**: Interactive Plotly maps (scatter and density heatmaps) uncovered both strong coverage areas and potential service gaps.
- **User-friendly Exploration**: The Streamlit dashboard enables dynamic, accessible exploration of stations, ratings, and cluster patterns.

---

## ⚙️ Requirements

To run and reproduce this project, make sure you have the following Python libraries installed:

- **pandas** – for data manipulation and cleaning
- **numpy** – for numerical operations
- **scikit-learn** – for clustering (K-Means, silhouette evaluation)
- **plotly** – for interactive maps and visualizations
- **matplotlib** – for static plots supporting EDA
- **streamlit** – for building the interactive dashboard framework

---

## 👥 Contributors
- Mudhawi Alshiha (@Mudhawish)  
- Abdulelah Alowaid (@AbAlowaid)  
- ⁠Fatima Alsubaie (@fatima_turki) 
---

## ✨ Acknowledgments

Developed as part of a data science learning journey, focusing on practical EDA and clustering techniques with real-world Riyadh Metro data.
