# 🚇 Riyadh Metro Stations — EDA & Clustering

An exploratory data analysis and clustering project using **Riyadh Metro station data** (~88 stations).  
The project covers **data cleaning, Q&A-style EDA, clustering with K-Means, model evaluation, and interactive Plotly visualizations** to better understand metro coverage and station groupings.

---

## 📁 Project Structure

```text
├── data/
│   ├── riyadh_metro_raw_1.csv          # Raw dataset (before cleaning)
|   ├── riyadh_metro_raw_2.csv   
│   └── riyadh_metro_clean.csv          # Cleaned dataset (after preprocessing)
├── notebooks/
│   ├── 01_clean_merge_addcols.ipynb    # Data cleaning, merging & feature engineering
│   └── 02_eda_and_model.ipynb          # EDA (Q&A) & clustering models
├── outputs/
│   └── riyadh_metro_with_clusters.csv  # (Optional) Data with cluster assignments
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation
```

---

## 🎯 Objectives

- **Clean & merge** raw datasets into a consistent, usable format
- **Explore** metro stations with a Q&A-style EDA (questions → code → insights)
- Analyze **ratings distribution**, top/bottom-rated stations, and overall coverage
- **Cluster** stations geographically using K-Means
- **Visualize** results with Plotly density heatmaps and interactive cluster maps

---

## 📊 Dataset

- **Source**: Manually cleaned & merged Riyadh Metro Stations Dataset
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

## 📈 Key Insights

- Central and northern Riyadh show the **highest station density**
- **Station ratings** vary significantly across metro lines and areas
- **Top-rated stations** are concentrated in key hubs and business districts
- **K-Means clustering** identified natural geographic zones of station coverage
- **Plotly heatmaps** highlight city-wide coverage patterns and gaps

---

## ⚙️ Requirements

- pandas  
- numpy  
- scikit-learn  
- plotly  
- matplotlib  

---

## 📄 License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute it with attribution.

---

## ✨ Acknowledgments

Developed as part of a data science learning journey, focusing on practical EDA and clustering techniques with real-world Riyadh Metro data.
