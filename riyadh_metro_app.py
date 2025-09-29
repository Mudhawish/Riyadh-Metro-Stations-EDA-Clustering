import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
import math

# Set page configuration
st.set_page_config(
    page_title="Riyadh Metro Dashboard",
    page_icon="🚇",
    layout="wide",
)

# Cache the data loading function for better performance
@st.cache_data
def load_data():
    """
    Load and prepare the Riyadh Metro stations data
    """
    # Load the data
    df = pd.read_csv('Cleaned_Dataset_Riyadh-Metro-Stations.csv')
    
    # Replace missing values in Rating and Number_of_Ratings with 0
    df['Rating'] = df['Rating'].fillna(0)
    df['Number_of_Ratings'] = df['Number_of_Ratings'].fillna(0)
    
    # Create Metro_Direction from Quadrant if not present
    if 'Metro_Direction' not in df.columns:
        # Map quadrant to direction
        quadrant_to_direction = {
            'Northeast': 'North-East',
            'Northwest': 'North-West',
            'Southeast': 'South-East',
            'Southwest': 'South-West',
            'Unclassified': 'Central'
        }
        df['Metro_Direction'] = df['Quadrant'].map(quadrant_to_direction)
    
    # Create Metro_Line_Color from 'Metro line name' if needed
    if 'Metro_Line_Color' not in df.columns:
        df['Metro_Line_Color'] = df['Metro line name']
    
    # Create Station_Type from 'Station type' if needed
    if 'Station_Type' not in df.columns:
        df['Station_Type'] = df['Station type']
    
    return df

# Haversine distance function to calculate distances in kilometers between coordinates
@st.cache_data
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

@st.cache_data
def compute_distance_matrix(df):
    """
    Compute a distance matrix for the given dataframe using Haversine distance
    """
    n = len(df)
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            dist = haversine_distance(
                df.iloc[i]['Latitude'], df.iloc[i]['Longitude'],
                df.iloc[j]['Latitude'], df.iloc[j]['Longitude']
            )
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    
    return dist_matrix

# Load data
df = load_data()

# Sidebar with filters
st.sidebar.title("Filters")
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/8/8d/Riyadh_Metro_logo.png", width=100)
st.sidebar.markdown("### Global Filters")

# Metro Direction Filter
all_directions = sorted(df['Metro_Direction'].unique().tolist())
selected_directions = st.sidebar.multiselect(
    "Metro Direction",
    options=all_directions,
    default=all_directions
)

# Metro Line Filter
all_lines = sorted(df['Metro_Line_Color'].unique().tolist())
selected_lines = st.sidebar.multiselect(
    "Metro Line",
    options=all_lines,
    default=all_lines
)

# Station Type Filter
all_types = sorted(df['Station_Type'].unique().tolist())
selected_types = st.sidebar.multiselect(
    "Station Type",
    options=all_types,
    default=all_types
)

# Rating Filter
min_rating = float(df['Rating'].min())
max_rating = float(df['Rating'].max())
selected_rating_range = st.sidebar.slider(
    "Station Rating",
    min_value=min_rating,
    max_value=max_rating,
    value=(min_rating, max_rating),
    step=0.1
)

# Number of Ratings Filter
min_num_ratings = float(df['Number_of_Ratings'].min())
max_num_ratings = float(df['Number_of_Ratings'].max())
selected_num_ratings_range = st.sidebar.slider(
    "Number of Ratings",
    min_value=min_num_ratings,
    max_value=max_num_ratings,
    value=(min_num_ratings, max_num_ratings),
    step=1.0
)

# Filter the dataframe based on selections
filtered_df = df[
    (df['Metro_Direction'].isin(selected_directions)) &
    (df['Metro_Line_Color'].isin(selected_lines)) &
    (df['Station_Type'].isin(selected_types)) &
    (df['Rating'] >= selected_rating_range[0]) & 
    (df['Rating'] <= selected_rating_range[1]) &
    (df['Number_of_Ratings'] >= selected_num_ratings_range[0]) & 
    (df['Number_of_Ratings'] <= selected_num_ratings_range[1]) &
    df['Latitude'].notna() & df['Longitude'].notna()
]

# Create a multi-page structure
pages = ["Geographical Visualization", "EDA Visualizations", "Station Clustering Analysis"]
page = st.radio("Select Page", pages, horizontal=True)

if page == "Geographical Visualization":
    st.title("Riyadh Metro Geographical Dashboard")
    st.markdown("### Interactive maps for exploring metro station locations")
    
    # Create sub-pages for different map views
    map_views = ["Station Scatter Maps", "Density Heatmaps"]
    selected_map = st.selectbox("Select Map View", map_views)
    
    if selected_map == "Station Scatter Maps":
        # Color options
        color_options = {
            'Metro Line': 'Metro_Line_Color',
            'Rating': 'Rating',
            'Geographic Direction': 'Metro_Direction'
        }
        
        color_by = st.selectbox("Color Stations By:", list(color_options.keys()))
        color_column = color_options[color_by]
        
        if not filtered_df.empty:
            if color_by == 'Rating':
                fig = px.scatter_mapbox(
                    filtered_df,
                    lat="Latitude", lon="Longitude",
                    hover_name="Station_Name",
                    color="Rating",
                    color_continuous_scale="Viridis",
                    zoom=10, height=600,
                    hover_data={
                        "Station_Name": True,
                        "Rating": True,
                        "Number_of_Ratings": True,
                        "Metro_Direction": True,
                        "Metro_Line_Color": True,
                        "Station_Type": True
                    }
                )
            else:
                fig = px.scatter_mapbox(
                    filtered_df,
                    lat="Latitude", lon="Longitude",
                    hover_name="Station_Name",
                    color=color_column,
                    zoom=10, height=600,
                    hover_data={
                        "Station_Name": True,
                        "Rating": True,
                        "Number_of_Ratings": True,
                        "Metro_Direction": True,
                        "Metro_Line_Color": True,
                        "Station_Type": True
                    }
                )
            
            fig.update_layout(mapbox_style="carto-darkmatter", margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No stations match your filter criteria.")
    
    elif selected_map == "Density Heatmaps":
        # Heatmap type options
        heatmap_type = st.radio("Heatmap Type:", ["Station Density", "Rating-Weighted Density"])
        
        if not filtered_df.empty:
            if heatmap_type == "Station Density":
                fig = px.density_mapbox(
                    filtered_df,
                    lat="Latitude", lon="Longitude",
                    radius=15,
                    zoom=10, height=600,
                    hover_name="Station_Name",
                )
            else:  # Rating-Weighted Density
                fig = px.density_mapbox(
                    filtered_df.dropna(subset=["Rating"]),
                    lat="Latitude", lon="Longitude",
                    z="Rating",
                    radius=20, 
                    zoom=10, height=600,
                    hover_name="Station_Name",
                )
            
            fig.update_layout(mapbox_style="carto-darkmatter", margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No stations match your filter criteria.")

elif page == "EDA Visualizations":
    st.title("Riyadh Metro EDA Visualizations")
    st.markdown("### Summary charts and insights about the metro stations")
    
    # Create tabs for better organization
    overview_tab, distributions_tab, rankings_tab = st.tabs(["Station Overview", "Distributions", "Rankings & Top Stations"])
    
    with overview_tab:
        st.subheader("Metro Station Overview")
        
        # Create a layout with 2 columns for the first row
        col1, col2 = st.columns(2)
        
        with col1:
            # Average Rating by Direction
            st.subheader("Average Station Rating by Geographic Direction")
            
            if not filtered_df.empty:
                avg_by_direction = filtered_df.groupby('Metro_Direction')['Rating'].mean().reset_index()
                fig_avg_rating = px.bar(
                    avg_by_direction, 
                    x='Metro_Direction', 
                    y='Rating',
                    color='Metro_Direction',
                    labels={'Rating': 'Average Rating', 'Metro_Direction': 'Geographic Direction'},
                    text_auto='.2f'
                )
                fig_avg_rating.update_layout(
                    height=400,
                    template="plotly_dark",
                    plot_bgcolor='rgba(0, 0, 0, 0)',
                    paper_bgcolor='rgba(0, 0, 0, 0)'
                )
                st.plotly_chart(fig_avg_rating, use_container_width=True)
            else:
                st.warning("No data available for this visualization with current filters.")
        
        with col2:
            # Station Distribution by Line
            st.subheader("Number of Stations Per Metro Line")
            
            if not filtered_df.empty:
                stations_by_line = filtered_df.groupby('Metro_Line_Color').size().reset_index(name='Count')
                fig_line_dist = px.bar(
                    stations_by_line, 
                    x='Metro_Line_Color', 
                    y='Count',
                    color='Metro_Line_Color',
                    labels={'Count': 'Number of Stations', 'Metro_Line_Color': 'Metro Line'},
                    text_auto=True
                )
                fig_line_dist.update_layout(
                    height=400,
                    template="plotly_dark",
                    plot_bgcolor='rgba(0, 0, 0, 0)',
                    paper_bgcolor='rgba(0, 0, 0, 0)'
                )
                st.plotly_chart(fig_line_dist, use_container_width=True)
            else:
                st.warning("No data available for this visualization with current filters.")
        
        # Direction Distribution and Station Type in a new row
        col1, col2 = st.columns(2)
        
        with col1:
            # Metro Direction Overview
            st.subheader("Percentage of Stations by Geographic Direction")
            
            if not filtered_df.empty:
                direction_counts = filtered_df['Metro_Direction'].value_counts().reset_index()
                direction_counts.columns = ['Metro_Direction', 'Count']
                fig_pie = px.pie(
                    direction_counts, 
                    values='Count', 
                    names='Metro_Direction',
                    hole=0.4
                )
                fig_pie.update_layout(
                    height=450,
                    template="plotly_dark",
                    plot_bgcolor='rgba(0, 0, 0, 0)',
                    paper_bgcolor='rgba(0, 0, 0, 0)'
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.warning("No data available for this visualization with current filters.")
        
        with col2:
            # Additional Visualization: Average Rating by Station Type
            st.subheader("Average Rating by Station Type")
            
            if not filtered_df.empty:
                avg_by_type = filtered_df.groupby('Station_Type')['Rating'].mean().reset_index()
                fig_type_rating = px.bar(
                    avg_by_type, 
                    x='Station_Type', 
                    y='Rating',
                    color='Station_Type',
                    labels={'Rating': 'Average Rating', 'Station_Type': 'Station Type'},
                    text_auto='.2f'
                )
                fig_type_rating.update_layout(
                    height=450,
                    template="plotly_dark",
                    plot_bgcolor='rgba(0, 0, 0, 0)',
                    paper_bgcolor='rgba(0, 0, 0, 0)'
                )
                st.plotly_chart(fig_type_rating, use_container_width=True)
            else:
                st.warning("No data available for this visualization with current filters.")
    
    with distributions_tab:
        st.subheader("Distribution Analysis")
        
        # Create columns for the distributions
        col1, col2 = st.columns(2)
        
        with col1:
            # Rating Distribution with KDE
            st.subheader("Distribution of Station Ratings")
            
            if not filtered_df.empty:
                fig_rating_dist = px.histogram(
                    filtered_df,
                    x="Rating",
                    nbins=20,
                    marginal="rug",  # Add a rug plot for additional detail
                    histnorm="probability density",  # Normalize for KDE comparison
                    opacity=0.7,
                    color_discrete_sequence=["#636EFA"]
                )
                
                # Create a KDE curve using numpy's histogram and a gaussian filter
                from scipy import stats
                
                # Get the values for the KDE curve
                kde = stats.gaussian_kde(filtered_df['Rating'].dropna())
                x_range = np.linspace(min(filtered_df['Rating']), max(filtered_df['Rating']), 100)
                y_kde = kde(x_range)
                
                # Add KDE curve as a line trace
                fig_rating_dist.add_trace(
                    dict(
                        type='scatter',
                        x=x_range,
                        y=y_kde,
                        mode='lines',
                        line=dict(color='#EF553B', width=3),
                        name='KDE'
                    )
                )
                
                fig_rating_dist.update_layout(
                    height=450,
                    template="plotly_dark",
                    plot_bgcolor='rgba(0, 0, 0, 0)',
                    paper_bgcolor='rgba(0, 0, 0, 0)',
                    xaxis_title="Station Rating",
                    yaxis_title="Density",
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                # Update trace names for better legend
                fig_rating_dist.data[0].name = "Histogram"
                fig_rating_dist.data[1].name = "KDE"
                
                st.plotly_chart(fig_rating_dist, use_container_width=True)
            else:
                st.warning("No data available for this visualization with current filters.")
        
        with col2:
            # Number of Ratings Distribution with KDE
            st.subheader("Distribution of Number of Ratings")
            
            if not filtered_df.empty:
                fig_numrat_dist = px.histogram(
                    filtered_df,
                    x="Number_of_Ratings",
                    nbins=20,
                    marginal="rug",
                    histnorm="probability density",
                    opacity=0.7,
                    color_discrete_sequence=["#636EFA"]
                )
                
                # Create a KDE curve using numpy's histogram and a gaussian filter
                from scipy import stats
                
                # Get the values for the KDE curve
                kde = stats.gaussian_kde(filtered_df['Number_of_Ratings'].dropna())
                x_range = np.linspace(min(filtered_df['Number_of_Ratings']), max(filtered_df['Number_of_Ratings']), 100)
                y_kde = kde(x_range)
                
                # Add KDE curve as a line trace
                fig_numrat_dist.add_trace(
                    dict(
                        type='scatter',
                        x=x_range,
                        y=y_kde,
                        mode='lines',
                        line=dict(color='#EF553B', width=3),
                        name='KDE'
                    )
                )
                
                fig_numrat_dist.update_layout(
                    height=450,
                    template="plotly_dark",
                    plot_bgcolor='rgba(0, 0, 0, 0)',
                    paper_bgcolor='rgba(0, 0, 0, 0)',
                    xaxis_title="Number of Ratings",
                    yaxis_title="Density",
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                # Update trace names for better legend
                fig_numrat_dist.data[0].name = "Histogram"
                fig_numrat_dist.data[1].name = "KDE"
                
                st.plotly_chart(fig_numrat_dist, use_container_width=True)
            else:
                st.warning("No data available for this visualization with current filters.")
    
    with rankings_tab:
        st.subheader("Top Stations Analysis")
        
        # Top 10 Most Reviewed Stations
        st.subheader("Top 10 Stations by Number of Reviews")
        
        if not filtered_df.empty:
            # Sort by Number_of_Ratings and get top 10
            top_reviewed = filtered_df.sort_values(by='Number_of_Ratings', ascending=False).head(10)
            
            fig_top_reviewed = px.bar(
                top_reviewed,
                y='Station_Name',  # Station names on y-axis for horizontal bars
                x='Number_of_Ratings',
                orientation='h',  # Horizontal bars
                color='Metro_Line_Color',  # Color by metro line
                hover_data=['Rating', 'Station_Type', 'Metro_Direction'],
                labels={
                    'Number_of_Ratings': 'Number of Reviews',
                    'Station_Name': 'Station',
                    'Metro_Line_Color': 'Metro Line'
                },
                height=600
            )
            
            fig_top_reviewed.update_layout(
                yaxis={'categoryorder':'total ascending'},  # Sort by value
                template="plotly_dark",
                plot_bgcolor='rgba(0, 0, 0, 0)',
                paper_bgcolor='rgba(0, 0, 0, 0)',
                xaxis_title="Number of Reviews",
                yaxis_title=""
            )
            
            st.plotly_chart(fig_top_reviewed, use_container_width=True)
            
            # Add a table showing the top 10 stations with their ratings
            st.subheader("Details of Top Reviewed Stations")
            top_reviewed_table = top_reviewed[['Station_Name', 'Number_of_Ratings', 'Rating', 'Metro_Line_Color', 'Station_Type']]
            top_reviewed_table.columns = ['Station', 'Number of Reviews', 'Rating', 'Metro Line', 'Station Type']
            st.dataframe(top_reviewed_table, use_container_width=True)
        else:
            st.warning("No data available for this visualization with current filters.")

elif page == "Station Clustering Analysis":
    st.title("Riyadh Metro Station Clustering Analysis")
    st.markdown("### Discover station patterns and groupings based on geographical coordinates")
    
    # Create tabs for different clustering algorithms
    kmeans_tab, dbscan_tab = st.tabs(["K-Means Clustering", "DBSCAN Clustering"])
    
    with kmeans_tab:
        st.header("K-Means Clustering")
        st.markdown("Group stations into clusters based on their geographical proximity.")
        
        if not filtered_df.empty and len(filtered_df) >= 3:  # Need at least 3 points for meaningful clustering
            # Control for number of clusters (k)
            k_clusters = st.slider("Number of clusters (k)", 2, min(10, len(filtered_df)-1), 5)
            
            # Extract coordinates and standardize them for K-means
            coords = filtered_df[['Latitude', 'Longitude']].values
            scaler = StandardScaler()
            coords_scaled = scaler.fit_transform(coords)
            
            # Run K-means clustering
            kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
            filtered_df['Cluster'] = kmeans.fit_predict(coords_scaled)
            
            # Calculate silhouette score for the current k
            silhouette_avg = silhouette_score(coords_scaled, filtered_df['Cluster']) if k_clusters > 1 and k_clusters < len(filtered_df) else 0
            
            # Display silhouette score
            st.metric("Silhouette Score", f"{silhouette_avg:.3f}", 
                      delta="Higher is better (max 1.0)", delta_color="normal")
            
            # Calculate silhouette scores for a range of k values
            st.subheader("Optimal k Value Analysis")
            
            # Only compute if we have enough data points
            if len(filtered_df) > 10:
                with st.spinner("Computing optimal k values..."):
                    k_range = range(2, min(11, len(filtered_df)))
                    silhouette_scores = []
                    
                    for k in k_range:
                        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
                        cluster_labels = kmeans_temp.fit_predict(coords_scaled)
                        
                        # Calculate silhouette score
                        silhouette_avg = silhouette_score(coords_scaled, cluster_labels)
                        silhouette_scores.append(silhouette_avg)
                    
                    # Create optimal k plot
                    k_df = pd.DataFrame({
                        'k': list(k_range),
                        'Silhouette Score': silhouette_scores
                    })
                    
                    fig_k = px.line(
                        k_df, x='k', y='Silhouette Score',
                        markers=True, 
                        title="Silhouette Scores for Different k Values"
                    )
                    fig_k.update_layout(height=400)
                    st.plotly_chart(fig_k, use_container_width=True)
                    
                    # Recommend optimal k
                    optimal_k = k_range[silhouette_scores.index(max(silhouette_scores))]
                    st.info(f"💡 The optimal number of clusters based on silhouette score is k = {optimal_k}")
            
            # Display cluster map
            st.subheader("Cluster Map")
            
            fig_clusters = px.scatter_mapbox(
                filtered_df,
                lat="Latitude", lon="Longitude",
                hover_name="Station_Name",
                color="Cluster",
                color_continuous_scale=None,  # Use categorical colors
                zoom=10, height=600,
                hover_data={
                    "Station_Name": True,
                    "Rating": True,
                    "Metro_Line_Color": True,
                    "Metro_Direction": True,
                    "Cluster": False  # Hide duplicate cluster info
                }
            )
            fig_clusters.update_layout(mapbox_style="carto-darkmatter", margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig_clusters, use_container_width=True)
            
            # Display cluster summary
            st.subheader("Cluster Summary")
            
            # Aggregate stats by cluster
            cluster_summary = filtered_df.groupby('Cluster').agg({
                'Station_Name': 'count',
                'Rating': 'mean',
                'Number_of_Ratings': 'mean',
                'Metro_Line_Color': lambda x: x.value_counts().index[0] if not x.empty else 'None',
                'Metro_Direction': lambda x: x.value_counts().index[0] if not x.empty else 'None',
                'Station_Type': lambda x: x.value_counts().index[0] if not x.empty else 'None'
            }).reset_index()
            
            # Rename columns for clarity
            cluster_summary.columns = [
                'Cluster', 'Station Count', 'Avg Rating', 'Avg Number of Ratings',
                'Dominant Line', 'Dominant Direction', 'Dominant Station Type'
            ]
            
            # Display summary
            st.dataframe(cluster_summary, use_container_width=True)
        else:
            st.warning("Insufficient data for clustering. Please adjust filters to include more stations.")
    
    with dbscan_tab:
        st.header("DBSCAN Clustering")
        st.markdown("Identify dense clusters of stations and detect outliers.")
        
        if not filtered_df.empty and len(filtered_df) >= 4:  # Need at least 4 points for DBSCAN with min_samples=4
            # DBSCAN parameters
            col1, col2 = st.columns(2)
            with col1:
                eps_km = st.number_input(
                    "Maximum distance between stations (eps, km)",
                    min_value=0.1,
                    max_value=10.0,
                    value=1.5,
                    step=0.1,
                    help="Maximum distance between two stations to be considered neighbors (in kilometers)"
                )
            
            with col2:
                min_samples = st.number_input(
                    "Minimum stations per cluster",
                    min_value=2,
                    max_value=len(filtered_df) - 1,
                    value=min(4, len(filtered_df) - 1),
                    step=1,
                    help="Minimum number of stations required to form a cluster"
                )
            
            # Compute distance matrix using Haversine distance
            with st.spinner("Computing distance matrix..."):
                dist_matrix = compute_distance_matrix(filtered_df)
                
                # Run DBSCAN clustering
                dbscan = DBSCAN(eps=eps_km, min_samples=min_samples, metric='precomputed')
                filtered_df['Cluster'] = dbscan.fit_predict(dist_matrix)
                
                # Count clusters and noise points
                n_clusters = len(set(filtered_df['Cluster'])) - (1 if -1 in filtered_df['Cluster'].values else 0)
                n_noise = list(filtered_df['Cluster']).count(-1)
            
            # Display metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Number of Clusters", n_clusters)
            with col2:
                st.metric("Number of Noise Points", n_noise, 
                         delta=f"{n_noise/len(filtered_df):.1%} of stations" if n_noise > 0 else "No noise points",
                         delta_color="inverse")
            
            # Display cluster map
            st.subheader("Cluster Map")
            
            # Create a copy of the dataframe with cluster labels as strings for better coloring
            plot_df = filtered_df.copy()
            plot_df['Cluster Label'] = plot_df['Cluster'].apply(
                lambda x: f"Cluster {x}" if x >= 0 else "Noise"
            )
            
            fig_dbscan = px.scatter_mapbox(
                plot_df,
                lat="Latitude", lon="Longitude",
                hover_name="Station_Name",
                color="Cluster Label",
                color_discrete_map={
                    "Noise": "gray",  # Make noise points gray
                },
                zoom=10, height=600,
                hover_data={
                    "Station_Name": True,
                    "Rating": True,
                    "Metro_Line_Color": True,
                    "Metro_Direction": True,
                    "Cluster Label": True
                }
            )
            
            fig_dbscan.update_layout(mapbox_style="carto-darkmatter", margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig_dbscan, use_container_width=True)
            
            # Display cluster summary (excluding noise points)
            st.subheader("Cluster Summary")
            
            if n_clusters > 0:
                # Filter out noise points
                cluster_data = filtered_df[filtered_df['Cluster'] >= 0]
                
                # Aggregate stats by cluster
                cluster_summary = cluster_data.groupby('Cluster').agg({
                    'Station_Name': 'count',
                    'Rating': 'mean',
                    'Number_of_Ratings': 'mean',
                    'Metro_Line_Color': lambda x: x.value_counts().index[0] if not x.empty else 'None',
                    'Metro_Direction': lambda x: x.value_counts().index[0] if not x.empty else 'None',
                    'Station_Type': lambda x: x.value_counts().index[0] if not x.empty else 'None'
                }).reset_index()
                
                # Rename columns for clarity
                cluster_summary.columns = [
                    'Cluster', 'Station Count', 'Avg Rating', 'Avg Number of Ratings',
                    'Dominant Line', 'Dominant Direction', 'Dominant Station Type'
                ]
                
                # Display summary
                st.dataframe(cluster_summary, use_container_width=True)
            else:
                st.info("No clusters were formed with the current parameters. Try adjusting eps or min_samples.")
        else:
            st.warning("Insufficient data for DBSCAN clustering. Please adjust filters to include more stations.")

# Footer
st.markdown("---")
st.markdown("### Riyadh Metro Stations Dashboard - Data Analysis Project")
st.markdown("Source: Riyadh Metro Stations Dataset")