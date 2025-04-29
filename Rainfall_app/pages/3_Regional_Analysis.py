import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium

# Page configuration
st.set_page_config(page_title="Performance Dashboard", layout="wide")

# Custom CSS for styling (responsive)
st.markdown("""
    <style>
    body {
        font-family: 'Segoe UI', sans-serif;
        font-size: 16px;
    }
    .main {
        padding: 1rem;
    }
    .card {
        background-color: #f0f2f6;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    .metric-card {
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Cached data loaders
@st.cache_data
def load_regional_performance_regression():
    return pd.read_csv('data/regional_performance_regression.csv', index_col='station_id')

@st.cache_data
def load_regional_performance_classification():
    return pd.read_csv('data/regional_performance_classification.csv', index_col='station_id')

@st.cache_data
def load_station_location():
    return pd.read_csv('data/station_location.csv', index_col='station_id')

@st.cache_data
def load_feature_data():
    df = pd.read_csv('data/feature_data.csv', index_col='station_id')
    return df.rename(columns={'lat(deg)': 'lat', 'lon(deg)': 'lon'})

# Load data
reg_perf = load_regional_performance_regression()
clf_perf = load_regional_performance_classification()
locations_df = load_station_location()
feature_data = load_feature_data()

# Merge for location-aware performance
locations_df = locations_df.join(reg_perf, how='left')
locations_df = locations_df.join(clf_perf, how='left')

# Main layout
st.title("📊 Regional Performance Dashboard")

with st.container():
    st.markdown("### 📍 Overview Metrics")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Regression: MAE vs MSE")
        fig1 = px.scatter(reg_perf.reset_index(), x='mae', y='mse',
                         hover_name='station_id', trendline='ols',
                         title='MAE vs MSE')
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown("#### Classification: Accuracy vs F1")
        fig2 = px.scatter(clf_perf.reset_index(), x='accuracy', y='f1_score',
                         hover_name='station_id', trendline='ols',
                         title='Accuracy vs F1')
        st.plotly_chart(fig2, use_container_width=True)

with st.container():
    st.markdown("### 🗺️ Station-wise Metrics Map")
    st.markdown("Use this map to explore each station's geographic performance.")

    default_coords = [feature_data['lat'].mean(), feature_data['lon'].mean()]
    map_fig = folium.Map(location=default_coords, zoom_start=6)

    for station in locations_df.index:
        coords = feature_data.loc[station] if station in feature_data.index else None
        if coords is not None:
            folium.Marker(
                location=[coords['lat'], coords['lon']],
                popup=(
                    f"<b>Station:</b> {station}<br>"
                    f"<b>MAE:</b> {reg_perf.at[station, 'mae']:.2f}<br>"
                    f"<b>Accuracy:</b> {clf_perf.at[station, 'accuracy']:.2f}"
                ),
                tooltip=station,
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(map_fig)

    st_data = st_folium(map_fig, width=700, height=500)

with st.container():
    st.markdown("### 📌 Detailed Metrics Table")
    selected_station = st.selectbox("Select a station to view metrics:", options=locations_df.index)

    if selected_station in locations_df.index:
        st.markdown(f"#### Metrics for Station: `{selected_station}`")
        reg_metrics = reg_perf.loc[selected_station]
        clf_metrics = clf_perf.loc[selected_station]

        col1, col2 = st.columns(2)

        with col1:
            st.metric(label="Regression MAE", value=f"{reg_metrics['mae']:.3f}")
            st.metric(label="Regression MSE", value=f"{reg_metrics['mse']:.3f}")

        with col2:
            st.metric(label="Classification Accuracy", value=f"{clf_metrics['accuracy']:.3f}")
            st.metric(label="Classification F1 Score", value=f"{clf_metrics['f1_score']:.3f}")

        st.markdown("##### Full Metric Summary")
        all_metrics = pd.concat([reg_metrics, clf_metrics])
        st.dataframe(all_metrics.to_frame(name='Value'))
    else:
        st.error("Selected station not found in the dataset.")
