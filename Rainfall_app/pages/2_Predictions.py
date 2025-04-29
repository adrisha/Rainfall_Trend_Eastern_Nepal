import streamlit as st # type: ignore
import pandas as pd
from utils.data_utils import load_feature_data, load_reg_model, load_clf_model
from utils.visualization_utils import plot_time_series
import uuid

# Set page configuration
st.set_page_config(page_title="Rainfall Prediction Dashboard", layout="centered", initial_sidebar_state="expanded")

# Custom CSS for attractive and responsive styling
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
        padding: 20px;
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        transform: translateY(-2px);
    }
    .card {
        background-color: white;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        width: 100%;
        box-sizing: border-box;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-box {
        background-color: #eff6ff;
        padding: 16px;
        border-radius: 8px;
        text-align: center;
        margin: 10px 0;
        width: 100%;
        box-sizing: border-box;
    }
    h1 {
        color: #1e3a8a;
        font-weight: 700;
        font-size: clamp(1.8rem, 5vw, 2.5rem);
    }
    h2 {
        color: #1e40af;
        font-weight: 600;
        font-size: clamp(1.4rem, 4vw, 1.8rem);
    }
    /* Mobile adjustments */
    @media (max-width: 600px) {
        .main {
            padding: 10px;
        }
        .card {
            padding: 16px;
        }
        .metric-box {
            padding: 12px;
            margin: 8px 0;
        }
        .stButton>button {
            padding: 8px 16px;
            font-size: 0.9rem;
        }
        .sidebar .sidebar-content {
            padding: 10px;
        }
        .st-columns > div {
            margin-bottom: 10px;
        }
        .stPlotlyChart {
            width: 100% !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Main title
with st.container():
    st.title("🌧️ Rainfall Prediction Dashboard")
    st.markdown("Analyze historical rainfall data and make real-time predictions with a modern interface.")

# Load data and models
data = load_feature_data()
reg_model = load_reg_model()
clf_model = load_clf_model()

# Create station options
station_options = data[['station_id', 'station_name_x']].drop_duplicates().set_index('station_id')['station_name_x'].to_dict()
display_options = [f"{station_options.get(sid, 'Unknown')} (ID: {sid})" for sid in data['station_id'].unique()]

# Sidebar filters
with st.sidebar:
    st.markdown('<div class="card" role="region" aria-label="Sidebar Filters">', unsafe_allow_html=True)
    st.header("📍 Filters")
    selected_display = st.multiselect("Select Stations", options=display_options, default=display_options, 
                                    help="Choose one or more stations to analyze")
    selected_stations = [sid for sid, name in station_options.items() for opt in selected_display if name in opt or f"ID: {sid}" in opt]
    date_range = st.date_input("Select Date Range", 
                              [pd.to_datetime(data['date']).min(), pd.to_datetime(data['date']).max()],
                              help="Select the date range for historical data")
    st.markdown('</div>', unsafe_allow_html=True)

# Filter data
filtered_data = data[
    (data['station_id'].isin(selected_stations)) &
    (pd.to_datetime(data['date']) >= pd.to_datetime(date_range[0])) &
    (pd.to_datetime(data['date']) <= pd.to_datetime(date_range[1]))
]

# Feature columns
feature_columns = [
    'ele(meter)', 'lat(deg)', 'lon(deg)', 'year', 'month', 'day_of_year',
    'yearly_rainfall', 'monthly_rainfall', 'prev_day_rainfall',
    'rolling_mean_7d', 'station_name_x_encoded', 'log_rainfall_sum',
    'log_monthly_rainfall', 'log_prev_day_rainfall', 'log_rolling_mean_7d',
    'pca_component_1', 'pca_component_2', 'pca_component_3'
]
required_columns = ['date', 'rainfall_sum'] + feature_columns

# Main content
if not filtered_data.empty:
    # Filter data to include only required columns
    filtered_data = filtered_data[required_columns]

    # Handle missing columns
    missing_columns = [col for col in required_columns if col not in filtered_data.columns]
    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
        for col in missing_columns:
            filtered_data[col] = 0

    # Historical Predictions Section
    with st.container():
        st.markdown('<div class="card" role="region" aria-label="Historical Predictions Section">', unsafe_allow_html=True)
        st.subheader("📊 Historical Predictions")
        
        # Debug information in expanders for cleaner UI
        with st.expander("Debug Information"):
            st.write(f"Filtered Data Columns: {filtered_data.columns.tolist()}")
            st.write(f"Filtered Data Shape: {filtered_data.shape}")

        # Generate predictions
        if 'pred_rainfall' not in filtered_data.columns:
            try:
                filtered_data['pred_rainfall'] = reg_model.predict(filtered_data[feature_columns])
            except Exception as e:
                st.error(f"Error generating predictions: {str(e)}")
                st.code("""
import pickle
with open('../Rainfall_app/models/best_random_forest_regressor_model.pkl', 'rb') as f:
    model = pickle.load(f)
print(model.feature_names_in_)
                """)
                required_columns = ['date', 'rainfall_sum']
        else:
            required_columns.append('pred_rainfall')

        # Plot time series
        if all(col in filtered_data.columns for col in required_columns):
            y_columns = ['rainfall_sum', 'pred_rainfall'] if 'pred_rainfall' in filtered_data.columns else ['rainfall_sum']
            title = "Actual vs Predicted Rainfall" if 'pred_rainfall' in filtered_data.columns else "Actual Rainfall"
            try:
                fig = plot_time_series(filtered_data, y_columns=y_columns, title=title)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error plotting time series: {str(e)}")
        else:
            st.error(f"Missing required columns: {[col for col in required_columns if col not in filtered_data.columns]}")
        st.markdown('</div>', unsafe_allow_html=True)
else:
    with st.container():
        st.markdown('<div class="card" role="region" aria-label="No Data Warning">', unsafe_allow_html=True)
        st.warning("No data available for the selected filters. Please adjust your filters.")
        st.markdown('</div>', unsafe_allow_html=True)

# New Prediction Section
with st.container():
    st.markdown('<div class="card" role="region" aria-label="New Prediction Section">', unsafe_allow_html=True)
    st.subheader("🔮 Make a New Prediction")
    
    # Input form with expanders for better mobile usability
    input_data = {}
    with st.expander("Geographical Features"):
        for col in ['ele(meter)', 'lat(deg)', 'lon(deg)']:
            if col in data.columns:
                input_data[col] = st.number_input(f"{col}", 
                                                value=float(data[col].mean()), 
                                                step=0.1,
                                                key=f"input_{col}_{uuid.uuid4()}")
    with st.expander("Temporal Features"):
        for col in ['year', 'month', 'day_of_year']:
            if col in data.columns:
                input_data[col] = st.number_input(f"{col}", 
                                                value=float(data[col].mean()), 
                                                step=0.1,
                                                key=f"input_{col}_{uuid.uuid4()}")
    with st.expander("Rainfall Features"):
        for col in ['yearly_rainfall', 'monthly_rainfall', 'prev_day_rainfall', 'rolling_mean_7d']:
            if col in data.columns:
                input_data[col] = st.number_input(f"{col}", 
                                                value=float(data[col].mean()), 
                                                step=0.1,
                                                key=f"input_{col}_{uuid.uuid4()}")
    with st.expander("Encoded and Transformed Features"):
        for col in ['station_name_x_encoded', 'log_rainfall_sum', 'log_monthly_rainfall', 
                    'log_prev_day_rainfall', 'log_rolling_mean_7d', 'pca_component_1', 
                    'pca_component_2', 'pca_component_3']:
            if col in data.columns:
                input_data[col] = st.number_input(f"{col}", 
                                                value=float(data[col].mean()), 
                                                step=0.1,
                                                key=f"input_{col}_{uuid.uuid4()}")
    
    if st.button("Predict", key="predict_button"):
        input_df = pd.DataFrame([input_data])
        try:
            reg_pred = reg_model.predict(input_df)[0]
            clf_pred = clf_model.predict(input_df)[0]
            clf_proba = clf_model.predict_proba(input_df)[0][1]
            
            # Display predictions in metric boxes
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("Predicted Rainfall", f"{reg_pred:.2f} mm")
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("Extreme Rainfall Prob.", f"{clf_proba:.2%}")
                st.markdown('</div>', unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("Extreme Rainfall", "Yes" if clf_pred else "No")
                st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.code("""
import pickle
with open('../Rainfall_app/models/best_random_forest_regressor_model.pkl', 'rb') as f:
    model = pickle.load(f)
print(model.feature_names_in_)
            """)
    st.markdown('</div>', unsafe_allow_html=True)