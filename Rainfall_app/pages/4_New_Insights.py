import os
import streamlit as st
import pandas as pd
from textblob import TextBlob
from utils.data_utils import load_nlp_results, load_lda_topics
import plotly.express as px

# Ensure working directory is correct
os.chdir(os.path.dirname(__file__))

# Set page configuration
st.set_page_config(page_title="News Insights Dashboard", layout="centered", initial_sidebar_state="expanded")

# Custom CSS for attractive and responsive styling
st.markdown("""
    <style>
    .main { background-color: #f8fafc; padding: 20px; }
    .stButton>button {
        background-color: #2563eb; color: white; border-radius: 8px;
        padding: 12px 24px; font-weight: 500; transition: all 0.3s;
    }
    .stButton>button:hover { background-color: #1d4ed8; transform: translateY(-2px); }
    .card {
        background-color: white; padding: 24px; border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px;
    }
    .metric-box {
        background-color: #eff6ff; padding: 16px; border-radius: 8px;
        text-align: center; margin: 10px 0;
    }
    h1 { color: #1e3a8a; font-weight: 700; font-size: clamp(1.8rem, 5vw, 2.5rem); }
    h2 { color: #1e40af; font-weight: 600; font-size: clamp(1.4rem, 4vw, 1.8rem); }
    .stDataFrame { border-radius: 8px; overflow: hidden; }
    .stDataFrame .dataframe { word-wrap: break-word; white-space: pre-wrap; }
    .topics-box {
        background-color: #f1f5f9; padding: 16px; border-radius: 8px;
        word-wrap: break-word; white-space: pre-wrap;
    }
    @media (max-width: 600px) {
        .main { padding: 10px; }
        .card { padding: 16px; }
        .metric-box { padding: 12px; margin: 8px 0; }
        .stButton>button { padding: 8px 16px; font-size: 0.9rem; }
        .topics-box { padding: 12px; }
        .stPlotlyChart { width: 100% !important; }
    }
    </style>
""", unsafe_allow_html=True)

# Title
with st.container():
    st.title("📰 News Insights Dashboard")
    st.markdown("Explore NLP analysis results with sentiment distribution and topic modeling.")

# Load data
try:
    nlp_data = load_nlp_results()
    topics = load_lda_topics()
except FileNotFoundError as e:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.error(f"Error loading data: {str(e)}")
    st.write("Please check the file paths and ensure the data files exist in the `data` directory.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# Article Summaries
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📄 Article Summaries")
    available_columns = [col for col in ['source', 'summary'] if col in nlp_data.columns]
    if available_columns:
        display_data = nlp_data[available_columns].copy()
        if 'summary' in display_data.columns:
            display_data['summary'] = display_data['summary'].apply(
                lambda x: x[:200] + '...' if isinstance(x, str) and len(x) > 200 else x)
        st.dataframe(display_data, use_container_width=True, height=300)
    else:
        st.warning("Expected columns not found in nlp_results.csv")
    st.markdown('</div>', unsafe_allow_html=True)

# Sentiment Analysis
def get_sentiment(text):
    if isinstance(text, str):
        analysis = TextBlob(text)
        if analysis.sentiment.polarity > 0:
            return 'Positive'
        elif analysis.sentiment.polarity < 0:
            return 'Negative'
        else:
            return 'Neutral'
    return 'Unknown'

nlp_data['sentiment'] = nlp_data['summary'].apply(get_sentiment)

# Sentiment Chart
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("😊 Sentiment Distribution")
    sentiment_counts = nlp_data['sentiment'].value_counts()
    fig = px.bar(
        x=sentiment_counts.index,
        y=sentiment_counts.values,
        title="Sentiment Distribution",
        color=sentiment_counts.index,
        color_discrete_map={'Positive': '#22c55e', 'Negative': '#ef4444', 'Neutral': '#3b82f6', 'Unknown': '#6b7280'},
        labels={'x': 'Sentiment', 'y': 'Count'}
    )
    fig.update_layout(showlegend=False, title_x=0.5, font=dict(size=12), margin=dict(l=10, r=10, t=50, b=50))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# LDA Topics
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔍 Topics")
    st.markdown(f'<div class="topics-box">{topics}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
