import pandas as pd
import os
import pickle

# Get the base directory of the Rainfall_app (parent of utils directory)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

def load_feature_data():
    file_path = os.path.join(DATA_DIR, 'feature_engineered_data.csv')
    _check_file_exists(file_path)
    return pd.read_csv(file_path)

def load_reg_model():
    file_path = os.path.join(DATA_DIR, 'best_random_forest_regressor_model.pkl')
    _check_file_exists(file_path)
    return pd.read_pickle(file_path)

def load_clf_model():
    file_path = os.path.join(DATA_DIR, 'best_random_forest_classifier_model.pkl')
    _check_file_exists(file_path)
    return pd.read_pickle(file_path)

def load_nlp_results():
    file_path = os.path.join(DATA_DIR, 'nlp_results.csv')
    _check_file_exists(file_path)
    return pd.read_csv(file_path)

def load_lda_topics():
    file_path = os.path.join(DATA_DIR, 'lda_topics.txt')
    _check_file_exists(file_path)
    with open(file_path, 'r') as f:
        return f.read()

def load_regional_performance_regression():
    file_path = os.path.join(DATA_DIR, 'regional_performance_regression.csv')
    _check_file_exists(file_path)
    return pd.read_csv(file_path, index_col='station_id')

def load_regional_performance_classification():
    file_path = os.path.join(DATA_DIR, 'regional_performance_classification.csv')
    _check_file_exists(file_path)
    return pd.read_csv(file_path, index_col='station_id')

def load_model_evaluation_results():
    file_path = os.path.join(DATA_DIR, 'model_evaluation_results.csv')
    _check_file_exists(file_path)
    return pd.read_csv(file_path)

def _check_file_exists(file_path):
    """Helper function to check if a file exists and raise a descriptive error if not."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at: {file_path}")