
import pandas as pd

def load_feature_data():
      return pd.read_csv('../Rainfall_app/data/feature_engineered_data.csv')

def load_reg_model():
      return pd.read_pickle('../Rainfall_app/data/best_random_forest_regressor_model.pkl')

def load_clf_model():
      return pd.read_pickle('../Rainfall_app/data/best_random_forest_classifier_model.pkl')

def load_nlp_results():
      return pd.read_csv('../Rainfall_app/data/nlp_results.csv')

def load_lda_topics():
      with open('../Rainfall_app/data/lda_topics.txt', 'r') as f:
          return f.read()

def load_regional_performance_regression():
      return pd.read_csv('../Rainfall_app/data/regional_performance_regression.csv', index_col='station_id')

def load_regional_performance_classification():
      return pd.read_csv('../Rainfall_app/data/regional_performance_classification.csv', index_col='station_id')

def load_model_evaluation_results():
    return pd.read_csv('../Rainfall_app/data/model_evaluation_results.csv')
