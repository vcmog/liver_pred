import pandas as pd
import numpy as np
import utils.preprocessing.feature_generation as fg
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.linear_model import LogisticRegression
import utils.preprocessing.missing_data as md
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, f1_score, classification_report, precision_score, recall_score, roc_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from utils.Models import models, pytorch_models
from utils.Evaluation import evaluation
import torch
import pickle
from pathlib import Path

dir = r"C:\Users\victo\OneDrive - University of Leeds\Documents\Uni Work\Project\MIMIC Work\Liver Cancer Prediction\liver_pred\data\interim"
model_dir = r"C:\Users\victo\OneDrive - University of Leeds\Documents\Uni Work\Project\MIMIC Work\Liver Cancer Prediction\liver_pred\data\models"

# Experiment Settings
lead_time = 0
max_history = 365*2
experiment_dir = r"C:\Users\victo\OneDrive - University of Leeds\Documents\Uni Work\Project\MIMIC Work\Liver Cancer Prediction\liver_pred\outputs\leadtime=0"
output_dir =  r"C:\Users\victo\OneDrive - University of Leeds\Documents\Uni Work\Project\MIMIC Work\Liver Cancer Prediction\liver_pred\outputs\lead_time_experiment_results"

cohort_ids = pd.read_csv(dir + r"\matched_cohort_ids.csv", index_col=0)
processed_labs = pd.read_csv(dir + r"\processed_lab_data.csv", parse_dates=["charttime", "index_date"], index_col=0)

train_ids = cohort_ids.sample(frac=0.8, random_state = 32) 
test_ids = cohort_ids.drop(train_ids.index)

lab_df = processed_labs.loc[processed_labs['subject_id'].isin(train_ids['subject_id'])].copy()
final_test_set = processed_labs[processed_labs['subject_id'].isin(test_ids['subject_id'])].copy()

current_df, trend_features, feature_df = fg.generate_features(lab_df, cohort_ids, lead_time=lead_time,  current_window_preindex=7, current_window_postindex=1, historical_window=0)
test_current, test_trend, feature_df_test = fg.generate_features(final_test_set, cohort_ids, lead_time=lead_time, current_window_preindex=7, current_window_postindex=1, historical_window=0)



def run_FEng_model(feature_dfs, feature_df_tests, mode="current+trend"):
    """
    Runs the feature engineering model for liver cancer prediction.

    Args:
        feature_dfs (list): A list of feature dataframes for training.
        feature_df_tests (list): A list of feature dataframes for testing.
        lead_time (int, optional): The lead time for prediction. Defaults to 0.
        mode (str, optional): The mode for feature engineering. Must be one of 'current+trend', 'trend', 'current'. Defaults to "current+trend".

    Raises:
        ValueError: If an invalid mode is provided.

    Returns:
        None
    """
    
    if mode == "current+trend":
        feature_df = feature_dfs[2]
        feature_df_test = feature_dfs[2]
    elif mode == 'trend':
        feature_df = feature_df[1].merge(feature_df[2][['subject_id', 'outcome']], on='subject_id')
        feature_df_test = feature_df_tests[1].merge(feature_df_tests[2][['subject_id', 'outcome']], on='subject_id')
    elif mode == 'current':
        feature_df = feature_df[0]
        feature_df_test = feature_df_tests[0]
    else:
        raise ValueError("Invalid mode. Must be one of 'current+trend', 'trend', 'current'")
    
    y_train = feature_df['outcome']
    X_train = feature_df.drop(columns=['outcome','subject_id'])
    y_test = feature_df_test['outcome']
    X_test = feature_df_test.drop(columns=['outcome', 'subject_id'])

    #X_train, X_test,y_train, y_test = train_test_split(X, y, train_size = 0.8)
    #print(f"Train Length: {len(X_train)}        Train cases: {len(y_train[y_train==1])}    Proportion: {len(y_train[y_train==1])/len(y_train)*100} %")
    #print(f"Test Length: {len(X_test)}          Test cases: {len(y_test[y_test==1])}       Proportion: {len(y_test[y_test==1])/len(y_test)*100} %")

    #### Scale training #####
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train))
    X_train_scaled = X_train_scaled.set_axis(X_train.columns,axis=1).fillna(0)

    #### Scale Test using training scaler ####
    X_test_scaled = pd.DataFrame(scaler.transform(X_test))
    X_test_scaled = X_test_scaled.set_axis(X_test.columns,axis=1).fillna(0)

    print("Loading model...")
    current_trend_model = joblib.load(experiment_dir+r"\{}_nnmodel.pkl".format(mode))
    print("Fitting model...")
    current_trend_model.fit(X_train_scaled, y_train)
    print("Model Fitted.")
    print("Evaluating model...")
    train_preds = current_trend_model.predict_proba(X_train_scaled)
    train_results = evaluation.evaluate_performance_nontorch(train_preds, y_train)
    train_results['index'] = lead_time
    # append a csv with train_results as a row
    train_results_df = pd.DataFrame.from_dict(train_results)
    train_results_df.to_csv(output_dir + r"\{}_train_results.csv".format(mode), mode='a', header=False)

    test_preds = current_trend_model.predict_proba(X_test_scaled)
    test_results = evaluation.evaluate_performance_nontorch(test_preds, y_test)
    test_results['index'] = lead_time
    # append a csv with train_results as a row
    test_results_df = pd.DataFrame.from_dict(test_results)
    test_results_df.to_csv(output_dir + r"\{}_test_results.csv".format(mode), mode='a', header=False)
    print(""{}" Model Complete.".format(mode))



run_FEng_model([current_df, trend_features, feature_df], [test_current, test_trend, feature_df_test], mode="current+trend")
run_FEng_model([current_df, trend_features, feature_df], [test_current, test_trend, feature_df_test], mode="trend")
run_FEng_model([current_df, trend_features, feature_df], [test_current, test_trend, feature_df_test], mode="current")


