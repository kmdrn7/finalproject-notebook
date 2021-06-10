#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests
import time
import pandas as pd
import numpy as np
from statistics import mean
from joblib import dump, load
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, balanced_accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


# In[ ]:


datasets = [
    {"dataset": "10", "path": "/media/kmdr7/Seagate/TA/DATASETS/Preparation/Feature Importance/Dataset10.csv"},
    {"dataset": "13", "path": "/media/kmdr7/Seagate/TA/DATASETS/Preparation/Feature Importance/Dataset13.csv"},
    {"dataset": "15", "path": "/media/kmdr7/Seagate/TA/DATASETS/Preparation/Feature Importance/Dataset15.csv"},
    {"dataset": "17", "path": "/media/kmdr7/Seagate/TA/DATASETS/Preparation/Feature Importance/Dataset17.csv"},
    {"dataset": "20", "path": "/media/kmdr7/Seagate/TA/DATASETS/Preparation/Feature Importance/Dataset20.csv"},
]

mls = [
    {"clf": AdaBoostClassifier(), "model": "AdaBoost"},
    {"clf": DecisionTreeClassifier(criterion="entropy"), "model": "DecisionTree"},
    {"clf": LinearSVC(), "model": "SVM"},
    {"clf": MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000), "model": "ANN"},
    {"clf": LogisticRegression(solver='newton-cg'), "model": "LogisticRegression"},
    {"clf": RandomForestClassifier(), "model": "RandomForest"},
    {"clf": XGBClassifier(use_label_encoder=False), "model": "XGBoost"},
    {"clf": BaggingClassifier(), "model": "Bagging"},
]

scoring = ['accuracy', 'balanced_accuracy', 'recall_macro', 'f1_macro', 'precision_macro', 'roc_auc']

scaler = MinMaxScaler()


# In[ ]:


for dts in datasets:

    dataset = pd.read_csv(dts["path"])
    X = dataset.drop(columns=["Label"])
    y = dataset["Label"]
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=1 / 7.0, random_state=1
    )
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)

    for ml in mls:
        clf = ml["clf"]
        start = time.time()
        result = cross_validate(clf, X, y, scoring=scoring, cv=10, n_jobs=4)
        end = time.time()
        fit_time = mean(result["fit_time"])
        score_time = mean(result["score_time"])
        accuracy = mean(result["test_accuracy"])
        balanced_accuracy = mean(result["test_balanced_accuracy"])
        recall = mean(result["test_recall_macro"])
        f1 = mean(result["test_f1_macro"])
        precision = mean(result["test_precision_macro"])
        roc_auc = mean(result["test_roc_auc"])
        requests.post(
            "http://localhost:8000/api/v1/model_performance",
            json={
                "code": "003-MinMaxScaler",
                "algorithm": ml["model"],
                "dataset": "IoT-23 Mirai 48-1 + IoT Traffic Traces " + dts["dataset"],
                "accuracy": float(accuracy),
                "balanced_accuracy": float(balanced_accuracy),
                "recall": float(recall),
                "f1": float(f1),
                "precision": float(precision),
                "roc_auc": float(roc_auc),
                "time": float(end-start)
            }
        )

    del dataset    
    del X
    del y


# In[ ]:




