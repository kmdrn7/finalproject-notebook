#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import pandas as pd
import numpy as np
from joblib import dump, load
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, balanced_accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


# In[2]:


cols = ['Label', 'SYN Flag Count', 'Fwd Seg Size Min', 'FWD Init Win Bytes',
       'FIN Flag Count', 'Average Packet Size', 'Packet Length Mean',
       'Packet Length Max', 'Protocol', 'Idle Max', 'Idle Mean', 'Idle Min',
       'Flow Duration', 'Fwd IAT Total', 'Fwd Packet Length Max',
       'Fwd Segment Size Avg', 'Fwd Packet Length Mean',
       'Bwd Packet Length Mean', 'Bwd Segment Size Avg', 'Packet Length Std',
       'Bwd IAT Total', 'Bwd Packet Length Max']

collection = [
    {"clf": AdaBoostClassifier(), "model": "AdaBoost"},
    {"clf": DecisionTreeClassifier(criterion="entropy"), "model": "DecisionTree"},
    {"clf": LinearSVC(), "model": "SVM"},
    {"clf": MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000), "model": "ANN"},
    {"clf": LogisticRegression(solver='newton-cg'), "model": "LogisticRegression"},
    {"clf": RandomForestClassifier(), "model": "RandomForest"},
    {"clf": XGBClassifier(use_label_encoder=False), "model": "XGBoost"},
    {"clf": BaggingClassifier(), "model": "Bagging"},
]


# In[3]:


dataset = pd.read_csv("/media/kmdr7/Seagate/TA/DATASETS/newDatasetSampledEncoded.csv")[cols]

X = dataset.drop(columns=["Label"])
y = dataset["Label"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=1 / 7.0, random_state=1
)


# In[4]:


for col in collection:
    clf = col["clf"]
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    acc = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    try: roc_auc = roc_auc_score(y_test, y_pred)
    except: roc_auc = 0
    
    dump(clf, "/media/kmdr7/Seagate/TA/MODELS/ModelWithoutScaler/" + col["model"] + ".joblib")
    
    requests.post(
        "http://localhost:8000/api/v1/training",
        json={
            "code": "002-WithoutScaler",
            "algorithm": col["model"],
            "dataset": "IoT-23 Mirai 48-1 + IoT Traffic Traces",
            "matrix": {
                "tn": float(tn),
                "fp": float(fp),
                "fn": float(fn),
                "tp": float(tp)
            },
            "accuracy": float(acc),
            "balanced_accuracy": float(bacc),
            "recall": float(recall),
            "f1": float(f1),
            "precision": float(precision),
            "roc_auc": float(roc_auc),
        }
    )

