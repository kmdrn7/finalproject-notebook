#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import requests
from glob import glob
from joblib import dump, load
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import DMatrix
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, balanced_accuracy_score, roc_auc_score, confusion_matrix


# In[2]:


def get_csv(uri):
    csvs = []
    if "*" in uri:
        all_csv = glob(uri)
        [ csvs.append(pd.read_csv(uri)) for uri in all_csv ]
        return pd.concat(csvs)
    return pd.read_csv(uri)


# In[3]:


cols = ['Label', 'Protocol', 'Flow Duration', 'Fwd Packet Length Max',
    'Fwd Packet Length Mean', 'Bwd Packet Length Mean', 'Fwd IAT Total',
    'Bwd IAT Total', 'Packet Length Max', 'Packet Length Mean',
    'Packet Length Std', 'FIN Flag Count', 'SYN Flag Count',
    'Down/Up Ratio', 'Average Packet Size', 'Fwd Segment Size Avg',
    'Bwd Segment Size Avg', 'FWD Init Win Bytes', 'Fwd Seg Size Min',
    'Idle Mean', 'Idle Max', 'Idle Min']

models = [
    # {"model": "ModelStandardScaler/AdaBoost"},
    # {"model": "ModelStandardScaler/LogisticRegression"},
    # {"model": "ModelStandardScaler/RandomForest"},
    # {"model": "ModelStandardScaler/DecisionTree"},
    # {"model": "ModelStandardScaler/SVM"},
    # {"model": "ModelStandardScaler/ANN"},
    # {"model": "ModelStandardScaler/XGBoost"},
    {"model": "ModelStandardScaler/Bagging"},
]

collection = [
    {"dataset": "Unseen Dataset", "path": "/media/kmdr7/Seagate/TA/DATASETS/newUnseenDataset.csv", "type": -1},
#     {"dataset": "Malware 48-1 *", "path": "/media/kmdr7/Seagate/DATASETS/IOT-23/CTU-IoT-Malware-Capture-48-1/out2/*", "type": 1},
#     {"dataset": "Benign IoTTT *", "path": "/media/kmdr7/Seagate/DATASETS/IoT-Traffic-Traces/out/*", "type": 0},
]

scaler = StandardScaler()


# In[4]:


for model in models:
    print("#======================================")
    print("# Predicting model " + model["model"])
    print("#======================================")

    clf = load("/media/kmdr7/Seagate/TA/MODELS/" + model["model"] + ".joblib")

    for col in collection:

        y_real = []
        y_pred = []
        datates = get_csv(col["path"])[cols]
        if col["type"] == 1:
            datates["Label"] = 1
        elif col["type"] == 0:
            datates["Label"] = 0

        iter = 0
        for i in datates.index:
            if (i % 1000 == 0): print("Iter: " + str(iter))
            iter+=1

            single = datates.iloc[i]
            X = [single.drop("Label")]
            y = [single["Label"]]
            X = scaler.fit_transform(X)

#             X = X.to_numpy()[0][:, np.newaxis]
#             X = pd.DataFrame(scaler.fit_transform(X))
#             X = np.transpose(X.values)

            if model["model"].__contains__("XGBoost"):
                X = pd.DataFrame(X)

            pred = clf.predict(X)
            y_real.append(int(y[0]))
            y_pred.append(int(pred[0]))

        tn, fp, fn, tp = confusion_matrix(y_real, y_pred, labels=[0,1]).ravel()
        acc = accuracy_score(y_real, y_pred)
        bacc = balanced_accuracy_score(y_real, y_pred)
        recall = recall_score(y_real, y_pred, zero_division=0)
        f1 = f1_score(y_real, y_pred, zero_division=0)
        precision = precision_score(y_real, y_pred, zero_division=0)
        try: roc_auc = roc_auc_score(y_real, y_pred)
        except: roc_auc = 0

        requests.post(
            "http://localhost:8000/api/v1/realtime",
            json={
                "algorithm": model["model"],
                "dataset": "UnseenDataset",
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

