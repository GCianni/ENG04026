import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pipeline import pipeline
from models import model_list

""" Highly Unbalanced datasets and its impacts on Machine Learning Models
    Tested Model: Logistic Regression, SGD Classifier, SLP, MLP
    Unbalanced Rate: [98.61%, 94.69%, 89.92%, 78.11%, 62.74%]
    Dataset: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
"""

DATA_PATH = r'C:/Users/55519/Desktop/UFRGS/Matérias/ENG04026 - Tópicos Especiais em Instrumentação II/Materiais/Datasets/'
#header = list(pd.read_csv(DATA_PATH +'wdbc_header.txt', sep=","))
df =  pd.read_csv(DATA_PATH +'wdbc.data', sep=",", header=None)

X = df.loc[:, 2:].values 
y =df.loc[:, 1].values 

le = LabelEncoder() 
y = le.fit_transform(y) 
print("Label Encoder Classes:",le.classes_)
for unsamp in [5, 20, 40, 100, 212]:
    print("---------------------------------",unsamp)
    uns_rate = len(X[y==0])/(len(X[y==0])+unsamp)
    print("Unsample rate:", uns_rate)
    X_adaptado = np.vstack((X[y==0], X[y==1][:unsamp])) 
    y_adaptado = np.hstack((y[y==0], y[y==1][:unsamp]))

    y_pred = np.zeros(y_adaptado.shape[0]) # Dummy Classifier
    print("Dummy Classifier:",np.mean(y_pred == y_adaptado) * 100)

    X_train, X_test, y_train, y_test = train_test_split(X_adaptado, y_adaptado, test_size = 0.20, stratify=y_adaptado, random_state =1)

    models, names = model_list()
    for i, model in enumerate(models):
        pipeline(model, names[i], X_train, X_test, y_train, y_test, str(unsamp)+'-Highly_Unbalanced_BWC_')