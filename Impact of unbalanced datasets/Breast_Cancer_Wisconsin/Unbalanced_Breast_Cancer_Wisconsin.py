import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pipeline import pipeline
from models import model_list

""" Unbalanced datasets and its impacts
    Tested Model: Logistic Regression, SGD Classifier, SLP, MLP
    Unbalanced Rate: 62.74%
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, stratify=y, random_state =1)

models, names = model_list()
for i, model in enumerate(models):
    pipeline(model, names[i], X_train, X_test, y_train, y_test, 'Unbalanced_BCW_')