import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pipeline import pipeline
from models import model_list

""" Upsampling datasets and its impacts on Machine Learning Models
    Tested Model: Logistic Regression, SGD Classifier, SLP, MLP
    Upsampling data lenght: 357
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

X_reamostrado, y_reamostrado = resample(X[y == 1], 
                                            y[y == 1], replace=True, 
                                            n_samples=X[y == 0].shape[0], 
                                            random_state = 123)

print('Número de Dados da Classe 1 (após a reamostragem)', X_reamostrado.shape[0]) 
x_bal = np.vstack((X[y == 0], X_reamostrado)) 
y_bal = np.hstack((y[y == 0], y_reamostrado)) 
y_pred = np.zeros(y_bal.shape[0])

print("Dummy Classifier - Balanced", np.mean(y_pred == y_bal) * 100)

X_train, X_test, y_train, y_test = train_test_split(x_bal, y_bal, test_size = 0.20, stratify=y_bal, random_state =1)

models, names = model_list()
for i, model in enumerate(models):
    pipeline(model, names[i], X_train, X_test, y_train, y_test, 'Upsampling_BCW_')