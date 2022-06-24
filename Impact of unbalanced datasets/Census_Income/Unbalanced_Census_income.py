import pandas as pd 
import numpy as np

from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import metrics 

from models import model_list
from data_cleaning import clean_data
import matplotlib.pyplot as plt 
import seaborn as sns


DATA_PATH = r'C:/Users/55519/Desktop/UFRGS/Matérias/ENG04026 - Tópicos Especiais em Instrumentação II/Materiais/Datasets/'
DATA_PATH +'wdbc.data'
FIG_BASE_LABEL='Unbalanced_Census_Income_'
dados = pd.read_csv(DATA_PATH +'adultoUCI.csv') 

print('--------------------')
print('DATA HEAD - Before Data Cleaning')
print(dados.head())

dados = clean_data(dados)

print('--------------------')
print('Value Counts')
print(dados['income'].value_counts())


print('--------------------')
print('DATA HEAD - After Data Cleaning')
print(dados.head() )

X = dados.drop(['income'],axis=1) 
y = dados['income'] 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=123)

models, names = model_list('not_bal')
for i, model in enumerate(models):
    print('--------------------')
    print(names[i])
    model.fit(X_train,y_train) 
    y_pred=model.predict(X_test) 
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred) 
    cm_df = pd.DataFrame(cm,index = ['<=50K', '>50K'], columns = ['<=50K', '>50K']) 
    plt.figure(figsize=(8,6)) 
    sns.heatmap(cm_df, annot=True,fmt='g',cmap='Greys_r') 
    plt.title(names[i]+'\nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred))) 
    plt.ylabel('Valores "Verdadeiros"') 
    plt.xlabel('Valores Preditos')
    plt.savefig(FIG_BASE_LABEL+names[i]+'.png')
    plt.show()
    plt.close()