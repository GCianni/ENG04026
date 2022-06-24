from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd

def pipeline(model, name:str, X_train, X_test, y_train, y_test, FIG_BASE_LABEL):
    pipe_lr = make_pipeline (StandardScaler(), PCA(n_components=2), model)
    pipe_lr.fit(X_train, y_train)
    y_pred = pipe_lr.predict(X_test) 
    print (name+' - Acurácia do Teste: %.3f' %pipe_lr.score(X_test, y_test))

    cm = confusion_matrix(y_test, y_pred) 
    cm_df = pd.DataFrame(cm,index = ['B', 'M'], columns = ['B', 'M']) 
    plt.figure(figsize=(8,6)) 
    sns.heatmap(cm_df, annot=True,fmt='g',cmap='Greys_r') 
    plt.title(name+'\nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred))) 
    plt.ylabel('Valores "Verdadeiros"') 
    plt.xlabel('Valores Preditos')
    plt.savefig(FIG_BASE_LABEL+name+'.png')
    plt.show()
    plt.close()