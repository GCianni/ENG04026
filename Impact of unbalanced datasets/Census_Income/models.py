from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier 
def model_list(class_weights):
        if class_weights!= 'not_bal':
                clfs = [RandomForestClassifier(random_state=0, class_weight='balanced'),
                        LogisticRegression(random_state=0, solver='lbfgs',class_weight='balanced'),
                        SGDClassifier(max_iter=1000, tol=1e-3,random_state=0, class_weight='balanced'),
                        Perceptron(tol=1e-3, random_state=0, class_weight='balanced')]
                names = ['Random Forest','Regresão Logística', 'Classificador SGD', 'SLP']
        else:
                clfs = [RandomForestClassifier(random_state=0),
                        LogisticRegression(random_state=0, solver='lbfgs'),
                        SGDClassifier(max_iter=1000, tol=1e-3,random_state=0),
                        Perceptron(tol=1e-3, random_state=0)]
                names = ['Random Forest','Regresão Logística', 'Classificador SGD', 'SLP']
        return clfs, names