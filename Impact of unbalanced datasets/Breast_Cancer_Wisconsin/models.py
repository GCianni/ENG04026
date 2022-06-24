from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron
from sklearn.neural_network import MLPClassifier
def model_list():
    clfs = [LogisticRegression(random_state=0, solver='lbfgs'),
            SGDClassifier(max_iter=1000, tol=1e-3,random_state=0),
            Perceptron(tol=1e-3, random_state=0),
            MLPClassifier(random_state=1, max_iter=450)]
    names = ['Regresão Logística', 'Classificador SGD', 'SLP', 'MLP']
    return clfs, names