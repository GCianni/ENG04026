import numpy as np
from sklearn.preprocessing import LabelEncoder

def clean_data(dados):

    dados.replace('?',np.nan,inplace=True) 
    dados.dropna(inplace=True)

    Labelenc_workclass = LabelEncoder() 
    dados['workclass'] = Labelenc_workclass.fit_transform(dados['workclass'])

    Labelenc_education = LabelEncoder() 
    dados['education'] = Labelenc_education.fit_transform(dados['education'])

    Labelenc_marital_status = LabelEncoder() 
    dados['marital-status'] = Labelenc_marital_status.fit_transform(dados['marital-status']) 

    Labelenc_occupation = LabelEncoder() 
    dados['occupation'] = Labelenc_occupation.fit_transform(dados['occupation'])

    Labelenc_relationship = LabelEncoder() 
    dados['relationship'] = Labelenc_relationship.fit_transform(dados['relationship'])

    Labelenc_race = LabelEncoder() 
    dados['race'] = Labelenc_race.fit_transform(dados['race'])
        
    Labelenc_gender = LabelEncoder() 
    dados['gender'] = Labelenc_gender.fit_transform(dados['gender'])

    Labelenc_native_country = LabelEncoder() 
    dados['native-country'] = Labelenc_native_country.fit_transform(dados['native-country'])

    Labelenc_income = LabelEncoder() 
    dados['income'] = Labelenc_income.fit_transform(dados['income'])
    return dados