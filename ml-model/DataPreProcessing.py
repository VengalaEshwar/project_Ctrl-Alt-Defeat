import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
def preprocess_CSV(csv) :
    dt=pd.read_csv(csv)
    mode=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
    dt_imputer=pd.DataFrame(mode.fit_transform(dt))
    dt_imputer.columns=dt.columns
    dt_imputer['dm']=dt_imputer['dm'].apply(lambda x:'yes' if x=='\tyes' else x)
    dt_imputer['dm']=dt_imputer['dm'].apply(lambda x:'no' if x=='\tno' else x)
    dt_imputer['cad']=dt_imputer['cad'].apply(lambda x:'no' if x=='\tno' else x)
    dt_imputer['classification']=dt_imputer['classification'].apply(lambda x:'ckd' if x=='\tckd' else x)
    dt_imputer['rc']=dt_imputer['rc'].apply(lambda x:'5.2'if x=='\t?' else x)
    dt_imputer['wc']=dt_imputer['wc'].apply(lambda x:'9800' if x=='\t?'or x=='\t6200' else x)
    dt_imputer['pcv']=dt_imputer['pcv'].apply(lambda x:'41' if x=='\t?' or x=='\t43' else x)
    mapping={'ckd\t':'ckd','ckd':'ckd','notckd':'notckd'}
    dt_imputer['classification']=dt_imputer['classification'].replace(mapping)
    dt_imputer['wc']=dt_imputer['wc'].apply(lambda x:'9800' if x=='\t?'or x=='\t6200'or x=='\t8400' else x)
    mapping1={'yes':'yes','no':'no',' yes':'yes'}
    dt_imputer['dm']=dt_imputer['dm'].replace(mapping1)
    encode=dt_imputer.apply(preprocessing.LabelEncoder().fit_transform)
    return encode.iloc[len(encode)-1]
def preprocess(data): 
    # Convert the input data to a DataFrame
    data = pd.DataFrame(data, index=[0])
    
    # Select columns with character data
    character_data = data.select_dtypes(include=['object'])

    # Apply LabelEncoder to each column with character data
    le = preprocessing.LabelEncoder()
    character_data_encoded = character_data.apply(lambda col: le.fit_transform(col))

    # Combine the encoded character data with the numerical data
    numerical_data = data.select_dtypes(exclude=['object'])
    encoded_data = pd.concat([numerical_data, character_data_encoded], axis=1)
    
    # Drop the 'id' column if present
    if 'id' in encoded_data.columns:
        encoded_data.drop('id', inplace=True, axis=1)
        print(encoded_data.columns)
    return encoded_data
