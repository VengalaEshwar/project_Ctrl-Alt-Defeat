import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import random
import DataPreProcessing as dp
df= pd.read_csv("D:\coding\Project_Ctrl-Alt-Defeat\ml-model\kidney_disease.csv")

df.drop('id', axis = 1, inplace = True)
df.columns = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 
              'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 
              'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'classification']

df['pcv'] = pd.to_numeric(df['pcv'], errors='coerce')
df['wc'] = pd.to_numeric(df['wc'], errors='coerce')
df['rc'] = pd.to_numeric(df['rc'], errors='coerce')

cat_cols = [col for col in df.columns if df[col].dtype == 'object']
num_cols = [col for col in df.columns if df[col].dtype != 'object']

df['dm'].replace(to_replace={'\tno': 'no', '\tyes': 'yes', ' yes': 'yes'}, inplace=True)
df['cad'] = df['cad'].replace(to_replace='\tno', value='no')
df['classification'] = df['classification'].replace(to_replace={'ckd\t': 'ckd', 'notckd': 'not ckd'})

cols = ['dm', 'cad', 'classification']
df.isna().sum().sort_values(ascending=False)
df[num_cols].isnull().sum()
df[cat_cols].isnull().sum()

def random_value_imputation(feature):
    random_sample = df[feature].dropna().sample(df[feature].isna().sum())
    random_sample.index = df[df[feature].isnull()].index
    df.loc[df[feature].isnull(), feature] = random_sample

def impute_mode(feature):
    mode = df[feature].mode()[0]
    df[feature] = df[feature].fillna(mode)

for col in num_cols:
    random_value_imputation(col)
df[num_cols].isnull().sum()

random_value_imputation('rbc')
random_value_imputation('pc')

for col in cat_cols:
    impute_mode(col)
df[cat_cols].isnull().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

ind_col = [col for col in df.columns if col != 'classification']
dep_col = 'classification'

X = df[ind_col]
y = df[dep_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# # KNN
# knn = KNeighborsClassifier()
# knn.fit(X_train, y_train)
# knn_acc = accuracy_score(y_test, knn.predict(X_test))

# # Decision Tree Classifier
# dtc = DecisionTreeClassifier()
# dtc.fit(X_train, y_train)
# dtc_acc = accuracy_score(y_test, dtc.predict(X_test))

# Random Forest Classifier
rfc = RandomForestClassifier(criterion='entropy', max_depth=11, max_features='sqrt', min_samples_leaf=2, min_samples_split=3, n_estimators=130)
rfc.fit(X_train, y_train)
# print(rfc.get_params)
rfc_acc = accuracy_score(y_test, rfc.predict(X_test))

# Sorted accuracy
models = pd.DataFrame({
    'Model': ['Random Forest Classifier'],
    'Score': [rfc_acc]
})
models_sorted = models.sort_values(by='Score', ascending=False)
# print(models_sorted)
# print(type(X_train))
# print(X_train.columns)
def predict(test) :   
# # Decision Tree Classifier
# dtc = DecisionTreeClassifier()
# dtc.fit(X_train, y_train)
# dtc_acc = accuracy_score(y_test, dtc.predict(X_test))
    return 1 if random.randrange(1,100) >50  else 0
