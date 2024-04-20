""" import pandas as pd
import numpy as np
df= pd.read_csv('kidney_disease.csv')
df.drop('id', axis = 1, inplace = True)
df['packed_cell_volume'] = pd.to_numeric(df['packed_cell_volume'], errors='coerce')
df['white_blood_cell_count'] = pd.to_numeric(df['white_blood_cell_count'], errors='coerce')
df['red_blood_cell_count'] = pd.to_numeric(df['red_blood_cell_count'], errors='coerce')
cat_cols = [col for col in df.columns if df[col].dtype == 'object']
num_cols = [col for col in df.columns if df[col].dtype != 'object']
for col in cat_cols:
    print(f"{col} has {df[col].unique()} values\n")
df['diabetes_mellitus'].replace(to_replace = {'\tno':'no','\tyes':'yes',' yes':'yes'},inplace=True)
df['coronary_artery_disease'] = df['coronary_artery_disease'].replace(to_replace = '\tno', value='no')
df['class'] = df['class'].replace(to_replace = {'ckd\t': 'ckd', 'notckd': 'not ckd'})
df['class'] = df['class'].map({'ckd': 0, 'not ckd': 1})
df['class'] = pd.to_numeric(df['class'], errors='coerce')
cols = ['diabetes_mellitus', 'coronary_artery_disease', 'class']
df.isna().sum().sort_values(ascending = False)
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
random_value_imputation('red_blood_cells')
random_value_imputation('pus_cell')

for col in cat_cols:
    impute_mode(col)
df[cat_cols].isnull().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])
ind_col = [col for col in df.columns if col != 'class']
dep_col = 'class'

X = df[ind_col]
y = df[dep_col]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

#knn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_acc = accuracy_score(y_test, knn.predict(X_test))

print(f"Training Accuracy of KNN is {accuracy_score(y_train, knn.predict(X_train))}")
print(f"Test Accuracy of KNN is {knn_acc} \n")

#dtc
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
dtc_acc = accuracy_score(y_test, dtc.predict(X_test))

print(f"Training Accuracy of DTC is {accuracy_score(y_train, dtc.predict(X_train))}")
print(f"Test Accuracy of DTC is {dtc_acc} \n")

#rfc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rfc = RandomForestClassifier(criterion='entropy', max_depth=11, max_features='sqrt', min_samples_leaf=2, min_samples_split=3, n_estimators=130)
rfc.fit(X_train, y_train)
rfc_acc = accuracy_score(y_test, rfc.predict(X_test))

print(f"Training Accuracy of Random Forest Classifier is {accuracy_score(y_train, rfc.predict(X_train))}")
print(f"Test Accuracy of Random Forest Classifier is {rfc_acc} \n")

models = pd.DataFrame({
    'Model': ['KNN', 'Decision Tree Classifier', 'Random Forest Classifier'],
    'Score': [knn_acc, dtc_acc, rfc_acc]
})

models_sorted = models.sort_values(by='Score', ascending=False)
print(models_sorted)

 """
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Read the dataset
df = pd.read_csv('kidney_disease.csv')

# Drop 'id' column
df.drop('id', axis=1, inplace=True)

# Convert certain columns to numeric
numeric_cols = ['packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Handle categorical columns
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    if col in ['class', 'coronary_artery_disease']:
        df[col] = df[col].replace({'\tno': 'no', 'ckd\t': 'ckd', 'notckd': 'not ckd'})
        df[col] = df[col].map({'ckd': 0, 'not ckd': 1})
    elif col == 'diabetes_mellitus':
        df[col] = df[col].replace({'\tno': 'no', '\tyes': 'yes', ' yes': 'yes'})

# Impute missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].mean(), inplace=True)

# Label encode categorical columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Split data into features and target variable
X = df.drop('class', axis=1)
y = df['class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Model training and evaluation

# KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_acc = accuracy_score(y_test, knn.predict(X_test))

# Decision Tree Classifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
dtc_acc = accuracy_score(y_test, dtc.predict(X_test))

# Random Forest Classifier
rfc = RandomForestClassifier(criterion='entropy', max_depth=11, max_features='sqrt', min_samples_leaf=2, min_samples_split=3, n_estimators=130)
rfc.fit(X_train, y_train)
rfc_acc = accuracy_score(y_test, rfc.predict(X_test))

# Create a DataFrame to store model scores
models = pd.DataFrame({
    'Model': ['KNN', 'Decision Tree Classifier', 'Random Forest Classifier'],
    'Score': [knn_acc, dtc_acc, rfc_acc]
})

# Sort models by score
models_sorted = models.sort_values(by='Score', ascending=False)

# Print the sorted DataFrame
print(models_sorted.to_string(index=False))
