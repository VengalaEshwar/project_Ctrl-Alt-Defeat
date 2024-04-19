import numpy as np
import pandas as pd
import sklearn
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

data=pd.read_csv("kidney_disease.csv")

X = data.drop('classification', axis=1)  
y = data['classification']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
""" X_train_np = X_train.to_numpy()
y_train_np = y_train.to_numpy()

model = RandomForestClassifier()
model.fit(X_train_np, y_train_np) """
from sklearn.preprocessing import LabelEncoder
model = RandomForestClassifier()
# Encode y_train
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)

# Now, y_train should be ready for training
model.fit(X_train, y_train)

"""
# Make predictions
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
 """
""" models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'Neural Network': MLPClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Extra Trees': ExtraTreesClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Bagging': BaggingClassifier(),
    'SGD': SGDClassifier()
}
 """
# Train and evaluate each model
""" accuracies = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies[name] = accuracy
    print(f'{name} - Accuracy: {accuracy}') """
"""
# Compare the accuracies
best_model = max(accuracies, key=accuracies.get)
print(f'\nBest Model: {best_model} - Accuracy: {accuracies[best_model]}')
 """