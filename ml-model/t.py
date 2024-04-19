""" import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

data = pd.read_csv("kidney_disease.csv")
data.replace('\t?', np.nan, inplace=True)

X = data.drop("classification", axis=1)
y = data["classification"]

categorical_cols = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]
numerical_cols = ["age", "bp", "sg", "al", "su", "bgr", "bu", "hemo", "pcv", "wc", "rc"]

numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)


model = LinearRegression()
model.fit(X_train_processed, y_train_encoded)

train_score = model.score(X_train_processed, y_train_encoded)
test_score = model.score(X_test_processed, label_encoder.transform(y_test))

print(f"Train Score: {train_score}")
print(f"Test Score: {test_score}")

 """
""" import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("kidney_disease.csv")
data.replace('\t?', np.nan, inplace=True)

# Separate features (X) and target variable (y)
X = data.drop("classification", axis=1)
y = data["classification"]

# Define categorical and numerical columns
categorical_cols = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]
numerical_cols = ["age", "bp", "sg", "al", "su", "bgr", "bu", "hemo", "pcv", "wc", "rc"]

# Preprocessing for numerical data: impute missing values and scale
numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

# Preprocessing for categorical data: impute missing values and one-hot encode
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Preprocess the data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Encode the target variable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Define the models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Classifier": RandomForestClassifier(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier()
}

# Train and evaluate each model
for name, model in models.items():
    # Train the model
    model.fit(X_train_processed, y_train_encoded)
    
    # Calculate scores
    train_score = model.score(X_train_processed, y_train_encoded)
    test_score = model.score(X_test_processed, label_encoder.transform(y_test))
    
    print(f"{name}:")
    print(f"Train Score: {train_score}")
    print(f"Test Score: {test_score}")
    print()
 """
 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Replace 'path/to/your/file.csv' with the actual path to your data
data = pd.read_csv("kidney_disease.csv")
data.replace('\t?', np.nan, inplace=True)

X = data.drop("classification", axis=1)
y = data["classification"]

categorical_cols = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]
numerical_cols = ["age", "bp", "sg", "al", "su", "bgr", "bu", "hemo", "pcv", "wc", "rc"]

numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Encode target variable for metrics calculation
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

models = [
    ("Linear Regression", LinearRegression()),
    ("Random Forest Classifier", RandomForestClassifier()),
    ("KNeighbors Classifier", KNeighborsClassifier()),
    ("Support Vector Classifier", SVC())
]

best_accuracy = 0
best_model_name = ""

for model_name, model in models:
    # Use preprocessed data (one-hot encoded for categorical features)
    model.fit(X_train_processed, y_train_encoded)  # Use encoded labels for fitting (except Linear Regression)
    train_score = model.score(X_train_processed, y_train_encoded)
    test_score = model.score(X_test_processed, y_test_encoded)
    print(f"{model_name}: Train Score: {train_score}, Test Score: {test_score}")

    # Use original labels for prediction (for all models)
    y_test_pred = model.predict(X_test_processed)

    # Use accuracy_score for classification models
    if model_name != "Linear Regression":
        accuracy = accuracy_score(y_test, y_test_pred)
        print(f"Accuracy: {accuracy}")

print(f"Best Model (based on classification models): {best_model_name}, Best Accuracy: {best_accuracy}")


