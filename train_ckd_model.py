import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# Load dataset
dataset_path = "kidney_disease.csv"
df = pd.read_csv(dataset_path)

# Clean column names
df.columns = df.columns.str.strip()

# Drop id column
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)

# Function to clean text values
def clean_text(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip().lower()
    value = value.replace("\t", "")
    value = value.replace("\n", "")
    value = value.replace('"', "")
    value = value.replace("'", "")
    value = value.replace(" ", "")
    if value in ["?", "", "nan", "none"]:
        return np.nan
    return value

# Clean object columns
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].apply(clean_text)

# Fix target column
df["classification"] = df["classification"].replace({
    "ckd\t": "ckd",
    "notckd": "notckd",
    "not ckd": "notckd"
})

df["classification"] = df["classification"].apply(
    lambda x: "ckd" if x == "ckd" else ("notckd" if x == "notckd" else np.nan)
)

df = df.dropna(subset=["classification"])

# Convert numeric columns
numeric_columns = [
    'age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc',
    'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc'
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Clean categorical columns
valid_values = {
    "yes": "yes",
    "no": "no",
    "present": "present",
    "notpresent": "notpresent",
    "normal": "normal",
    "abnormal": "abnormal",
    "good": "good",
    "poor": "poor"
}

categorical_columns = [
    'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
]

for col in categorical_columns:
    if col in df.columns:
        df[col] = df[col].map(lambda x: valid_values[x] if x in valid_values else np.nan)

# Features and target
X = df.drop("classification", axis=1)
y = df["classification"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Identify numeric and categorical features
num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

# Preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, num_features),
    ("cat", categorical_transformer, cat_features)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# Models
log_reg = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=2000, random_state=42))
])

rf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
])

svc = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", SVC(probability=True, kernel="rbf", random_state=42))
])

gb = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42))
])

# Train individual models
log_reg.fit(X_train, y_train)
rf.fit(X_train, y_train)
svc.fit(X_train, y_train)
gb.fit(X_train, y_train)

# Voting classifier using transformed data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

voting_clf = VotingClassifier(
    estimators=[
        ("lr", LogisticRegression(max_iter=2000, random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
        ("svc", SVC(probability=True, kernel="rbf", random_state=42)),
        ("gb", GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42))
    ],
    voting="soft"
)

voting_clf.fit(X_train_processed, y_train)

# Accuracy results
models = {
    "Logistic Regression": log_reg,
    "Random Forest": rf,
    "Support Vector Classifier": svc,
    "Gradient Boosting": gb
}

accuracy_results = {}

for name, model in models.items():
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    accuracy_results[name] = accuracy

voting_predictions = voting_clf.predict(X_test_processed)
voting_accuracy = accuracy_score(y_test, voting_predictions)
accuracy_results["Voting Classifier"] = voting_accuracy

accuracy_df = pd.DataFrame(list(accuracy_results.items()), columns=["Model", "Accuracy"])

print("Model Accuracy Results:")
print(accuracy_df)

# Final model pipeline for saving
final_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=2000, random_state=42)),
            ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
            ("svc", SVC(probability=True, kernel="rbf", random_state=42)),
            ("gb", GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42))
        ],
        voting="soft"
    ))
])

final_model.fit(X_train, y_train)

# Save files
pickle.dump(final_model, open("ckd_model.sav", "wb"))
pickle.dump(label_encoder, open("ckd_label_encoder.sav", "wb"))
pickle.dump(X.columns.tolist(), open("ckd_feature_columns.sav", "wb"))

print("Voting Classifier saved as ckd_model.sav")
print("Label Encoder saved as ckd_label_encoder.sav")
print("Feature Columns saved as ckd_feature_columns.sav")