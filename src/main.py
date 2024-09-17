from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report, confusion_matrix, roc_curve
import io

app = FastAPI()

def handle_nan(data):
    """Helper function to replace NaN values with None."""
    if isinstance(data, (list, np.ndarray)):
        return [handle_nan(item) for item in data]
    elif isinstance(data, dict):
        return {k: handle_nan(v) for k, v in data.items()}
    elif isinstance(data, float) and (data != data):  # Check for NaN
        return None
    else:
        return data

def perform_eda(df):
    # Basic info
    info = df.info()
    description = df.describe(include='all')  # Include all types for description
    missing_values = df.isnull().sum()

    # Distribution of the target variable
    if 'churn' in df.columns:
        plt.figure()
        sns.countplot(x='churn', data=df)
        plt.title("Target Variable Distribution")
        plt.close()  # Close to prevent display in API response

    # Correlation heatmap (only for numerical features)
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        plt.figure(figsize=(10, 6))
        sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm')
        plt.title("Correlation Heatmap")
        plt.close()  # Close to prevent display in API response

    return handle_nan({
        "info": str(info),
        "description": description.to_dict(),
        "missing_values": missing_values.to_dict()
    })

def preprocess_data(X):
    # Define the numerical and categorical columns
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Create the preprocessing pipelines for numerical and categorical features
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Create a ColumnTransformer to apply the appropriate pipeline to each column
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ]
    )

    return preprocessor


def churn_prediction_metrics(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = preprocess_data(X)

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_prob)

    class_report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    return handle_nan({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "auc_roc": auc_roc,
        "classification_report": class_report,
        "confusion_matrix": conf_matrix,
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist()
        }
    })

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

    
    eda_results = perform_eda(df)

    
    if 'churn' not in df.columns:
        return JSONResponse(content={"error": "The dataset does not contain a 'churn' column."}, status_code=400)

 
    X = df.drop(columns='churn')
    y = df['churn']

   
    metrics = churn_prediction_metrics(X, y)

    return JSONResponse(content={
        "eda": eda_results,
        "metrics": metrics
    })

