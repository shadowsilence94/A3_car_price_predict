#!/usr/bin/env python3
"""
A3 MLflow Final - No Polynomial Features
Student ID: st126010 - Htut Ko Ko
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from LogisticRegression import LogisticRegression
import mlflow
from mlflow import MlflowClient
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

def prepare_data():
    """Prepare data without polynomial features"""
    # Load data
    data = pd.read_csv('Cars.csv')
    
    # Create price classes
    def classify_price(price):
        if price <= 2500000: return 0  # Low
        elif price <= 5000000: return 1  # Medium
        elif price <= 10000000: return 2  # High
        else: return 3  # Premium
    
    data['price_class'] = data['selling_price'].apply(classify_price)
    
    # Define features
    numeric_columns = ['year', 'km_driven']
    categorical_columns = ['fuel', 'seller_type', 'transmission', 'owner']
    
    # Handle missing values
    for col in numeric_columns:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].median())
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_columns:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            label_encoders[col] = le
    
    # Features and target
    feature_names = numeric_columns + categorical_columns
    X = data[feature_names]
    y = data['price_class'].values
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names, label_encoders

def custom_classification_metrics(y_true, y_pred, n_classes):
    """Calculate metrics"""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label, pred_label] += 1
    
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    
    precisions, recalls, f1s = [], [], []
    for i in range(n_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    
    return {
        'accuracy': accuracy,
        'macro_precision': np.mean(precisions),
        'macro_recall': np.mean(recalls),
        'macro_f1': np.mean(f1s),
        'weighted_f1': np.average(f1s, weights=np.sum(cm, axis=1))
    }

def main():
    print("🚀 A3 MLflow Final - Original Experiment")
    
    # MLflow setup
    os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow.ml.brain.cs.ait.ac.th/"
    os.environ["MLFLOW_TRACKING_USERNAME"] = "admin"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "password"
    
    experiment_name = "st126010-a3"  # Original name
    mlflow.set_experiment(experiment_name)
    
    # Prepare data
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names, label_encoders = prepare_data()
    n_classes = len(np.unique(y_train))
    
    print(f"Training shape: {X_train_scaled.shape}")
    print(f"Classes: {n_classes}")
    
    # Test key configurations
    configs = [
        {'penalty': None, 'init_method': 'xavier', 'learning_rate': 0.01},
        {'penalty': 'ridge', 'lambda_reg': 0.01, 'init_method': 'xavier', 'learning_rate': 0.01},
        {'penalty': None, 'init_method': 'zeros', 'learning_rate': 0.01},
    ]
    
    best_accuracy = 0
    best_run_id = None
    best_model_artifacts = None
    
    for i, config in enumerate(configs):
        run_name = f"final-model-{i+1}"
        
        with mlflow.start_run(run_name=run_name) as run:
            # Log parameters
            for key, value in config.items():
                mlflow.log_param(key, value)
            
            # Train model
            model = LogisticRegression(**config, max_iter=1000)
            model.fit(X_train_scaled, y_train, n_classes=n_classes)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            metrics = custom_classification_metrics(y_test, y_pred, n_classes)
            
            # Log metrics
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            
            # Save artifacts
            model_artifacts = {
                'model': model,
                'scaler': scaler,
                'feature_names': feature_names,
                'label_encoders': label_encoders,
                'n_classes': n_classes
            }
            
            with open('a3_model_artifacts.pkl', 'wb') as f:
                pickle.dump(model_artifacts, f)
            mlflow.log_artifact('a3_model_artifacts.pkl')
            
            # Track best
            if metrics['accuracy'] > best_accuracy:
                best_accuracy = metrics['accuracy']
                best_run_id = run.info.run_id
                best_model_artifacts = model_artifacts
            
            print(f"Model {i+1}: Accuracy = {metrics['accuracy']:.4f}")
    
    print(f"\n🏆 Best accuracy: {best_accuracy:.4f}")
    
    # Stage best model
    if best_run_id:
        print("\n📦 Staging best model...")
        
        client = MlflowClient()
        model_name = "st126010-a3-model"
        
        try:
            try:
                client.create_registered_model(model_name)
            except:
                pass
            
            model_version = client.create_model_version(
                name=model_name,
                source=f"runs:/{best_run_id}/a3_model_artifacts.pkl",
                run_id=best_run_id
            )
            
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Staging"
            )
            
            print(f"✅ Model v{model_version.version} staged successfully")
            
        except Exception as e:
            print(f"❌ Staging error: {e}")
    
    # Save locally
    if best_model_artifacts:
        with open('model_artifacts.pkl', 'wb') as f:
            pickle.dump(best_model_artifacts, f)
        print("✅ Model saved locally")
    
    print("🎉 Complete!")

if __name__ == "__main__":
    main()
