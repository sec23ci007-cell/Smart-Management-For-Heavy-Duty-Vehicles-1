import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. Data Generation
# ==========================================
def generate_data(num_samples=5000):
    np.random.seed(42)
    
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.Timedelta(days=365)
    random_days = np.random.randint(0, 365, num_samples)
    timestamps = [start_date + pd.Timedelta(days=d) for d in random_days]

    # Expanded attributes for dashboard realism
    vehicle_names_pool = [f"Transport Truck Model-{chr(65+i)}" for i in range(20)]
    petrol_stations_pool = ["Shell City West", "BP North Highway", "Mobil South Route", "Local Gas Station 7", "Exxon East Wing"]
    
    # Generate random strings based on vehicle ID mapping
    v_ids = np.random.randint(1, 21, num_samples)
    v_names = [vehicle_names_pool[i-1] for i in v_ids]
    
    # Generate mock License Plates (e.g. ABC-1234)
    plates_map = {i: f"{chr(np.random.randint(65, 91))}{chr(np.random.randint(65, 91))}{chr(np.random.randint(65, 91))}-{np.random.randint(1000, 9999)}" for i in range(1, 21)}
    v_plates = [plates_map[i] for i in v_ids]

    data = {
        'timestamp': timestamps,
        'vehicle_id': v_ids, 
        'vehicle_name': v_names,
        'license_plate': v_plates,
        'nearest_petrol_station': np.random.choice(petrol_stations_pool, num_samples),
        'fuel_level': np.random.uniform(0, 100, num_samples),
        'fuel_pressure': np.random.uniform(20, 90, num_samples),
        'fuel_consumption_rate': np.random.uniform(5, 35, num_samples),
        'brake_pressure': np.random.uniform(400, 1500, num_samples),
        'brake_fluid_level': np.random.uniform(0, 100, num_samples),
        'brake_temperature': np.random.uniform(100, 500, num_samples),
        'engine_temperature': np.random.uniform(160, 270, num_samples),
        'engine_vibration': np.random.uniform(0, 10, num_samples),
        'engine_rpm': np.random.uniform(800, 7000, num_samples),
        'oil_level': np.random.uniform(0, 100, num_samples),
        'mileage': np.random.uniform(0, 300000, num_samples),
        'maintenance_history': np.random.randint(0, 2, num_samples)
    }
    
    df = pd.DataFrame(data)
    
    # 2. Target Labels Generation based on logical rules
    # rule: low fuel pressure + high consumption -> fuel_issue
    df['fuel_issue'] = np.where((df['fuel_pressure'] < 40) & (df['fuel_consumption_rate'] > 20), 1, 0)
    
    # rule: low brake pressure + low brake fluid -> brake_line_failure
    df['brake_line_failure'] = np.where((df['brake_pressure'] < 700) & (df['brake_fluid_level'] < 30), 1, 0)
    
    # rule: high engine temperature + high vibration -> engine_maintenance_required
    df['engine_maintenance_required'] = np.where((df['engine_temperature'] > 230) & (df['engine_vibration'] > 6), 1, 0)
    
    # Add minor noise (5%) for realism so the models aren't heavily overfitted to deterministic rules
    for col in ['fuel_issue', 'brake_line_failure', 'engine_maintenance_required']:
        noise = np.random.choice([0, 1], size=num_samples, p=[0.95, 0.05])
        df[col] = np.logical_xor(df[col], noise).astype(int)
        
    return df

# ==========================================
# 3. Data Preprocessing
# ==========================================
def preprocess_data(df):
    features = [
        'fuel_level', 'fuel_pressure', 'fuel_consumption_rate',
        'brake_pressure', 'brake_fluid_level', 'brake_temperature',
        'engine_temperature', 'engine_vibration', 'engine_rpm',
        'oil_level', 'mileage', 'maintenance_history'
    ]
    
    X = df[features]
    
    splits = {'X_train': {}, 'X_test': {}, 'y_train': {}, 'y_test': {}}
    
    # Split the dataset into training and testing sets for each component prediction
    for target in ['fuel_issue', 'brake_line_failure', 'engine_maintenance_required']:
        X_train, X_test, y_train, y_test = train_test_split(X, df[target], test_size=0.2, random_state=42)
        
        # We can share exactly the same splits for all models since we used the same random state 
        splits['X_train'] = X_train
        splits['X_test'] = X_test
        splits['y_train'][target] = y_train
        splits['y_test'][target] = y_test
        
    return features, splits

# ==========================================
# 4. Model Training
# ==========================================
def train_models(splits):
    models = {}
    targets = ['fuel_issue', 'brake_line_failure', 'engine_maintenance_required']
    
    for target in targets:
        # separate Random Forest models definition and fitting
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(splits['X_train'], splits['y_train'][target])
        models[target] = clf
        
    return models

# ==========================================
# 5. Evaluation
# ==========================================
def evaluate_models(models, splits):
    results = {}
    targets = ['fuel_issue', 'brake_line_failure', 'engine_maintenance_required']
    
    for target in targets:
        clf = models[target]
        y_test = splits['y_test'][target]
        y_pred = clf.predict(splits['X_test'])
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        results[target] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'confusion_matrix': cm
        }
        
        print(f"--- {target} Model ---")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}\n")
        
    return results

# ==========================================
# 6. Prediction System
# ==========================================
def predict_vehicle_issues(models, features_list, new_reading):
    """
    Accepts new sensor readings and predicts issues.
    """
    reading_df = pd.DataFrame([new_reading], columns=features_list)
    predictions = {}
    
    for target, model in models.items():
        pred = model.predict(reading_df)[0]
        predictions[target] = "Issue Detected" if pred == 1 else "Normal/Healthy"
        
    return predictions

# ==========================================
# 7. Visualization
# ==========================================
def visualize_results(df, models, features, results):
    # Try to set ggplot style, fail silently if unavailable
    try:
        plt.style.use('ggplot')
    except:
        pass
        
    targets = ['fuel_issue', 'brake_line_failure', 'engine_maintenance_required']
    
    # 7.1 Correlation Heatmap
    plt.figure(figsize=(12, 10))
    corr_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in ['timestamp', 'vehicle_id']]
    corr = df[corr_cols].corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Feature and Target Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()
    
    # 7.2 Feature Importance for each model
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for i, target in enumerate(targets):
        importances = models[target].feature_importances_
        indices = np.argsort(importances)[::-1]
        
        axes[i].bar(range(len(features)), importances[indices], align="center")
        axes[i].set_xticks(range(len(features)))
        axes[i].set_xticklabels([features[idx] for idx in indices], rotation=90)
        axes[i].set_title(f"Feature Importances: {target}")
    
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    # 7.3 Model Performance Graphs
    metrics = ['accuracy', 'precision', 'recall']
    performance = {metric: [results[t][metric] for t in targets] for metric in metrics}
    
    x = np.arange(len(targets))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, performance['accuracy'], width, label='Accuracy')
    ax.bar(x, performance['precision'], width, label='Precision')
    ax.bar(x + width, performance['recall'], width, label='Recall')
    
    ax.set_ylabel('Scores')
    ax.set_title('Model Performance Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.legend(loc='lower left')
    ax.set_ylim(0, 1.25) # Slightly extended to fit legend comfortably
    
    plt.tight_layout()
    plt.savefig('model_performance.png')
    plt.close()
    
    # 7.4 Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, target in enumerate(targets):
        sns.heatmap(results[target]['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'Confusion Matrix: {target}')
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('True Label')
        
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    plt.close()
    
    print("Visualizations saved as PNG files:")
    print(" - correlation_heatmap.png")
    print(" - feature_importance.png")
    print(" - model_performance.png")
    print(" - confusion_matrices.png")

if __name__ == "__main__":
    print("1. Generating synthetic dataset...")
    df = generate_data(5000)
    print("Dataset shape:", df.shape)
    
    print("\n2. Preprocessing and splitting data...")
    features, splits = preprocess_data(df)
    
    print("\n3. Training Random Forest models...")
    models = train_models(splits)
    
    print("\n4. Evaluating models...")
    results = evaluate_models(models, splits)
    
    print("\n5. Generating visualizations...")
    visualize_results(df, models, features, results)
    
    print("\n6. Testing Prediction System with a new sample...")
    # Sample reading with specific conditions engineered to trigger issue predictions
    # (Low Fuel Pressure & High Consumption -> Fuel Issue)
    # (High Engine Temp & High Vibration -> Engine Maintenance)
    sample_reading = {
        'fuel_level': 15.0,
        'fuel_pressure': 35.0,        # Low
        'fuel_consumption_rate': 25.0,# High
        'brake_pressure': 900.0,
        'brake_fluid_level': 80.0,
        'brake_temperature': 200.0,
        'engine_temperature': 240.0,  # High
        'engine_vibration': 7.5,      # High
        'engine_rpm': 3000.0,
        'oil_level': 45.0,
        'mileage': 120000.0,
        'maintenance_history': 0
    }
    
    predictions = predict_vehicle_issues(models, features, sample_reading)
    print("New Sensor Reading:")
    for k, v in sample_reading.items():
        print(f"  {k}: {v}")
    print("\nPredictions:")
    for system, status in predictions.items():
        print(f"  {system}: {status}")
