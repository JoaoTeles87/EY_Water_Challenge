import pandas as pd
import numpy as np
import datetime
import os
import sys
from sklearn.multioutput import MultiOutputRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, make_scorer
from xgboost import XGBRegressor
import warnings

from logger import ExperimentLogger

# Suppress warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    print("Loading datasets...")
    # Load Datasets (Paths updated for src/ execution context relative to root or ../data)
    # Assuming running from root via python src/baseline_pipeline.py or similar
    # Adjusting to absolute or relative paths from execution dir
    
    # Check if inside src or root
    base_path = "data" if os.path.exists("data") else "../data"
    
    wq_df = pd.read_csv(os.path.join(base_path, 'water_quality_training_dataset.csv'))
    landsat_df = pd.read_csv(os.path.join(base_path, 'landsat_features_training.csv'))
    climate_df = pd.read_csv(os.path.join(base_path, 'terraclimate_features_training.csv'))

    # 1. Coordinate Precision: Round to 4 decimals
    print("Rounding coordinates...")
    for df in [wq_df, landsat_df, climate_df]:
        df['lat_rounded'] = df['Latitude'].round(4)
        df['lon_rounded'] = df['Longitude'].round(4)
        # Ensure Sample Date is datetime
        df['Sample Date'] = pd.to_datetime(df['Sample Date'], dayfirst=True)

    # 2. Merge Strategy
    print("Merging datasets...")
    # Inner Join with Landsat ( Satellite data is essential)
    merged_df = pd.merge(
        wq_df, 
        landsat_df, 
        on=['lat_rounded', 'lon_rounded', 'Sample Date'], 
        how='inner',
        suffixes=('', '_landsat')
    )
    
    # Left Join with TerraClimate (Regional context, might be coarser)
    merged_df = pd.merge(
        merged_df, 
        climate_df, 
        on=['lat_rounded', 'lon_rounded', 'Sample Date'], 
        how='left',
        suffixes=('', '_climate')
    )

    print(f"Merged Dataset Shape: {merged_df.shape}")
    return merged_df

def create_features(df):
    print("Engineering features...")
    # 3. Temporal Features
    df['Month'] = df['Sample Date'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    
    # Drop non-numeric for X (keeping identifiers for grouping if needed)
    drop_cols = ['Latitude', 'Longitude', 'Sample Date', 'lat_rounded', 'lon_rounded', 
                 'Latitude_landsat', 'Longitude_landsat', 'Latitude_climate', 'Longitude_climate', 'Month']
    
    # Ensure we don't drop columns that don't exist
    drop_cols = [c for c in drop_cols if c in df.columns]
    
    return df, drop_cols

def get_spatial_groups(df, n_clusters=10):
    print("Creating spatial clusters...")
    # Use original/rounded coordinates for clustering
    coords = df[['lat_rounded', 'lon_rounded']].copy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['spatial_group'] = kmeans.fit_predict(coords)
    return df

def train_and_evaluate(df, drop_cols):
    print("Training and evaluating model...")
    # Define Targets and Features
    targets = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
    
    # Separate X and y
    X = df.drop(columns=targets + drop_cols + ['spatial_group'])
    y = df[targets]
    groups = df['spatial_group']
    
    # Check for NaN in features
    print(f"Features with NaNs: {X.columns[X.isna().any()].tolist()}")

    # Pipeline: Imputer -> Scaler -> Model
    # XGBoost can handle NaNs, but Imputer is requested in the prompt
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('regressor', MultiOutputRegressor(
            XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                random_state=42,
                n_jobs=-1
            )
        ))
    ])

    # Validation: Spatial Group K-Fold
    gkf = GroupKFold(n_splits=5)
    
    # Custom scoring or simple loop
    rmse_scores = []
    
    print("\nStarting Spatial Cross-Validation...")
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_val)
        
        # Clip negative predictions (physical constraint)
        preds = np.maximum(preds, 0)
        
        rmse = np.sqrt(mean_squared_error(y_val, preds, multioutput='raw_values'))
        print(f"Fold {fold+1} RMSE per target (Alk, Cond, Phos): {np.round(rmse, 2)}")
        rmse_scores.append(rmse)

    avg_rmse = np.mean(rmse_scores, axis=0)
    overall_rmse = np.mean(avg_rmse)
    print(f"\nAverage Spatial RMSE: {np.round(avg_rmse, 2)}")
    print(f"Overall Mean RMSE: {overall_rmse:.2f}")
    
    # Log results using Professional Logger
    # Extract params from model (example)
    params = pipeline.named_steps['regressor'].estimator.get_params()
    # Filter only key params for brevity in log
    key_params = {k: v for k, v in params.items() if k in ['n_estimators', 'learning_rate', 'max_depth']}
    
    logger = ExperimentLogger(experiment_name="Baseline_XGBoost", log_dir="experiments")
    logger.log_run(
        params=key_params,
        metrics={
            "overall_rmse": overall_rmse,
            "rmse_alkalinity": avg_rmse[0],
            "rmse_conductance": avg_rmse[1],
            "rmse_phosphorus": avg_rmse[2]
        },
        notes="Baseline run with structure refactor."
    )

    # Restore Legacy Text Logging (User Preference)
    # Check if inside src or root to find experiments dir
    exp_dir = "experiments" if os.path.exists("experiments") else "../experiments"
    txt_log_path = os.path.join(exp_dir, "experiment_history.txt")
    
    with open(txt_log_path, "a") as f:
        f.write(f"\n{'='*30}\n")
        f.write(f"Experiment: Day 1 Baseline (XGBoost) - Refactored\n")
        f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
        f.write(f"Features: Cyclical Month, Landsat, TerraClimate\n")
        f.write(f"CV Strategy: Spatial Group K-Fold (5 splits)\n")
        f.write(f"RMSE per Target (Alk, Cond, Phos): {np.round(avg_rmse, 2)}\n")
        f.write(f"Overall Mean RMSE: {overall_rmse:.4f}\n")
        f.write(f"{'='*30}\n")
        
    print(f"\nResults saved to '{txt_log_path}' and JSON Logger.")
    
    return pipeline

if __name__ == "__main__":
    # Execution Flow
    data = load_and_preprocess_data()
    data, drop_cols = create_features(data)
    data = get_spatial_groups(data)
    
    model = train_and_evaluate(data, drop_cols)
    print("\nBaseline Pipeline Completed successfully.")
