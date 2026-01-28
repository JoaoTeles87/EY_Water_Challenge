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
    
    # 3.1 Advanced Features (Physics-Informed)
    # Digital Year (Trend Analysis)
    df['digital_year'] = df['Sample Date'].dt.year + (df['Sample Date'].dt.dayofyear / 365.0)
    
    # SWIR Ratio (Salinity/Mineral Proxy)
    # Add epsilon to avoid division by zero
    epsilon = 1e-6
    df['swir_ratio'] = df['swir16'] / (df['swir22'] + epsilon)
    
    # NIR / Green Ratio (Algae vs Sediment)
    df['nir_green_ratio'] = df['nir'] / (df['green'] + epsilon)
    
    # Evaporation Concentration (Pollutant Concentration)
    # Logic: High Evap (pet) + Low Water (MNDWI) -> Higher Concentration
    # MNDWI is typically -1 to 1. Adding 1.1 ensures positive denominator.
    df['evap_concentration'] = df['pet'] / (df['MNDWI'] + 1.1)
    
    # Clean up Infs (if any slipped through)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
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

def generate_submission(model, drop_cols):
    print("\nStarting Submission Generation...")
    base_path = "data" if os.path.exists("data") else "../data"
    
    # 1. Load Template and Validation Features
    sub_df = pd.read_csv(os.path.join(base_path, 'submission_template.csv'))
    landsat_val = pd.read_csv(os.path.join(base_path, 'landsat_features_validation.csv'))
    climate_val = pd.read_csv(os.path.join(base_path, 'terraclimate_features_validation.csv'))
    
    print(f"Template Shape: {sub_df.shape}")
    
    # 2. Preprocess (Round Coords + Date)
    for df in [sub_df, landsat_val, climate_val]:
        df['lat_rounded'] = df['Latitude'].round(4)
        df['lon_rounded'] = df['Longitude'].round(4)
        df['Sample Date'] = pd.to_datetime(df['Sample Date'], dayfirst=True)

    # 3. Merge (Same strategy as training)
    # Inner Join with Landsat
    merged_sub = pd.merge(
        sub_df, 
        landsat_val, 
        on=['lat_rounded', 'lon_rounded', 'Sample Date'], 
        how='inner',
        suffixes=('', '_landsat')
    )
    
    # Left Join with TerraClimate
    merged_sub = pd.merge(
        merged_sub, 
        climate_val, 
        on=['lat_rounded', 'lon_rounded', 'Sample Date'], 
        how='left',
        suffixes=('', '_climate')
    )
    
    print(f"Merged Submission Shape: {merged_sub.shape}")
    
    # 4. Feature Engineering (Same function)
    # Note: create_features expects 'Month' which is derived from 'Sample Date'
    merged_sub, _ = create_features(merged_sub)
    
    # 5. Prepare X (Features only)
    targets = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
    
    # Ensure we use exactly the same columns as training
    # We need to drop targets from the dataframe if they exist (they verify 200 rows)
    # But in template they are empty or placeholder, so just ensure we don't include them in X
    
    # Construct X using the pipeline's feature names? 
    # The pipeline handles raw input, but we need to ensure columns match.
    # We used drop_cols in training to remove non-features. We do the same here.
    # Also drop spatial_group if it was added (it's not needed for prediction if not used as feature)
    # If the model uses spatial_group as feature, we need to generate it.
    # In training: X = df.drop(columns=targets + drop_cols + ['spatial_group'])
    # So spatial_group is NOT a feature.
    
    X_sub = merged_sub.drop(columns=targets + drop_cols, errors='ignore')
    
    # Check for NaNs
    if X_sub.isna().any().any():
        print("Warning: NaNs found in submission features. Pipeline imputer will handle them.")
        
    # 6. Predict
    predictions = model.predict(X_sub)
    
    # 7. Post-process (Clip negatives)
    predictions = np.maximum(predictions, 0)
    
    # 8. Create Submission File
    submission = sub_df[['Latitude', 'Longitude', 'Sample Date']].copy()
    submission[targets] = predictions
    
    # Save
    save_path = "submission.csv"
    submission.to_csv(save_path, index=False)
    print(f"Submission saved to: {os.path.abspath(save_path)}")
    print(submission.head())

if __name__ == "__main__":
    # Execution Flow
    data = load_and_preprocess_data()
    data, drop_cols = create_features(data)
    data = get_spatial_groups(data)
    
    model = train_and_evaluate(data, drop_cols)
    
    # Generate Submission
    # We use the fitted pipeline 'model'
    # Important: The 'model' returned by train_and_evaluate is trained on the LAST fold in the loop?
    # Actually, inside the loop it fits on train_idx.
    # Refitting on FULL data is best practice for submission.
    
    print("\nRefitting model on FULL dataset for submission...")
    targets = ['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']
    X_full = data.drop(columns=targets + drop_cols + ['spatial_group'])
    y_full = data[targets]
    
    model.fit(X_full, y_full)
    
    generate_submission(model, drop_cols)
    
    print("\nBaseline Pipeline Completed successfully.")
