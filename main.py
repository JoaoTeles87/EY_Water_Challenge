import sys
import os

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from baseline_pipeline import load_and_preprocess_data, create_features, get_spatial_groups, train_and_evaluate

def main():
    print("🚀 Starting EY Water Challenge Project Pipeline...")
    
    # 1. Data Loading & Preprocessing
    data = load_and_preprocess_data()
    
    # 2. Feature Engineering
    data, drop_cols = create_features(data)
    
    # 3. Spatial Grouping
    data = get_spatial_groups(data)
    
    # 4. Model Training & Evaluation (with Logging)
    model = train_and_evaluate(data, drop_cols)
    
    print("\n✅ Pipeline Finished Successfully.")

if __name__ == "__main__":
    main()
