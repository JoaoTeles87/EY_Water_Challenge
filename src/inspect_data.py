import pandas as pd
import os

def analyze_data():
    base_path = "data"
    
    print("Loading data...")
    wq = pd.read_csv(os.path.join(base_path, 'water_quality_training_dataset.csv'))
    landsat = pd.read_csv(os.path.join(base_path, 'landsat_features_training.csv'))
    climate = pd.read_csv(os.path.join(base_path, 'terraclimate_features_training.csv'))
    

    with open('data_analysis.txt', 'w') as f:
        f.write(f"--- Data Shapes ---\n")
        f.write(f"Water Quality (Targets): {wq.shape}\n")
        f.write(f"Landsat (Features): {landsat.shape}\n")
        f.write(f"TerraClimate (Features): {climate.shape}\n")
        
        f.write(f"\n--- Columns ---\n")
        f.write(f"Landsat Columns: {landsat.columns.tolist()}\n")
        f.write(f"TerraClimate Columns: {climate.columns.tolist()}\n")
        
        # Simulate Merge
        f.write(f"\n--- Merge Analysis ---\n")
        for df in [wq, landsat, climate]:
            df['lat_rounded'] = df['Latitude'].round(4)
            df['lon_rounded'] = df['Longitude'].round(4)
            df['Sample Date'] = pd.to_datetime(df['Sample Date'], dayfirst=True)
            
        merged_inner = pd.merge(wq, landsat, on=['lat_rounded', 'lon_rounded', 'Sample Date'], how='inner')
        f.write(f"Merged (Inner Join with Landsat): {merged_inner.shape}\n")
        lost_rows = wq.shape[0] - merged_inner.shape[0]
        f.write(f"Rows Lost: {lost_rows} ({(lost_rows/wq.shape[0])*100:.2f}%)\n")

if __name__ == "__main__":
    analyze_data()
