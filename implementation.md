# technical_implementation_guide.md

**Role:** Senior Machine Learning Engineer & Geospatial Specialist  
**Context:** EY AI & Data Challenge 2026 - South Africa Water Quality Prediction  
**Objective:** Predict `Total Alkalinity`, `Electrical Conductance`, `Dissolved Reactive Phosphorus`.

---

## 1. Data Pipeline Strategy

### 1.1. Ingestion & Standardization (The "Merge" Problem)
We have three disparate sources (Water Quality Targets, Landsat Features, TerraClimate). Direct floating-point merging on coordinates is risky due to precision issues.

**Protocol:**
1.  **Coordinate Precision:** Round `Latitude` and `Longitude` to **4 decimal places** (~11m precision) in ALL datasets immediately upon loading. This creates a stable composite key.
2.  **Date Parsing:** Convert `Sample Date` to `datetime` objects.
3.  **Merge Order:**
    *   **Base:** `water_quality_training_dataset.csv` (contains the Targets).
    *   **Join 1:** Inner Join with `landsat_features_training.csv` on `['lat_rounded', 'lon_rounded', 'Sample Date']`.
    *   **Join 2:** Left Join with `terraclimate_features_training.csv` on `['lat_rounded', 'lon_rounded', 'Sample Date']`.
    *   *Note:* Use Left Join for TerraClimate as climate data might be coarser or monthly aggregates. If exact date match fails, map to the closest `Year-Month`.

### 1.2. Handling Missing Data
*   **Landsat Gaps:** Satellite data often has NaNs due to cloud cover.
    *   *Strategy:* Filling with `0` is dangerous. Use **IterativeImputer** (or KNN Imputer) based on available bands, or fill with the median of that specific location (`groupby(['Latitude', 'Longitude'])`).
*   **TerraClimate:** Should be complete, but if missing, forward-fill (ffill) temporarily.

---

## 2. Feature Engineering Pro

### 2.1. Temporal Features (Capturing Seasonality)
South Africa has distinct hydrological cycles. Raw dates are hard for trees to interpret.
*   **Cyclical Encoding:** Transform `Month` (1-12) into coordinates on a circle to preserve December-January proximity.
    *   `month_sin = sin(2 * pi * month / 12)`
    *   `month_cos = cos(2 * pi * month / 12)`
*   **Season Categorical:** Create a `Season` feature (Summer, Autumn, Winter, Spring).
*   **Digital Year:** `Year + (DayOfYear / 365)` to capture long-term trends (droughts over years).

### 2.2. Spectral Features (Landsat Band Math)
We typically use Red for Turbidity, but we only have `nir`, `green`, `swir16`, `swir22`. We must innovate.
*   **Existing:** `NDMI` (Moisture), `MNDWI` (Water Index).
*   **New Derived Indices:**
    *   **Turbidity Proxy:** `green` band is highly sensitive to turbidity. Use raw `green` intensity.
    *   **Organic Matter / Salinity:** `swir` bands correlate with mineral content. Try Ratio: `swir16 / swir22`.
    *   **NIR / Green Ratio:** `nir / green` can help separate algal blooms (high NIR) from sediment (high Green).
*   **Interaction Features:**
    *   `Evap_Concentration = pet / (MNDWI + 1.1)`: Logic is high evaporation (`pet`) + low water (`MNDWI`) -> Concentration of pollutants increases.

---

## 3. Model Strategy: Multi-Output Regressor

The 3 targets (`Total Alkalinity`, `Electrical Conductance`, `Phosphorus`) are physically correlated. Predicting them in isolation ignores this relationship.

### 3.1. Structure
Use **Scikit-Learn's `MultiOutputRegressor`** wrapping a strong Gradient Boosting machine (XGBoost, LightGBM, or CatBoost).

```python
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

model = MultiOutputRegressor(
    XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        objective='reg:squarederror',
        n_jobs=-1
    )
)
```

### 3.2. Alternative: Regressor Chain
If correlation analysis shows specifically that $Target_A \rightarrow Target_B$, use `RegressorChain`. For example, *Electrical Conductance* (Salinity) often affects *Alkalinity*.
*   Order: `Conductance` -> `Alkalinity` -> `Phosphorus`.

---

## 4. Validation Framework: Spatial Cross-Validation

**Critical:** Random K-Fold is INVALID for geospatial data. If you have two samples from the same river 100m apart, one in Train and one in Test, the model just "memorizes" the location.

### 4.1. Spatial Group K-Fold
We need to force the model to generalize to *unseen locations*.
1.  **Cluster Locations:** Run `KMeans` on the unique coordinates (`Latitude`, `Longitude`) to create, say, 10 spatial clusters (`location_cluster`).
2.  **Split:** Use `GroupKFold` where `groups = location_cluster`.
    *   *Effect:* All samples from "River A" are either entirely in Train or entirely in Validation.

```python
from sklearn.model_selection import GroupKFold
from sklearn.cluster import KMeans

# define groups
coords = df[['Latitude', 'Longitude']].drop_duplicates()
kmeans = KMeans(n_clusters=10, random_state=42).fit(coords)
df['spatial_group'] = kmeans.predict(df[['Latitude', 'Longitude']])

# CV Loop
gkf = GroupKFold(n_splits=5)
for train_idx, val_idx in gkf.split(X, y, groups=df['spatial_group']):
    ...
```

---

## 5. Submission Flow & Inference

### 5.1. Preparation
1.  Load `submission_template.csv`.
2.  Load `landsat_features_validation.csv` and `terraclimate_features_validation.csv`.
3.  Perform the **exact** same Data Pipeline merge (Round Coords!) on these files.
4.  Generate same features (`month_sin`, `swir_ratio`, etc.).

### 5.2. Prediction
1.  Predict using the trained `MultiOutputRegressor`.
2.  **Post-Processing:** Ensure no negative values!
    *   `predictions = np.maximum(predictions, 0)` (Chemicals cannot be negative).
3.  Format columns to match template exactly: `['Total Alkalinity', 'Electrical Conductance', 'Dissolved Reactive Phosphorus']`.
4.  Save as `submission_v1_strategy_name.csv`.

---
**Next Steps for Team:**
*   **Member A:** Implement Section 1 (Data Pipeline) & 5 (Submission Gen).
*   **Member B:** Implement Section 2 (Features), 3 (Model), & 4 (Validation).
