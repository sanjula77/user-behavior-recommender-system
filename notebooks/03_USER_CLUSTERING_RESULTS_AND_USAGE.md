# 📊 User Clustering - Results & Usage Guide

## 🎯 **What This Notebook Does**

The `03_user_clustering.ipynb` notebook performs **unsupervised machine learning** to discover hidden patterns in user behavior. It groups similar users together into segments without knowing the "answers" beforehand.

### **Main Purpose:**
- **Discover user segments** based on behavior patterns
- **Identify outliers** (anomalous users)
- **Create behavioral labels** for marketing and personalization

---

## 📋 **Complete Workflow**

### **Cell 0: Load Features**
- Loads processed user features from Phase 2
- **Input**: `features_user_ml.csv` (28 features per user)
- **Output**: Feature matrix ready for clustering
- **Data**: 5000 users with behavioral features

### **Cell 1: Find Optimal K (K-Means)**
- Uses **Elbow Method** and **Silhouette Score** to find best number of clusters
- **Result**: Optimal K = 4 (or detected automatically)
- **Output**: `best_k` value stored for next cell

### **Cell 2: K-Means Clustering**
- Groups users into K clusters based on behavior similarity
- **Result**: Each user gets a `kmeans_label` (0, 1, 2, 3, ...)
- **Output**: Cluster labels assigned to all users
- **Saves**: `user_clusters.csv`

### **Cell 3: DBSCAN Outlier Detection**
- Identifies anomalous users who don't fit normal patterns
- **Method**: Uses k-distance graph to find optimal `eps` parameter
- **Result**: Outlier rate ~1.94% (97 outliers out of 5000)
- **Output**: `dbscan_label` (0 = Regular, -1 = Outlier)
- **Key**: Dynamically calculates `eps` from data (not hardcoded)

### **Cell 4: Merge K-Means + DBSCAN**
- Combines both clustering results
- **Output**: Users have both `kmeans_label` and `dbscan_label`
- **Saves**: 
  - `user_clusters_merged.csv`
  - `user_clusters_summary.csv` (statistics by cluster)

### **Cell 5: Assign Behavioral Labels**
- Creates human-readable segment names based on cluster characteristics
- **Segments Created**:
  - **High-Value Buyers**: Users with high purchase activity
  - **Quick Visitors / Bouncers**: Users with low engagement
  - **Outlier / Anomaly**: Users with unusual behavior patterns
- **Output**: `behavior_segment` column added
- **Saves**: `user_segments_final.csv` ⭐ **MAIN OUTPUT**

### **Cell 6: PCA Visualization**
- Reduces 28 features to 2D for visualization
- Shows clusters in 2D space
- **Variance Explained**: ~27% (low, but useful for visualization)

### **Cell 7: DBSCAN Outlier Summary**
- Summary statistics of outliers vs regular users
- Shows which clusters have high outlier rates

### **Cells 8-10: Data Inspection**
- Final dataset validation
- Data quality checks
- Statistical summaries

### **Cells 11-12: Summary & Recommendations**
- Clustering quality assessment
- Recommendations for improvement

---

## 💾 **Output Files Generated**

### **1. Main Output File** ⭐
**File**: `data/processed/user_segments_final.csv`

**Contains**:
- All original features (28 columns)
- `user_id` - User identifier
- `behavior_label` - Original label from data generation
- `kmeans_label` - K-Means cluster number (0-3)
- `pca1`, `pca2` - 2D coordinates for visualization
- `dbscan_label` - Outlier detection (-1 = Outlier, 0 = Regular)
- `behavior_segment` - Human-readable segment name ⭐

**Shape**: (5000, 35) - 5000 users with 35 columns

**Usage**: This is the main file used by the ML models notebook!

### **2. Intermediate Files**

**`user_clusters.csv`**
- Contains: `user_id`, `kmeans_cluster`, `behavior_label`
- Usage: Quick reference for K-Means clusters only

**`user_clusters_merged.csv`**
- Contains: All features + K-Means + DBSCAN labels
- Usage: Before behavioral labels are assigned

**`user_clusters_summary.csv`**
- Contains: Statistical summary by cluster
- Usage: Understand cluster characteristics

---

## 📊 **Key Results**

### **Clustering Results**

#### **K-Means Clusters** (Optimal K = 4)
- **Cluster 0**: High-Value Buyers (976 users, 19.5%)
- **Cluster 1**: Quick Visitors / Bouncers (3927 users, 78.5%)
- **Cluster 2**: Outlier / Anomaly (97 users, 1.9%)

#### **DBSCAN Outlier Detection**
- **Outlier Rate**: ~1.94% (97 users)
- **Method**: Dynamic `eps` calculation (95th percentile of k-distances)
- **Result**: Much more reasonable than hardcoded threshold

#### **Behavioral Segments Created**
1. **High-Value Buyers** (976 users)
   - High purchase activity
   - High engagement
   - Longer session duration

2. **Quick Visitors / Bouncers** (3927 users)
   - Low engagement
   - Short sessions
   - High bounce rate

3. **Outlier / Anomaly** (97 users)
   - Unusual behavior patterns
   - Don't fit normal clusters
   - May need special attention

### **Clustering Quality Metrics**

- **Silhouette Score**: ~0.23 (Low - indicates weak clustering structure)
- **Note**: This is common with high-dimensional behavioral data
- **Recommendation**: Consider feature selection or dimensionality reduction

### **PCA Visualization**
- **Variance Explained**: ~27% in 2D
- **Usage**: Visual representation of clusters (not for prediction)

---

## 🚀 **How to Use These Results**

### **1. Load and Analyze Segments**

```python
import pandas as pd
from pathlib import Path

# Load the final segments file
DATA_DIR = Path("data/processed")
df = pd.read_csv(DATA_DIR / "user_segments_final.csv")

# View segment distribution
print(df['behavior_segment'].value_counts())

# Analyze segment characteristics
segment_stats = df.groupby('behavior_segment').agg({
    'total_purchases': 'mean',
    'avg_session_duration': 'mean',
    'total_clicks': 'mean',
    'conversion_rate': 'mean'
})
print(segment_stats)
```

### **2. Create Marketing Campaigns**

```python
# Target High-Value Buyers
high_value = df[df['behavior_segment'] == 'High-Value Buyers']
print(f"High-Value Buyers: {len(high_value)} users")
print(f"Average purchases: {high_value['total_purchases'].mean():.2f}")

# Send them premium offers, loyalty programs, exclusive products
high_value_user_ids = high_value['user_id'].tolist()
# Use these IDs for targeted marketing campaigns
```

### **3. Identify At-Risk Users**

```python
# Find Quick Visitors (likely to bounce)
quick_visitors = df[df['behavior_segment'] == 'Quick Visitors / Bouncers']
print(f"At-risk users: {len(quick_visitors)} users")

# Show them special offers, retargeting ads, exit-intent popups
at_risk_user_ids = quick_visitors['user_id'].tolist()
```

### **4. Handle Outliers**

```python
# Identify anomalous users
outliers = df[df['dbscan_label'] == -1]
print(f"Outliers detected: {len(outliers)} users")

# Investigate these users
outlier_stats = outliers.describe()
print(outlier_stats)

# May need:
# - Fraud detection
# - Special customer support
# - Data quality review
```

### **5. Create Dashboards**

```python
import matplotlib.pyplot as plt

# Segment distribution pie chart
segment_counts = df['behavior_segment'].value_counts()
plt.figure(figsize=(10, 6))
plt.pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%')
plt.title('User Segment Distribution')
plt.show()

# Segment comparison
segment_comparison = df.groupby('behavior_segment').agg({
    'total_purchases': 'mean',
    'avg_session_duration': 'mean',
    'conversion_rate': 'mean'
})
segment_comparison.plot(kind='bar', figsize=(12, 6))
plt.title('Segment Characteristics Comparison')
plt.show()
```

### **6. Join with Other Data**

```python
# Join segments with session data
sessions = pd.read_csv("data/processed/session_features.csv")
segments = pd.read_csv("data/processed/user_segments_final.csv")

# Merge on user_id
merged = sessions.merge(
    segments[['user_id', 'behavior_segment', 'kmeans_label']],
    on='user_id',
    how='left'
)

# Analyze session behavior by segment
session_by_segment = merged.groupby('behavior_segment').agg({
    'session_duration': 'mean',
    'page_views': 'mean',
    'events': 'mean'
})
print(session_by_segment)
```

### **7. Use for Personalization**

```python
def get_user_segment(user_id):
    """Get segment for a specific user"""
    user_data = df[df['user_id'] == user_id]
    if len(user_data) > 0:
        return user_data['behavior_segment'].iloc[0]
    return None

# Example: Personalize experience based on segment
user_id = "U00001"
segment = get_user_segment(user_id)

if segment == "High-Value Buyers":
    # Show premium products, loyalty rewards
    show_premium_content()
elif segment == "Quick Visitors / Bouncers":
    # Show discounts, exit-intent offers
    show_special_offers()
```

### **8. Export for Business Intelligence Tools**

```python
# Export to Excel for business analysis
df.to_excel("data/processed/user_segments_final.xlsx", index=False)

# Export specific columns for dashboards
dashboard_data = df[[
    'user_id',
    'behavior_segment',
    'kmeans_label',
    'total_purchases',
    'conversion_rate',
    'avg_session_duration'
]]
dashboard_data.to_csv("data/processed/segments_dashboard.csv", index=False)
```

---

## 📈 **Business Applications**

### **1. Marketing Segmentation**
- **High-Value Buyers**: Premium campaigns, loyalty programs
- **Quick Visitors**: Retargeting ads, special discounts
- **Outliers**: Investigate for fraud or data issues

### **2. Product Recommendations**
- Show different products based on segment
- High-Value Buyers: Premium products
- Quick Visitors: Popular/trending items

### **3. Pricing Strategies**
- Segment-based pricing
- Dynamic offers based on behavior patterns

### **4. Customer Support**
- Route High-Value Buyers to premium support
- Proactive outreach to Quick Visitors

### **5. A/B Testing**
- Test different experiences by segment
- Measure impact of changes per segment

---

## 🔍 **Understanding the Segments**

### **High-Value Buyers** (976 users, 19.5%)
**Characteristics**:
- High purchase activity
- Longer session duration
- Higher engagement
- Lower bounce rate

**Action**: Premium treatment, loyalty rewards, exclusive access

### **Quick Visitors / Bouncers** (3927 users, 78.5%)
**Characteristics**:
- Low engagement
- Short sessions
- High bounce rate
- Few purchases

**Action**: Retargeting, special offers, exit-intent popups

### **Outlier / Anomaly** (97 users, 1.9%)
**Characteristics**:
- Unusual behavior patterns
- Don't fit normal clusters
- May be bots, fraud, or edge cases

**Action**: Investigate, fraud detection, data quality review

---

## ⚠️ **Important Notes**

### **1. Clustering Quality**
- Silhouette Score is low (0.23) - this is common with behavioral data
- Clusters are still useful for business purposes
- Consider feature engineering to improve separation

### **2. Outlier Detection**
- DBSCAN uses dynamic `eps` calculation (not hardcoded)
- Outlier rate is reasonable (~2%)
- Review outliers manually for business insights

### **3. Segment Labels**
- Behavioral segments are assigned based on cluster characteristics
- Labels are interpretable for business use
- Can be refined based on business needs

### **4. Data Consistency**
- All features are standardized (mean=0, std=1)
- Original values can be recovered if needed
- PCA coordinates are for visualization only

---

## 🔄 **Integration with Other Notebooks**

### **Pipeline Flow**:
```
01_synthetic_data_generator.ipynb
    ↓ (generates raw data)
02_data_preprocessing_feature_engineering.ipynb
    ↓ (creates features)
03_user_clustering.ipynb  ← YOU ARE HERE
    ↓ (creates segments)
04_supervised_ml_models.ipynb
    ↓ (uses segments for predictions)
```

### **How ML Models Use This**:
- The `user_segments_final.csv` file is loaded in `04_supervised_ml_models.ipynb`
- Segments can be used as features in ML models
- Clustering helps understand user behavior patterns

---

## 📊 **Key Metrics to Monitor**

### **Segment Health**
- Segment size distribution
- Average purchase value by segment
- Conversion rate by segment
- Engagement metrics by segment

### **Clustering Quality**
- Silhouette Score (aim for >0.3)
- Within-cluster variance
- Between-cluster separation

### **Outlier Rate**
- Should be 1-5% for normal data
- Too high: may need parameter tuning
- Too low: may miss anomalies

---

## 🎯 **Quick Reference**

### **Main Output File**
- `data/processed/user_segments_final.csv` ⭐
- Contains all features + cluster labels + behavioral segments

### **Key Columns**
- `behavior_segment` - Human-readable segment name
- `kmeans_label` - Cluster number (0-3)
- `dbscan_label` - Outlier flag (-1 = outlier, 0 = regular)
- `user_id` - User identifier

### **Segment Distribution**
- High-Value Buyers: ~20%
- Quick Visitors: ~78%
- Outliers: ~2%

### **Next Steps**
1. ✅ Analyze segment characteristics
2. ✅ Create marketing campaigns by segment
3. ✅ Use segments in ML models (next notebook)
4. ✅ Build dashboards for business insights

---

**This notebook creates the foundation for user segmentation and personalization!** 🚀

