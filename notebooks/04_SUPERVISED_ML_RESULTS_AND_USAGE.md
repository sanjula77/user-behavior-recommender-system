# 🤖 Supervised ML Models - Results & Usage Guide

## 🎯 **What This Notebook Does**

The `04_supervised_ml_models.ipynb` notebook builds **predictive machine learning models** to forecast user behavior. It takes processed user data and trains models to predict:

1. **Conversion** - Will the user make a purchase? (Primary focus)
2. **Bounce** - Will the user leave immediately? (With threshold tuning)
3. **Return** - Will the user come back? (Labels created, model not trained)

---

## 📋 **Complete Workflow**

### **Cell 0: Load Dataset**
- **Purpose**: Loads the processed user segments from clustering notebook
- **Input**: `user_segments_final.csv` (from `03_user_clustering.ipynb`)
- **Output**: DataFrame with 5000 users and 35 features
- **Contains**: User behavior features (session duration, clicks, purchases, device type, traffic source, cluster labels, etc.)

### **Cell 1: Create Target Labels**
- **Purpose**: Creates the "answers" (target variables) for training
- **Creates 3 targets**:
  - `y_bounce`: 1 if user bounced (bounce_rate > 0), 0 otherwise
    - **Distribution**: 7% bounce rate (348/5000) - Highly imbalanced
  - `y_convert`: 1 if user purchased above median, 0 otherwise
    - **Distribution**: 20.8% conversion (1040/5000) - Moderately imbalanced
  - `y_return`: 1 if user had multiple sessions, 0 otherwise
    - **Distribution**: 96.3% return rate (4816/5000) - Highly imbalanced
- **Output**: Saves `user_targets.csv` with user_id and all 3 targets
- **Why important**: These are the "ground truth" labels the models learn from

### **Cell 2: Train/Test Split**
- **Purpose**: Splits data into training (80%) and testing (20%) sets
- **Critical Fix**: Uses the **same split** for all 3 targets (ensures consistency)
- **Split**: 4000 training samples, 1000 test samples
- **Features Used**: 11 features (excludes `total_purchases` and `bounce_rate` to prevent data leakage)
- **Output**: 
  - `X_train`, `X_test` (features)
  - `y_bounce_train/test`, `y_convert_train/test`, `y_return_train/test` (targets)
- **Why important**: Tests model on unseen data to measure real performance

### **Cell 3: Logistic Regression (Conversion)**
- **Purpose**: Simple linear model to predict conversions
- **Model**: Logistic Regression with `class_weight='balanced'`
- **Performance**: 
  - **Accuracy**: 93.70% ✅
  - **ROC-AUC**: 0.9270 ✅ (Excellent)
  - **Precision**: 0.86 (class 1), 0.96 (class 0)
  - **Recall**: 0.83 (class 1), 0.96 (class 0)
  - **F1-Score**: 0.85 (class 1), 0.96 (class 0)
- **Confusion Matrix**: 764 TN, 28 FP, 35 FN, 173 TP
- **Output**: Trained model + feature importance (coefficients) + ROC curve
- **Use case**: Baseline model, interpretable results, good performance

### **Cell 4: Random Forest (Conversion)**
- **Purpose**: Ensemble tree model for conversion prediction
- **Model**: Random Forest with regularization (max_depth=10, min_samples_split=10)
- **Performance**: 
  - **Accuracy**: 93.20% ✅
  - **ROC-AUC**: 0.9731 ✅✅ (Outstanding!)
  - **Precision**: 0.86 (class 1), 0.95 (class 0)
  - **Recall**: 0.80 (class 1), 0.97 (class 0)
  - **F1-Score**: 0.83 (class 1), 0.96 (class 0)
- **Confusion Matrix**: 765 TN, 27 FP, 41 FN, 167 TP
- **Top Feature**: `total_add_to_cart` (53.8% importance) ⭐
- **Output**: Trained model + feature importances + ROC curve
- **Use case**: **Best overall performance** for conversion prediction

### **Cell 5: XGBoost (Conversion)**
- **Purpose**: Advanced gradient boosting model
- **Model**: XGBoost with `scale_pos_weight` for class imbalance
- **Performance**: 
  - **Accuracy**: 92.60% ✅
  - **ROC-AUC**: 0.9716 ✅✅ (Outstanding!)
  - **Precision**: 0.76 (class 1), 0.98 (class 0)
  - **Recall**: 0.94 (class 1), 0.92 (class 0) ⭐ **Highest recall!**
  - **F1-Score**: 0.84 (class 1), 0.95 (class 0)
- **Confusion Matrix**: 731 TN, 61 FP, 13 FN, 195 TP
- **Top Feature**: `total_add_to_cart` (81.9% importance) ⭐
- **Output**: Trained model + feature importances + ROC curve
- **Use case**: **High recall (94%)** - catches most conversions, fewer false negatives

### **Cell 6: Random Forest (Bounce) with Threshold Tuning**
- **Purpose**: Predicts if users will bounce (leave immediately)
- **Challenge**: Highly imbalanced (only 7% bounce rate)
- **Model**: Random Forest with `class_weight='balanced'` and threshold tuning
- **Baseline Performance** (threshold=0.5):
  - **Accuracy**: 80.50%
  - **ROC-AUC**: 0.8282
  - **Precision**: 21.92% (many false positives)
  - **Recall**: 66.67%
  - **F1-Score**: 0.3299
- **Optimized Performance** (threshold=0.5327):
  - **Accuracy**: 82.80% (+2.3%)
  - **ROC-AUC**: 0.8282 (unchanged)
  - **Precision**: 23.96% (+2.04% improvement) ✅
  - **Recall**: 63.89% (-2.78%)
  - **F1-Score**: 0.3485 (+1.86% improvement) ✅
- **Confusion Matrix**: 782 TN, 146 FP, 26 FN, 46 TP
- **Optimal Threshold**: 0.5327 (not 0.5!) ⭐
- **Top Features**: `avg_pages_per_session` (42.9%), `avg_events_per_session` (23.1%)
- **Output**: 
  - Trained model
  - Optimal threshold value (0.5327)
  - Comparison metrics (baseline vs optimized)
  - Visualizations (ROC, Precision-Recall, F1 vs Threshold, Feature Importance)
- **Use case**: Early warning system to identify users at risk of bouncing

---

## 💾 **Output Files Generated**

### **1. Main Output File** ⭐
**File**: `data/processed/user_targets.csv`

**Contains**:
- `user_id` - User identifier
- `y_bounce` - Bounce label (1 = bounced, 0 = did not bounce)
- `y_convert` - Conversion label (1 = converted, 0 = did not convert)
- `y_return` - Return label (1 = returned, 0 = did not return)

**Shape**: (5000, 4) - 5000 users with 4 columns

**Usage**: 
- Join with user data for analysis
- Create dashboards
- Business intelligence reports

### **2. Trained Models** (In Memory - Need to Save Manually)
The notebook creates trained models but does NOT save them automatically. You need to add code to save them (see "How to Use Later" section).

**Models Available**:
- `logreg` - Logistic Regression for conversion
- `rf_standard` - Random Forest for conversion ⭐ **Best overall**
- `xgb_model` - XGBoost for conversion ⭐ **Highest recall**
- `rf_bounce` - Random Forest for bounce (with optimal threshold)

---

## 📊 **Key Results Summary**

### **Conversion Prediction Models**

| Model | Accuracy | ROC-AUC | Precision (Class 1) | Recall (Class 1) | Best For |
|-------|----------|---------|---------------------|------------------|----------|
| **Logistic Regression** | 93.70% | 0.9270 | 0.86 | 0.83 | Interpretability |
| **Random Forest** ⭐ | 93.20% | **0.9731** | 0.86 | 0.80 | **Best overall** |
| **XGBoost** ⭐ | 92.60% | **0.9716** | 0.76 | **0.94** | **High recall** |

### **Bounce Prediction Model**

| Metric | Baseline (0.5) | Optimized (0.5327) | Improvement |
|--------|----------------|-------------------|-------------|
| **Accuracy** | 80.50% | 82.80% | +2.3% ✅ |
| **Precision** | 21.92% | 23.96% | +2.04% ✅ |
| **Recall** | 66.67% | 63.89% | -2.78% |
| **F1-Score** | 0.3299 | 0.3485 | +1.86% ✅ |
| **ROC-AUC** | 0.8282 | 0.8282 | - |

### **Top Feature Insights**

#### **Conversion Prediction**
1. **`total_add_to_cart`** - Most important feature (53-82% importance)
   - **Insight**: Users who add items to cart are highly likely to convert
   - **Action**: Focus on cart abandonment recovery

2. **`avg_session_duration`** - Second most important (7-29% importance)
   - **Insight**: Longer sessions indicate higher conversion probability
   - **Action**: Improve engagement to increase session duration

#### **Bounce Prediction**
1. **`avg_pages_per_session`** - Most important (42.9% importance)
   - **Insight**: Users viewing fewer pages are more likely to bounce
   - **Action**: Improve navigation and content discovery

2. **`avg_events_per_session`** - Second most important (23.1% importance)
   - **Insight**: Low engagement (few events) indicates bounce risk
   - **Action**: Increase interactive elements and engagement

---

## 🚀 **How to Use These Results**

### **1. Save Models for Later Use**

Add this code at the end of each model cell:

```python
import pickle
from pathlib import Path

# Create models directory
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Save Random Forest conversion model (BEST OVERALL)
with open(MODELS_DIR / "rf_standard_conversion.pkl", 'wb') as f:
    pickle.dump(rf_standard, f)

# Save XGBoost conversion model (HIGH RECALL)
with open(MODELS_DIR / "xgb_conversion.pkl", 'wb') as f:
    pickle.dump(xgb_model, f)

# Save Bounce model with optimal threshold
bounce_model_data = {
    'model': rf_bounce,
    'optimal_threshold': optimal_threshold,  # 0.5327
    'feature_cols': list(X_train_b.columns)
}
with open(MODELS_DIR / "rf_bounce.pkl", 'wb') as f:
    pickle.dump(bounce_model_data, f)

# Save Logistic Regression (for interpretability)
with open(MODELS_DIR / "logreg_conversion.pkl", 'wb') as f:
    pickle.dump(logreg, f)

print("✓ All models saved successfully!")
```

### **2. Make Predictions on New Users**

```python
import pandas as pd
import pickle
from pathlib import Path

# Load saved model
MODELS_DIR = Path("models")
with open(MODELS_DIR / "rf_standard_conversion.pkl", 'rb') as f:
    model = pickle.load(f)

# Load new user data (must have same features)
new_users = pd.read_csv("data/new_users.csv")

# Select same features used in training
feature_cols = [
    'avg_session_duration', 'avg_events_per_session', 'avg_pages_per_session',
    'total_clicks', 'total_scrolls', 'total_add_to_cart',
    'device_type_mobile', 'device_type_tablet',
    'traffic_source_direct', 'traffic_source_organic', 'traffic_source_referral'
]

X_new = new_users[feature_cols]

# Make predictions
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)[:, 1]

# Add to dataframe
new_users['predicted_conversion'] = predictions
new_users['conversion_probability'] = probabilities

# Save results
new_users.to_csv("data/predictions.csv", index=False)

print(f"Predictions made for {len(new_users)} users")
print(f"High probability users (>0.7): {(probabilities > 0.7).sum()}")
```

### **3. Real-Time User Scoring**

```python
def score_user(user_features_dict):
    """
    Score a user in real-time as they browse
    
    Args:
        user_features_dict: Dictionary with user features
    
    Returns:
        Dictionary with predictions and recommendations
    """
    import pandas as pd
    
    # Convert to DataFrame
    df = pd.DataFrame([user_features_dict])
    
    # Select features
    feature_cols = [
        'avg_session_duration', 'avg_events_per_session', 'avg_pages_per_session',
        'total_clicks', 'total_scrolls', 'total_add_to_cart',
        'device_type_mobile', 'device_type_tablet',
        'traffic_source_direct', 'traffic_source_organic', 'traffic_source_referral'
    ]
    X = df[feature_cols]
    
    # Predict conversion (using Random Forest - best overall)
    conversion_prob = rf_standard.predict_proba(X)[0, 1]
    conversion_pred = rf_standard.predict(X)[0]
    
    # Predict bounce (using optimal threshold)
    bounce_prob = rf_bounce.predict_proba(X)[0, 1]
    bounce_pred = 1 if bounce_prob >= optimal_threshold else 0
    
    # Recommendations
    if conversion_prob > 0.7:
        recommendation = "high_priority"
        action = "Show premium products, checkout button prominently"
    elif conversion_prob > 0.5:
        recommendation = "medium_priority"
        action = "Show product recommendations"
    elif bounce_pred == 1:
        recommendation = "at_risk"
        action = "Show exit-intent popup, special offers"
    else:
        recommendation = "normal"
        action = "Standard experience"
    
    return {
        'conversion_probability': float(conversion_prob),
        'will_convert': bool(conversion_pred),
        'bounce_probability': float(bounce_prob),
        'will_bounce': bool(bounce_pred),
        'recommendation': recommendation,
        'suggested_action': action
    }

# Example usage
user_data = {
    'avg_session_duration': 120,
    'avg_events_per_session': 25,
    'avg_pages_per_session': 8,
    'total_clicks': 15,
    'total_scrolls': 50,
    'total_add_to_cart': 2,  # They added items!
    'device_type_mobile': 1,
    'device_type_tablet': 0,
    'traffic_source_direct': 0,
    'traffic_source_organic': 1,
    'traffic_source_referral': 0
}

score = score_user(user_data)
print(score)
# Output: {'conversion_probability': 0.85, 'will_convert': True, ...}
```

### **4. Marketing Campaign Targeting**

```python
# Load user data with predictions
df = pd.read_csv("data/processed/user_segments_final.csv")
targets = pd.read_csv("data/processed/user_targets.csv")

# Merge
df = df.merge(targets, on='user_id', how='left')

# Make predictions for all users
feature_cols = [
    'avg_session_duration', 'avg_events_per_session', 'avg_pages_per_session',
    'total_clicks', 'total_scrolls', 'total_add_to_cart',
    'device_type_mobile', 'device_type_tablet',
    'traffic_source_direct', 'traffic_source_organic', 'traffic_source_referral'
]

X_all = df[feature_cols]
df['conversion_probability'] = rf_standard.predict_proba(X_all)[:, 1]
df['bounce_probability'] = rf_bounce.predict_proba(X_all)[:, 1]
df['will_bounce'] = (df['bounce_probability'] >= optimal_threshold).astype(int)

# Segment users for marketing
high_value = df[df['conversion_probability'] > 0.7]
at_risk = df[df['will_bounce'] == 1]
medium_value = df[(df['conversion_probability'] > 0.5) & (df['conversion_probability'] <= 0.7)]

print(f"High-value users (send premium offers): {len(high_value)}")
print(f"At-risk users (prevent bounce): {len(at_risk)}")
print(f"Medium-value users (nurture): {len(medium_value)}")

# Export for marketing campaigns
high_value[['user_id', 'conversion_probability']].to_csv(
    "data/marketing/high_value_users.csv", index=False
)
at_risk[['user_id', 'bounce_probability']].to_csv(
    "data/marketing/at_risk_users.csv", index=False
)
```

### **5. Create Prediction API**

```python
from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load models at startup
with open('models/rf_standard_conversion.pkl', 'rb') as f:
    conversion_model = pickle.load(f)

with open('models/rf_bounce.pkl', 'rb') as f:
    bounce_data = pickle.load(f)
    bounce_model = bounce_data['model']
    bounce_threshold = bounce_data['optimal_threshold']

@app.route('/predict', methods=['POST'])
def predict():
    user_data = request.json
    
    # Convert to DataFrame
    df = pd.DataFrame([user_data])
    
    # Select features
    feature_cols = [
        'avg_session_duration', 'avg_events_per_session', 'avg_pages_per_session',
        'total_clicks', 'total_scrolls', 'total_add_to_cart',
        'device_type_mobile', 'device_type_tablet',
        'traffic_source_direct', 'traffic_source_organic', 'traffic_source_referral'
    ]
    X = df[feature_cols]
    
    # Make predictions
    conversion_prob = conversion_model.predict_proba(X)[0, 1]
    bounce_prob = bounce_model.predict_proba(X)[0, 1]
    bounce_pred = 1 if bounce_prob >= bounce_threshold else 0
    
    return jsonify({
        'conversion_probability': float(conversion_prob),
        'will_convert': bool(conversion_prob > 0.5),
        'bounce_probability': float(bounce_prob),
        'will_bounce': bool(bounce_pred),
        'recommendation': 'high_priority' if conversion_prob > 0.7 else 'normal'
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

### **6. Dashboard Creation**

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load targets file
targets = pd.read_csv("data/processed/user_targets.csv")

# Conversion rate analysis
conversion_rate = targets['y_convert'].mean()
bounce_rate = targets['y_bounce'].mean()
return_rate = targets['y_return'].mean()

print(f"Overall Conversion Rate: {conversion_rate:.2%}")
print(f"Overall Bounce Rate: {bounce_rate:.2%}")
print(f"Overall Return Rate: {return_rate:.2%}")

# Join with segments for analysis
segments = pd.read_csv("data/processed/user_segments_final.csv")
merged = targets.merge(segments[['user_id', 'behavior_segment']], on='user_id')

# Conversion rate by segment
conversion_by_segment = merged.groupby('behavior_segment')['y_convert'].agg(['mean', 'count'])
print("\nConversion Rate by Segment:")
print(conversion_by_segment)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Conversion rate by segment
conversion_by_segment['mean'].plot(kind='bar', ax=axes[0])
axes[0].set_title('Conversion Rate by User Segment')
axes[0].set_ylabel('Conversion Rate')
axes[0].tick_params(axis='x', rotation=45)

# Bounce rate by segment
bounce_by_segment = merged.groupby('behavior_segment')['y_bounce'].mean()
bounce_by_segment.plot(kind='bar', ax=axes[1], color='red')
axes[1].set_title('Bounce Rate by User Segment')
axes[1].set_ylabel('Bounce Rate')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("data/dashboards/segment_analysis.png")
plt.show()
```

---

## 📈 **Business Applications**

### **1. Conversion Optimization**
- **Target high-probability users**: Focus marketing on users with >70% conversion probability
- **Cart abandonment recovery**: `total_add_to_cart` is the top predictor - focus on cart recovery
- **A/B testing**: Test different experiences for different probability ranges

### **2. Bounce Prevention**
- **Early intervention**: Identify at-risk users (bounce probability >= 0.5327)
- **Exit-intent popups**: Show special offers to users about to bounce
- **Content optimization**: Improve pages/session and events/session (top bounce predictors)

### **3. Personalization**
- **Dynamic pricing**: Offer discounts to high-conversion-probability users
- **Product recommendations**: Show different products based on conversion probability
- **Content customization**: Show different content to different user segments

### **4. Marketing ROI**
- **Budget allocation**: Focus spend on high-probability users
- **Campaign targeting**: Use predictions to target campaigns
- **Performance measurement**: Compare predicted vs actual conversions

---

## ⚠️ **Important Notes**

### **1. Model Selection**
- **For best overall performance**: Use **Random Forest** (ROC-AUC: 0.9731)
- **For catching most conversions**: Use **XGBoost** (Recall: 94%)
- **For interpretability**: Use **Logistic Regression** (coefficients explainable)

### **2. Threshold Selection**
- **Conversion models**: Use default 0.5 threshold (balanced)
- **Bounce model**: Use optimal threshold **0.5327** (not 0.5!) for better precision

### **3. Data Leakage Prevention**
- Models do NOT use `total_purchases` or `bounce_rate` as features (would be cheating!)
- Only uses behavioral features observable before the outcome

### **4. Feature Consistency**
- New data must have the **exact same features** as training data
- Missing features will cause errors
- Feature engineering must be identical

### **5. Model Retraining**
- Retrain periodically (weekly/monthly) as user behavior changes
- Monitor performance over time
- Update when accuracy drops below acceptable thresholds

---

## 🔄 **Integration with Clustering Notebook**

### **Pipeline Flow**:
```
01_synthetic_data_generator.ipynb
    ↓ (generates raw data)
02_data_preprocessing_feature_engineering.ipynb
    ↓ (creates features)
03_user_clustering.ipynb
    ↓ (creates segments) → user_segments_final.csv
04_supervised_ml_models.ipynb  ← YOU ARE HERE
    ↓ (creates predictions) → user_targets.csv
```

### **How Clustering Helps**:
- Segment labels can be used as features in ML models
- Clustering helps understand user behavior patterns
- Segments provide context for predictions

---

## 📊 **Key Metrics to Monitor**

### **Model Performance**
- **Accuracy**: Should stay above 90% for conversion models
- **ROC-AUC**: Should stay above 0.90
- **Precision/Recall**: Balance based on business needs

### **Business Metrics**
- **Conversion rate by segment**
- **Bounce rate by segment**
- **Prediction accuracy over time**

---

## 🎯 **Quick Reference**

### **Main Output Files**
- `data/processed/user_targets.csv` ⭐ - All user labels (bounce, convert, return)
- Models (need to save manually) - Trained ML models

### **Best Models**
- **Conversion**: Random Forest (ROC-AUC: 0.9731) ⭐
- **Conversion (High Recall)**: XGBoost (Recall: 94%) ⭐
- **Bounce**: Random Forest with threshold 0.5327

### **Key Numbers**
- **Optimal Bounce Threshold**: 0.5327 (not 0.5!)
- **Top Conversion Feature**: `total_add_to_cart` (53-82% importance)
- **Top Bounce Features**: `avg_pages_per_session` (42.9%), `avg_events_per_session` (23.1%)

### **Next Steps**
1. ✅ Save models to disk (add code from section 1)
2. ✅ Test on new data (section 2)
3. ✅ Create real-time scoring (section 3)
4. ✅ Build API for production (section 5)
5. ✅ Create dashboards (section 6)

---

**This notebook creates production-ready predictive models for user behavior!** 🚀

