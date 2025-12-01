# 📅 Timestamp Columns Reference

## ✅ **YES - You Have Historical Data with Timestamps!**

Your project has timestamps in multiple files, perfect for Phase 5 (comparing behaviors over time).

---

## 📊 **Timestamp Columns by File**

### **1. `sessions_raw.csv`** ⭐ **PRIMARY TIMESTAMP FILE**

**Location**: `data/raw/sessions_raw.csv`

**Timestamp Columns**:
- **`start_time`** - Session start timestamp
  - Format: `YYYY-MM-DD HH:MM:SS` (e.g., `2025-07-29 04:46:03`)
  - Type: String (converted to datetime in preprocessing)
  
- **`end_time`** - Session end timestamp
  - Format: `YYYY-MM-DD HH:MM:SS` (e.g., `2025-07-29 04:48:40`)
  - Type: String (converted to datetime in preprocessing)

**Date Range** (from settings):
- **Start Date**: `2025-06-03T10:29:01Z`
- **End Date**: `2025-11-30T10:29:01Z`
- **Span**: ~180 days (6 months)

**Additional Time Features** (extracted in preprocessing):
- `start_hour` - Hour of day (0-23)
- `start_dayofweek` - Day of week (0=Monday, 6=Sunday)
- `start_month` - Month (1-12)
- `is_weekend` - Binary (0=weekday, 1=weekend)

---

### **2. `events_raw.csv`** ⭐ **EVENT-LEVEL TIMESTAMPS**

**Location**: `data/raw/events_raw.csv`

**Timestamp Columns**:
- **`timestamp`** - Event timestamp
  - Format: `YYYY-MM-DD HH:MM:SS` (e.g., `2025-07-29 04:46:03`)
  - Type: String (converted to datetime in preprocessing)
  - **Granularity**: Individual event level (most detailed)

**Additional Time Features** (extracted in preprocessing):
- `event_hour` - Hour of day (0-23)
- `event_dayofweek` - Day of week (0-6)
- `event_month` - Month (1-12)
- `event_is_weekend` - Binary (0=weekday, 1=weekend)

---

### **3. `users_raw.csv`** - User Account Creation

**Location**: `data/raw/users_raw.csv`

**Timestamp Columns**:
- **`account_created_at`** - User account creation timestamp
  - Format: `YYYY-MM-DD HH:MM:SS`
  - Type: String (converted to datetime in preprocessing)
  - **Use Case**: Calculate user lifetime, cohort analysis

---

### **4. Processed Files with Time Features**

#### **`data/processed/sessions.csv`**
- Contains: `start_time`, `end_time` (as datetime objects)
- Plus: `start_hour`, `start_dayofweek`, `start_month`, `is_weekend`

#### **`data/processed/events.csv`**
- Contains: `timestamp` (as datetime object)
- Plus: `event_hour`, `event_dayofweek`, `event_month`, `event_is_weekend`

#### **`data/processed/user_features.csv`**
- Contains aggregated time features:
  - `first_session_date` - User's first session date
  - `last_session_date` - User's last session date
  - `days_active` - Number of days between first and last session
  - `preferred_hour` - Most common session hour
  - `weekend_session_ratio` - Ratio of weekend sessions

---

## 🔍 **How to Access Timestamps**

### **Option 1: Load Raw Files**

```python
import pandas as pd
from pathlib import Path

# Load sessions with timestamps
sessions_df = pd.read_csv("data/raw/sessions_raw.csv")
sessions_df['start_time'] = pd.to_datetime(sessions_df['start_time'])
sessions_df['end_time'] = pd.to_datetime(sessions_df['end_time'])

# Load events with timestamps
events_df = pd.read_csv("data/raw/events_raw.csv")
events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])

# Load users with account creation dates
users_df = pd.read_csv("data/raw/users_raw.csv")
users_df['account_created_at'] = pd.to_datetime(users_df['account_created_at'])
```

### **Option 2: Load Processed Files (Already Converted)**

```python
# Processed files already have datetime objects
sessions_df = pd.read_csv("data/processed/sessions.csv", 
                         parse_dates=['start_time', 'end_time'])
events_df = pd.read_csv("data/processed/events.csv",
                       parse_dates=['timestamp'])
```

---

## 📈 **Use Cases for Phase 5 (Time-Based Analysis)**

### **1. Daily/Weekly/Monthly Trends**
```python
# Group by date
sessions_df['date'] = sessions_df['start_time'].dt.date
daily_sessions = sessions_df.groupby('date').size()

# Group by week
sessions_df['week'] = sessions_df['start_time'].dt.to_period('W')
weekly_sessions = sessions_df.groupby('week').size()

# Group by month
sessions_df['month'] = sessions_df['start_time'].dt.to_period('M')
monthly_sessions = sessions_df.groupby('month').size()
```

### **2. User Behavior Over Time**
```python
# Track user behavior changes
user_timeline = sessions_df.groupby(['user_id', sessions_df['start_time'].dt.date]).agg({
    'duration_sec': 'mean',
    'num_events': 'sum',
    'session_id': 'count'
}).reset_index()
```

### **3. Cohort Analysis**
```python
# Group users by account creation month
users_df['cohort_month'] = users_df['account_created_at'].dt.to_period('M')
cohort_analysis = users_df.groupby('cohort_month').size()
```

### **4. Time-of-Day Patterns**
```python
# Analyze behavior by hour
hourly_activity = sessions_df.groupby('start_hour').agg({
    'session_id': 'count',
    'duration_sec': 'mean',
    'num_events': 'mean'
})
```

### **5. Weekend vs Weekday Comparison**
```python
# Compare weekend vs weekday behavior
weekend_comparison = sessions_df.groupby('is_weekend').agg({
    'duration_sec': 'mean',
    'num_events': 'mean',
    'session_id': 'count'
})
```

---

## 📋 **Summary Table**

| File | Timestamp Column(s) | Format | Granularity | Use Case |
|------|---------------------|--------|-------------|----------|
| **`sessions_raw.csv`** | `start_time`, `end_time` | `YYYY-MM-DD HH:MM:SS` | Session level | Session trends, time patterns |
| **`events_raw.csv`** | `timestamp` | `YYYY-MM-DD HH:MM:SS` | Event level | Detailed event timeline |
| **`users_raw.csv`** | `account_created_at` | `YYYY-MM-DD HH:MM:SS` | User level | Cohort analysis, user lifetime |
| **`sessions.csv`** (processed) | `start_time`, `end_time` | datetime object | Session level | Ready for analysis |
| **`events.csv`** (processed) | `timestamp` | datetime object | Event level | Ready for analysis |

---

## 🎯 **Key Timestamp Columns for Phase 5**

### **Primary Columns**:
1. **`sessions_raw.csv` → `start_time`** ⭐ **MOST USEFUL**
   - Best for: Daily/weekly/monthly trends, session patterns
   
2. **`events_raw.csv` → `timestamp`** ⭐ **MOST DETAILED**
   - Best for: Event-level timeline, granular analysis

3. **`users_raw.csv` → `account_created_at`**
   - Best for: Cohort analysis, user lifetime metrics

### **Derived Time Features** (already calculated):
- `start_hour`, `start_dayofweek`, `start_month`, `is_weekend`
- `event_hour`, `event_dayofweek`, `event_month`, `event_is_weekend`
- `first_session_date`, `last_session_date`, `days_active`

---

## ✅ **Answer to Your Question**

**👉 Which file has timestamps?**

1. **`sessions_raw.csv`** - Has `start_time` and `end_time` ⭐
2. **`events_raw.csv`** - Has `timestamp` ⭐
3. **`users_raw.csv`** - Has `account_created_at`

**Column Names**:
- Sessions: **`start_time`**, **`end_time`**
- Events: **`timestamp`**
- Users: **`account_created_at`**

**Date Range**: June 3, 2025 to November 30, 2025 (~180 days)

---

**You have all the timestamp data needed for Phase 5!** 🎉

