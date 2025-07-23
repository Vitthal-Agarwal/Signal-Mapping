import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load the CSV data
data = pd.read_csv("synthetic_user_behavior_log.csv")

# Helper function to calculate time gaps
def calculate_time_gap(timestamps):
    if pd.isna(timestamps) or timestamps == "" or str(timestamps).strip() == "":
        return 0
    try:
        times = [datetime.fromisoformat(t.strip()) for t in str(timestamps).split(",")]
        if len(times) > 1:
            gaps = [(times[i] - times[i - 1]).days for i in range(1, len(times))]
            return max(gaps, default=0)
        return 0
    except (ValueError, AttributeError):
        return 0

# Function to calculate calendar void days - FIXED VERSION
def calculate_calendar_void_days(voids):
    if pd.isna(voids) or voids == "" or str(voids).strip() == "":
        return 0
    total_days = 0
    try:
        # Split on the pattern "YYYY-MM-DDTHH:MM:SS:YYYY-MM-DDTHH:MM:SS"
        void_str = str(voids).strip()
        # Find the pattern with date:date
        parts = void_str.split(':')
        if len(parts) >= 6:  # Should have at least 6 parts for two datetime stamps
            # Reconstruct the start and end dates
            start_date_str = ':'.join(parts[:3])  # First 3 parts for start date
            end_date_str = ':'.join(parts[3:6])   # Next 3 parts for end date
            
            start_date = datetime.fromisoformat(start_date_str)
            end_date = datetime.fromisoformat(end_date_str)
            total_days = (end_date - start_date).days
    except (ValueError, IndexError) as e:
        print(f"Error processing void '{voids}': {e}")
        return 0
    return max(total_days, 0)

# Enhanced feature engineering
def calculate_avg_edit_gap(timestamps):
    if pd.isna(timestamps) or timestamps == "" or str(timestamps).strip() == "":
        return 0
    try:
        times = [datetime.fromisoformat(t.strip()) for t in str(timestamps).split(",")]
        if len(times) > 1:
            gaps = [(times[i] - times[i - 1]).days for i in range(1, len(times))]
            return np.mean(gaps)
        return 0
    except (ValueError, AttributeError):
        return 0

def calculate_edit_consistency(timestamps):
    """Calculate the standard deviation of edit gaps (lower = more consistent)"""
    if pd.isna(timestamps) or timestamps == "" or str(timestamps).strip() == "":
        return 0
    try:
        times = [datetime.fromisoformat(t.strip()) for t in str(timestamps).split(",")]
        if len(times) > 2:
            gaps = [(times[i] - times[i - 1]).days for i in range(1, len(times))]
            return np.std(gaps)
        return 0
    except (ValueError, AttributeError):
        return 0

def calculate_meeting_frequency_recent(timestamps, days_back=7):
    """Calculate meeting frequency in recent days"""
    if pd.isna(timestamps) or timestamps == "" or str(timestamps).strip() == "":
        return 0
    try:
        times = [datetime.fromisoformat(t.strip()) for t in str(timestamps).split(",")]
        cutoff_date = datetime.now() - timedelta(days=days_back)
        recent_meetings = [t for t in times if t >= cutoff_date]
        return len(recent_meetings)
    except (ValueError, AttributeError):
        return 0

# Apply feature engineering
print("Calculating features...")
data["edit_gap_days"] = data["message_edit_timestamps"].apply(calculate_time_gap)
data["meeting_frequency"] = data["meeting_timestamps"].apply(
    lambda s: len(str(s).split(",")) if pd.notna(s) and str(s).strip() != "" else 0
)
data["calendar_void_days"] = data["calendar_voids"].apply(calculate_calendar_void_days)

# Additional advanced features
data["avg_edit_gap"] = data["message_edit_timestamps"].apply(calculate_avg_edit_gap)
data["edit_consistency"] = data["message_edit_timestamps"].apply(calculate_edit_consistency)
data["recent_meeting_frequency"] = data["meeting_timestamps"].apply(
    lambda s: calculate_meeting_frequency_recent(s, 14)
)

# Calculate engagement ratios and composite scores with safe division
data["void_ratio"] = data["calendar_void_days"] / (data["calendar_void_days"] + data["meeting_frequency"] + 1)
data["edit_to_meeting_ratio"] = data["edit_gap_days"] / (data["meeting_frequency"] + 1)
data["activity_score"] = data["meeting_frequency"] + (1 / (data["edit_gap_days"] + 1))

# Handle any potential infinity or NaN values
data = data.replace([np.inf, -np.inf], np.nan)
data = data.fillna(0)

# Define enhanced disengagement risk with multiple criteria
risk_weights = {
    "edit_gap_flag": 0.25,
    "low_meeting_flag": 0.30,
    "calendar_void_flag": 0.25,
    "consistency_flag": 0.20,
}

# Enhanced risk flags
data["edit_gap_flag"] = (data["edit_gap_days"] > 5).astype(int)
data["low_meeting_flag"] = (data["meeting_frequency"] < 2).astype(int)
data["calendar_void_flag"] = (data["calendar_void_days"] > 7).astype(int)  # More stringent
data["consistency_flag"] = (data["edit_consistency"] > 3).astype(int)  # High inconsistency

# Calculate composite risk score
data["disengagement_risk"] = (
    data["edit_gap_flag"] * risk_weights["edit_gap_flag"]
    + data["low_meeting_flag"] * risk_weights["low_meeting_flag"]
    + data["calendar_void_flag"] * risk_weights["calendar_void_flag"]
    + data["consistency_flag"] * risk_weights["consistency_flag"]
)

# Features for ML model
feature_columns = [
    "edit_gap_days",
    "meeting_frequency", 
    "calendar_void_days",
    "avg_edit_gap",
    "edit_consistency",
    "recent_meeting_frequency",
    "void_ratio",
    "edit_to_meeting_ratio",
    "activity_score"
]

X = data[feature_columns]
y = (data["disengagement_risk"] > 0.5).astype(int)

print(f"Dataset shape: {X.shape}")
print(f"Positive cases (disengaged): {y.sum()}/{len(y)} ({y.mean():.2%})")

# Handle any remaining NaN values and infinite values
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

# Check for any remaining problematic values
print(f"Checking data quality:")
print(f"NaN values: {X.isnull().sum().sum()}")
print(f"Infinite values: {np.isinf(X.values).sum()}")
print(f"Data range: {X.min().min():.3f} to {X.max().max():.3f}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if y.sum() > 0 else None
)

# Scale features for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train multiple models
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_split=10),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=5),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
    'SVM': SVC(random_state=42, probability=True)
}

results = {}
best_model = None
best_score = 0

print("\nModel Performance Comparison:")
print("-" * 50)

for name, model in models.items():
    if name == 'SVM':
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        pred_proba = model.predict_proba(X_test_scaled)[:, 1] if len(set(y_test)) > 1 else None
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        pred_proba = model.predict_proba(X_test)[:, 1] if len(set(y_test)) > 1 else None
    
    accuracy = accuracy_score(y_test, preds)
    
    results[name] = {
        'model': model,
        'predictions': preds,
        'probabilities': pred_proba,
        'accuracy': accuracy
    }
    
    print(f"{name:15} Accuracy: {accuracy:.3f}")
    
    if accuracy > best_score:
        best_score = accuracy
        best_model = name

print(f"\nBest Model: {best_model} (Accuracy: {best_score:.3f})")

# Detailed analysis of best model
best_model_obj = results[best_model]['model']
best_preds = results[best_model]['predictions']

print(f"\nDetailed Results for {best_model}:")
print("=" * 50)
print(f"Accuracy: {accuracy_score(y_test, best_preds):.3f}")

if len(set(y_test)) > 1:
    print(f"ROC AUC: {roc_auc_score(y_test, results[best_model]['probabilities']):.3f}")

print("\nClassification Report:")
print(classification_report(y_test, best_preds, zero_division=0))

# Feature importance for tree-based models
if best_model in ['Decision Tree', 'Random Forest', 'Gradient Boosting']:
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': best_model_obj.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title(f'Feature Importance - {best_model}')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()
    
    print("\nTop 5 Most Important Features:")
    for idx, row in feature_importance.head().iterrows():
        print(f"{row['feature']:20} {row['importance']:.3f}")

# Confusion Matrix
if len(set(y_test)) > 1:
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, best_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=['Engaged', 'Disengaged'],
                yticklabels=['Engaged', 'Disengaged'])
    plt.title(f'Confusion Matrix - {best_model}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# ROC Curve
if len(set(y_test)) > 1 and results[best_model]['probabilities'] is not None:
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, results[best_model]['probabilities'])
    plt.plot(fpr, tpr, linewidth=2, label=f'{best_model} (AUC = {roc_auc_score(y_test, results[best_model]["probabilities"]):.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Risk distribution analysis
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.hist(data['disengagement_risk'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Distribution of Disengagement Risk Scores')
plt.xlabel('Risk Score')
plt.ylabel('Frequency')

plt.subplot(2, 2, 2)
risk_summary = data.groupby('disengagement_risk').size()
plt.bar(risk_summary.index, risk_summary.values, alpha=0.7, color='lightcoral')
plt.title('Users by Risk Score')
plt.xlabel('Risk Score')
plt.ylabel('Number of Users')

plt.subplot(2, 2, 3)
high_risk_users = data[data['disengagement_risk'] > 0.5]
if len(high_risk_users) > 0:
    feature_means = high_risk_users[feature_columns].mean()
    plt.barh(range(len(feature_means)), feature_means.values, alpha=0.7, color='orange')
    plt.yticks(range(len(feature_means)), feature_means.index, rotation=0, fontsize=8)
    plt.title('Average Feature Values for High-Risk Users')
    plt.xlabel('Average Value')

plt.subplot(2, 2, 4)
correlation_matrix = data[feature_columns + ['disengagement_risk']].corr()
risk_correlations = correlation_matrix['disengagement_risk'].drop('disengagement_risk').sort_values(key=abs, ascending=False)
colors = ['red' if x < 0 else 'green' for x in risk_correlations.values]
plt.barh(range(len(risk_correlations)), risk_correlations.values, color=colors, alpha=0.7)
plt.yticks(range(len(risk_correlations)), risk_correlations.index, fontsize=8)
plt.title('Feature Correlation with Risk Score')
plt.xlabel('Correlation')

plt.tight_layout()
plt.show()

# Save results and model insights
results_summary = {
    'total_users': len(data),
    'high_risk_users': len(data[data['disengagement_risk'] > 0.5]),
    'model_performance': {name: res['accuracy'] for name, res in results.items()},
    'best_model': best_model,
    'feature_importance': feature_importance.to_dict('records') if best_model in ['Decision Tree', 'Random Forest', 'Gradient Boosting'] else None
}

print(f"\nSummary:")
print(f"Total users analyzed: {results_summary['total_users']}")
print(f"High-risk users identified: {results_summary['high_risk_users']} ({results_summary['high_risk_users']/results_summary['total_users']:.1%})")
print(f"Best performing model: {results_summary['best_model']}")
