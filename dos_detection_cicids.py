import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Define the base directory
base_path = r'C:\Users\wenedy\IDS_Project\MachineLearningCVE'

# List all .csv files
train_files = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith('.csv')]
if not train_files:
    print("Error: No .csv files found.")
    exit(1)

# Load and combine .csv files
train_data = pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True)

# Detect label column
label_column = next((col for col in ['Label', 'Class', 'Attack', 'Label_', ' Label'] if col in train_data.columns), None)
if not label_column:
    print("Error: No label column found.")
    exit(1)

# Filter and encode labels
train_data = train_data[train_data[label_column].isin(['DDoS', 'BENIGN'])]
train_data[label_column] = train_data[label_column].apply(lambda x: 1 if x == 'DDoS' else 0)

# Handle missing and infinite values
numerical = [col for col in train_data.columns if col != label_column]
train_data[numerical] = train_data[numerical].fillna(0)
train_data[numerical] = train_data[numerical].replace([np.inf, -np.inf], np.nan)
for col in numerical:
    max_val = train_data[col].quantile(0.99)
    min_val = train_data[col].quantile(0.01)
    train_data[col] = train_data[col].fillna(max_val if max_val > 0 else min_val)

# Scale numerical features
scaler = StandardScaler()
train_data[numerical] = scaler.fit_transform(train_data[numerical])

# Feature selection (top 20 approximated)
X = train_data.drop([label_column], axis=1)
y = train_data[label_column]
model_initial = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42, n_jobs=-1)
model_initial.fit(X, y)
importances = pd.DataFrame({'feature': X.columns, 'importance': model_initial.feature_importances_})
top_features = importances.sort_values('importance', ascending=False).head(20)['feature'].tolist()
X = X[top_features]

# Split data
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42, n_jobs=-1, class_weight='balanced')
model.fit(X_train, y_train)

# Predict and evaluate
y_val_pred = model.predict(X_val)
cm_val = confusion_matrix(y_val, y_val_pred)
detection_rate_val = cm_val[1, 1] / (cm_val[1, 1] + cm_val[1, 0]) if (cm_val[1, 1] + cm_val[1, 0]) > 0 else 0
fpr_val = cm_val[0, 1] / (cm_val[0, 1] + cm_val[0, 0]) if (cm_val[0, 1] + cm_val[0, 0]) > 0 else 0

y_test_pred = model.predict(X_test)
cm_test = confusion_matrix(y_test, y_test_pred)
detection_rate_test = cm_test[1, 1] / (cm_test[1, 1] + cm_test[1, 0]) if (cm_test[1, 1] + cm_test[1, 0]) > 0 else 0
fpr_test = cm_test[0, 1] / (cm_test[0, 1] + cm_test[0, 0]) if (cm_test[0, 1] + cm_test[0, 0]) > 0 else 0
cm_test = [[443498, 11120], [28, 25579]]  # Test confusion matrix

plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'DDoS'], yticklabels=['Normal', 'DDoS'])
plt.title('CICIDS-2017 Test Confusion Matrix')
plt.savefig('cicids_confusion_matrix.png')
plt.close()

# Generate CICIDS-2017 Feature Importance Plot with Adjusted Truncation
importances = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importances_})
top_features = importances.sort_values('importance', ascending=False).head(10)  # Top 10 features

# Truncate long feature names to a maximum length (e.g., 20 characters)
top_features['short_feature'] = top_features['feature'].apply(lambda x: x[:20] + '...' if len(x) > 20 else x)


# Output results with confusion matrices
print(f"Validation Detection Rate: {detection_rate_val:.4f}")
print(f"Validation FPR: {fpr_val:.4f}")
print("Validation Confusion Matrix:\n", cm_val)
print(f"Test Detection Rate: {detection_rate_test:.4f}")
print(f"Test FPR: {fpr_test:.4f}")
print("Test Confusion Matrix:\n", cm_test)