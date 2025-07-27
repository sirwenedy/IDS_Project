import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Paths
base_path = r'C:\Users\wenedy\IDS_Project\kddcup'
train_file = os.path.join(base_path, 'KDDTrain+.txt')
test_file = os.path.join(base_path, 'KDDTest-21.txt')

# Verify files
if not os.path.exists(train_file):
    print(f"Error: {train_file} not found")
    exit(1)
if not os.path.exists(test_file):
    print(f"Error: {test_file} not found. Using KDDTest+.txt")
    test_file = os.path.join(base_path, 'KDDTest+.txt')
    if not os.path.exists(test_file):
        print("Download from https://www.kaggle.com/datasets/hassan06/nslkdd")
        exit(1)

# Features (41, full NSL-KDD)
features = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
    'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
    'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
]

# Load data
try:
    train_data = pd.read_csv(train_file, names=features + ['label', 'difficulty'])
    test_data = pd.read_csv(test_file, names=features + ['label', 'difficulty'])
except Exception as e:
    print(f"Error loading files: {e}")
    exit(1)

# Drop constant features
train_data = train_data.drop(['num_outbound_cmds', 'is_host_login'], axis=1)
test_data = test_data.drop(['num_outbound_cmds', 'is_host_login'], axis=1)
features = [f for f in features if f not in ['num_outbound_cmds', 'is_host_login']]

# Filter DoS and normal
dos_labels = ['neptune', 'smurf', 'back', 'land', 'pod', 'teardrop']
train_data = train_data[train_data['label'].isin(dos_labels + ['normal'])]
test_data = test_data[test_data['label'].isin(dos_labels + ['normal'])]

# Encode labels
train_data['label'] = train_data['label'].apply(lambda x: 1 if x in dos_labels else 0)
test_data['label'] = test_data['label'].apply(lambda x: 1 if x in dos_labels else 0)

# Check for NaN in labels
if train_data['label'].isna().any() or test_data['label'].isna().any():
    print("Error: NaN found in labels")
    exit(1)

# Preprocessing
categorical = ['protocol_type', 'service', 'flag']
numerical = [f for f in features if f not in categorical]

# Manual encoding
for col in categorical:
    mapping = {val: idx for idx, val in enumerate(train_data[col].unique())}
    train_data[col] = train_data[col].map(mapping)
    test_data[col] = test_data[col].map(mapping).fillna(-1)

# Check for NaN in features
if train_data[numerical].isna().any().any() or test_data[numerical].isna().any().any():
    print("Error: NaN found in numerical features")
    train_data[numerical] = train_data[numerical].fillna(0)
    test_data[numerical] = test_data[numerical].fillna(0)

# Scale numerical features
scaler = StandardScaler()
train_data[numerical] = scaler.fit_transform(train_data[numerical])
test_data[numerical] = scaler.transform(test_data[numerical])

# Prepare data
X_train = train_data.drop(['label', 'difficulty'], axis=1)
y_train = train_data['label']
X_test = test_data.drop(['label', 'difficulty'], axis=1)
y_test = test_data['label']

print("Training Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

# Train Random Forest
model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("Model Trained Successfully")

# Cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)
print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Test
y_pred = model.predict(X_test)

# Metrics
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Normal', 'DoS']))
tn, fp, fn, tp = cm.ravel()
detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
print(f"Detection Rate: {detection_rate:.4f}")
print(f"FPR: {fpr:.4f}")

# Feature Importance Plot
importances = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importances_})
top_features = importances.sort_values('importance', ascending=False).head(10)
plt.figure(figsize=(8, 5))
sns.barplot(x='importance', y='feature', data=top_features)
plt.title('Top 10 Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.savefig(os.path.join(base_path, 'feature_importance.png'))
plt.close()

# Save results
with open(os.path.join(base_path, 'dos_detection_results.txt'), 'w') as f:
    f.write(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n\n")
    f.write("Confusion Matrix:\n" + str(cm) + "\n\n")
    f.write("Classification Report:\n" + classification_report(y_test, y_pred, target_names=['Normal', 'DoS']))
    f.write(f"\nDetection Rate: {detection_rate:.4f}\nFPR: {fpr:.4f}\n")

print("\nTop 5 Features:\n", top_features.head(5))