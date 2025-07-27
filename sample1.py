import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Paths
base_path = r'C:\Users\wenedy\IDS_Project\kddcup'
train_file = os.path.join(base_path, 'KDDTrain+.txt')
test_file = os.path.join(base_path, 'KDDTest-21.txt')  # Fallback: KDDTest+.txt

# Verify files
if not os.path.exists(train_file):
    print(f"Error: {train_file} not found")
    exit(1)
if not os.path.exists(test_file):
    print(f"Error: {test_file} not found. Using KDDTest+.txt instead")
    test_file = os.path.join(base_path, 'KDDTest+.txt')
    if not os.path.exists(test_file):
        print(f"Error: {test_file} not found. Download from https://www.kaggle.com/datasets/hassan06/nslkdd")
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

# Inspect raw data
print("Raw First Line (Train):", train_data.iloc[0].values)

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

# Simple preprocessing
categorical = ['protocol_type', 'service', 'flag']
numerical = [f for f in features if f not in categorical]

# Manual encoding (simplified)
for col in categorical:
    train_data[col] = train_data[col].astype('category').cat.codes
    test_data[col] = test_data[col].astype('category').cat.codes

# Scale numerical features
scaler = StandardScaler()
train_data[numerical] = scaler.fit_transform(train_data[numerical])
test_data[numerical] = scaler.transform(test_data[numerical])

# Prepare data
X_train = train_data.drop(['label', 'difficulty'], axis=1)
y_train = train_data['label']
X_test = test_data.drop(['label', 'difficulty'], axis=1)
y_test = test_data['label']

print("Training Shape:", X_train.shape)  # (~113270, 39)
print("Test Shape:", X_test.shape)      # (~11850, 39)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model Trained Successfully")

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

# Visualize
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'DoS'], yticklabels=['Normal', 'DoS'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(base_path, 'confusion_matrix.png'))

# Save results
with open(os.path.join(base_path, 'dos_detection_results.txt'), 'w') as f:
    f.write("Confusion Matrix:\n" + str(cm) + "\n\n")
    f.write("Classification Report:\n" + classification_report(y_test, y_pred, target_names=['Normal', 'DoS']))
    f.write(f"\nDetection Rate: {detection_rate:.4f}\nFPR: {fpr:.4f}\n")

# Feature Importance
importances = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importances_})
print("\nTop 5 Features:\n", importances.sort_values('importance', ascending=False).head(5))