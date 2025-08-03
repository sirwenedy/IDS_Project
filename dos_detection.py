import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler  # This is for standardizing numerical features
from sklearn.ensemble import RandomForestClassifier  # This is the core model for anomaly detection (RF)
from sklearn.metrics import classification_report, confusion_matrix  # Evaluation metrics
from sklearn.model_selection import cross_val_score  # This is for model validation
import seaborn as sns  # This is for enhanced visualization
import matplotlib.pyplot as plt  # This is for plotting feature importance
import os  # This is for file and directory handling

# To define the base directory and file paths for NSL-KDD dataset
base_path = r'C:\Users\wenedy\IDS_Project\kddcup'  # The root directory for project files
train_file = os.path.join(base_path, 'KDDTrain+.txt')  # The path to training dataset
test_file = os.path.join(base_path, 'KDDTest-21.txt')  # The path to test dataset

# To verify the existence of dataset files to prevent runtime errors
if not os.path.exists(train_file):
    print(f"Error: {train_file} not found")  # This is to notify user if the training file is missing
    exit(1)  # Exit with error code to halt execution
if not os.path.exists(test_file):
    print(f"Error: {test_file} not found. Using KDDTest+.txt")  # Fallback message
    test_file = os.path.join(base_path, 'KDDTest+.txt')  # Switch to alternative test file
    if not os.path.exists(test_file):
        print("Download from https://www.kaggle.com/datasets/hassan06/nslkdd")  # Guide user to dataset source
        exit(1)  # Exit if fallback also fails

# To define the 41 feature names of the NSL-KDD dataset for column assignment
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

# Load dataset into Pandas DataFrames with custom column names
try:
    train_data = pd.read_csv(train_file, names=features + ['label', 'difficulty'])  # Load training data
    test_data = pd.read_csv(test_file, names=features + ['label', 'difficulty'])  # Load test data
except Exception as e:
    print(f"Error loading files: {e}")  # Report any file reading issues
    exit(1)  # Exit on failure to ensure data integrity

# Drop features with constant or near-constant values to improve model efficiency
train_data = train_data.drop(['num_outbound_cmds', 'is_host_login'], axis=1)  # Remove from training set
test_data = test_data.drop(['num_outbound_cmds', 'is_host_login'], axis=1)  # Remove from test set
features = [f for f in features if f not in ['num_outbound_cmds', 'is_host_login']]  # Update feature list

# Filter dataset to focus only on DoS attacks and normal traffic for binary classification
dos_labels = ['neptune', 'smurf', 'back', 'land', 'pod', 'teardrop']  # List of DoS attack types
train_data = train_data[train_data['label'].isin(dos_labels + ['normal'])]  # Filter training data
test_data = test_data[test_data['label'].isin(dos_labels + ['normal'])]  # Filter test data

# Encode labels into binary format (1 for DoS, 0 for normal) for model input
train_data['label'] = train_data['label'].apply(lambda x: 1 if x in dos_labels else 0)  # Apply to training labels
test_data['label'] = test_data['label'].apply(lambda x: 1 if x in dos_labels else 0)  # Apply to test labels

# Check for missing values in labels to ensure data quality
if train_data['label'].isna().any() or test_data['label'].isna().any():
    print("Error: NaN found in labels")  # Warn user of missing labels
    exit(1)  # Exit to prevent invalid training

# Identify categorical and numerical features for separate preprocessing
categorical = ['protocol_type', 'service', 'flag']  # Categorical features with discrete values
numerical = [f for f in features if f not in categorical]  # All other features are numerical

# Manually encode categorical features into numerical indices
for col in categorical:
    mapping = {val: idx for idx, val in enumerate(train_data[col].unique())}  # Create mapping from unique values
    train_data[col] = train_data[col].map(mapping)  # Encode training data
    test_data[col] = test_data[col].map(mapping).fillna(-1)  # Encode test data, use -1 for unseen values

# Check and handle missing values in numerical features
if train_data[numerical].isna().any().any() or test_data[numerical].isna().any().any():
    print("Error: NaN found in numerical features")  # Warn user of missing numerical data
    train_data[numerical] = train_data[numerical].fillna(0)  # Impute with 0 for training
    test_data[numerical] = test_data[numerical].fillna(0)  # Impute with 0 for test

# Scale numerical features to normalize their range for model consistency
scaler = StandardScaler()  # Initialize scaler
train_data[numerical] = scaler.fit_transform(train_data[numerical])  # Fit and transform training data
test_data[numerical] = scaler.transform(test_data[numerical])  # Transform test data with training fit

# Prepare feature and label sets for model training and testing
X_train = train_data.drop(['label', 'difficulty'], axis=1)  # Features for training
y_train = train_data['label']  # Labels for training
X_test = test_data.drop(['label', 'difficulty'], axis=1)  # Features for testing
y_test = test_data['label']  # Labels for testing

print("Training Shape:", X_train.shape)  # Verify training data dimensions
print("Test Shape:", X_test.shape)  # Verify test data dimensions

# Train the Random Forest model with specified hyperparameters
model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)  # 50 trees, reproducible, multi-core
model.fit(X_train, y_train)  # Train the model
print("Model Trained Successfully")  # Confirm training completion

# Perform cross-validation to assess model generalization
cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)  # 3-fold CV
print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")  # Report mean and std

# Generate predictions on test set
y_pred = model.predict(X_test)  # Predict labels for test data

# Calculate and display performance metrics
cm = confusion_matrix(y_test, y_pred)  # Compute confusion matrix
print("Confusion Matrix:\n", cm)  # Display raw matrix
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Normal', 'DoS']))  # Detailed metrics
tn, fp, fn, tp = cm.ravel()  # Extract TN, FP, FN, TP
detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0  # Calculate detection rate (recall for DoS)
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # Calculate false positive rate
print(f"Detection Rate: {detection_rate:.4f}")  # Report detection rate
print(f"FPR: {fpr:.4f}")  # Report false positive rate

# Visualize top 10 feature importances for model interpretation
importances = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importances_})  # Create importance DataFrame
top_features = importances.sort_values('importance', ascending=False).head(10)  # Select top 10
plt.figure(figsize=(8, 5))  # Set plot size
sns.barplot(x='importance', y='feature', data=top_features)  # Create bar plot
plt.title('Top 10 Feature Importances')  # Add title
plt.xlabel('Importance')  # Label x-axis
plt.ylabel('Feature')  # Label y-axis
plt.savefig(os.path.join(base_path, 'feature_importance.png'))  # Save plot to file
plt.close()  # Close plot to free memory

# Save performance results to a text file for documentation
with open(os.path.join(base_path, 'dos_detection_results.txt'), 'w') as f:
    f.write(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n\n")  # Write CV accuracy
    f.write("Confusion Matrix:\n" + str(cm) + "\n\n")  # Write confusion matrix
    f.write("Classification Report:\n" + classification_report(y_test, y_pred, target_names=['Normal', 'DoS']))  # Write classification report
    f.write(f"\nDetection Rate: {detection_rate:.4f}\nFPR: {fpr:.4f}\n")  # Write custom metrics

print("\nTop 5 Features:\n", top_features.head(5))  # Display top 5 features for quick reference