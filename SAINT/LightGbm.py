import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix

pd_list = []
pf_list = []
bal_list = []
fir_list = []

def classifier_eval(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix: ', cm)
    PD = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    print('PD: ', PD)
    PF = cm[0, 1] / (cm[0, 0] + cm[0, 1])
    print('PF: ', PF)
    balance = 1 - np.sqrt((1 - PD)**2 + PF**2) / np.sqrt(2)
    print('Balance: ', balance)
    FI = (cm[1, 1] + cm[0, 1]) / (cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1])
    FIR = (PD - FI) / PD
    print('FIR: ', FIR)
    return PD, PF, balance, FIR

# CSV file path
csv_file_path = "EQ.csv"

# Read CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Extract features (X) and target variable (y) from the DataFrame
X = df.drop(columns=['class'])
y = df['class']

# Split the data into training, validation, and test sets
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_test_normalized = scaler.fit_transform(X_test)

# K-Fold cross-validation setup
k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# K-Fold cross-validation loop
for train_index, val_index in kf.split(X_train):
    X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]

    # Min-Max normalization
    X_fold_train_normalized = scaler.fit_transform(X_fold_train)
    X_fold_val_normalized = scaler.transform(X_fold_val)

    # SMOTE for oversampling
    smote = SMOTE(random_state=42)
    X_fold_train_resampled, y_fold_train_resampled = smote.fit_resample(X_fold_train_normalized, y_fold_train)

    # LightGBM model
    lgbm_classifier = LGBMClassifier(
        n_estimators=2048,
        learning_rate=0.01,
        max_depth=5,
        objective='binary',
        eval_metric=['logloss', 'auc'],
        verbosity=-1
    )

    lgbm_classifier.fit(X_fold_train_resampled, y_fold_train_resampled)

    # Test data predictions
    lgbm_preds = lgbm_classifier.predict(X_test_normalized)

    # Classifier evaluation and recording results
    PD, PF, balance, FIR = classifier_eval(y_test, lgbm_preds)
    pd_list.append(PD)
    pf_list.append(PF)
    bal_list.append(balance)
    fir_list.append(FIR)

# Print the results
print('Average PD: {}'.format((sum(pd_list) / len(pd_list))))
print('Average PF: {}'.format((sum(pf_list) / len(pf_list))))
print('Average balance: {}'.format((sum(bal_list) / len(bal_list))))
print('Average FIR: {}'.format((sum(fir_list) / len(fir_list))))
