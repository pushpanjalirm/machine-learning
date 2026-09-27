"""Week 2 lab assignment: BernoulliNB comparison on banking.csv."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
# NOTE: keep this file alongside your banking.csv file in the same directory.
df = pd.read_csv('banking.csv')

# Basic preprocessing
for col in ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']:
    df = df.join(pd.get_dummies(df[col], prefix=col))

# Keep only the relevant columns from the lab notebook
selected = [
    'previous', 'euribor3m',
    'job_blue-collar', 'job_retired', 'job_services', 'job_student',
    'default_no', 'month_aug', 'month_dec', 'month_jul',
    'month_nov', 'month_oct', 'month_sep',
    'day_of_week_fri', 'day_of_week_wed',
    'poutcome_failure', 'poutcome_nonexistent', 'poutcome_success'
]

X = df[selected].copy()
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 1) Default BernoulliNB
bnb_default = BernoulliNB()
bnb_default.fit(X_train, y_train)
y_pred = bnb_default.predict(X_test)
print('Default BernoulliNB accuracy:', accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 2) Median threshold on euribor3m
median_val = X_train['euribor3m'].median()
X_train_med = X_train.copy()
X_test_med = X_test.copy()
X_train_med['euribor3m'] = (X_train_med['euribor3m'] > median_val).astype(int)
X_test_med['euribor3m'] = (X_test_med['euribor3m'] > median_val).astype(int)

bnb_med = BernoulliNB()
bnb_med.fit(X_train_med, y_train)
y_pred_med = bnb_med.predict(X_test_med)
print('Median-threshold BernoulliNB accuracy:', accuracy_score(y_test, y_pred_med))
print(confusion_matrix(y_test, y_pred_med))
print(classification_report(y_test, y_pred_med))

# 3) KBinsDiscretizer for euribor3m
kbin = KBinsDiscretizer(n_bins=4, encode='onehot-dense', strategy='quantile')
train_bins = kbin.fit_transform(X_train[['euribor3m']])
test_bins = kbin.transform(X_test[['euribor3m']])
X_train_bin = pd.concat([X_train.drop(columns='euribor3m').reset_index(drop=True),
                        pd.DataFrame(train_bins, columns=[f'euribor_bin_{i}' for i in range(train_bins.shape[1])])], axis=1)
X_test_bin = pd.concat([X_test.drop(columns='euribor3m').reset_index(drop=True),
                       pd.DataFrame(test_bins, columns=[f'euribor_bin_{i}' for i in range(test_bins.shape[1])])], axis=1)

bnb_bin = BernoulliNB()
bnb_bin.fit(X_train_bin, y_train)
y_pred_bin = bnb_bin.predict(X_test_bin)
print('KBinsDiscretizer BernoulliNB accuracy:', accuracy_score(y_test, y_pred_bin))
print(confusion_matrix(y_test, y_pred_bin))
print(classification_report(y_test, y_pred_bin))
