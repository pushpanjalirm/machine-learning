#!/usr/bin/env python3
"""
Run regularization exercises (Linear, Ridge, Lasso, ElasticNet) on the Boston Housing CSV

Produces printed metrics and saves coefficient plots into outputs/.

Usage: python run_regularization.py
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score


def find_boston_csv():
    start = Path(__file__).resolve().parent
    # search upwards and in subtree for the CSV
    for p in list(start.rglob('BostonHousing.csv')) + list(start.parents[0].rglob('BostonHousing.csv')):
        return p
    # fallback: try sibling known folder name
    candidate = start.parent / 'Wk-1-Regression-Lab Activity-20260915' / 'BostonHousing.csv'
    if candidate.exists():
        return candidate
    raise FileNotFoundError('BostonHousing.csv not found in workspace')


def load_and_preprocess(csv_path: Path):
    df = pd.read_csv(csv_path)
    # Basic cleaning
    df = df.dropna()
    # For Boston dataset all features are numeric; target column might be 'MEDV' or 'target'
    if 'MEDV' in df.columns:
        target_col = 'MEDV'
    elif 'target' in df.columns:
        target_col = 'target'
    else:
        # assume last column is target
        target_col = df.columns[-1]

    X = df.drop(columns=[target_col])
    y = df[target_col].values

    # Scale numeric features
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[numeric_cols])
    X_scaled = pd.DataFrame(X_scaled, columns=numeric_cols)
    return X_scaled, y


def evaluate_model(model, X_train, X_test, y_train, y_test):
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    return dict(train_mse=train_mse, test_mse=test_mse, train_r2=train_r2, test_r2=test_r2)


def plot_coefficients(alphas, coefs, title, outpath):
    plt.figure(figsize=(8, 6))
    coefs = np.array(coefs)
    for i in range(coefs.shape[1]):
        plt.plot(alphas, coefs[:, i], marker='o')
    plt.xscale('log')
    plt.xlabel('alpha')
    plt.ylabel('coefficient value')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def main():
    warnings.filterwarnings('ignore')
    sns.set()

    csv_path = find_boston_csv()
    print('Using dataset:', csv_path)

    X, y = load_and_preprocess(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    outdir = Path(__file__).resolve().parent / 'outputs'
    outdir.mkdir(parents=True, exist_ok=True)

    # Exercise 1: Baseline Linear Regression
    print('\nExercise 1: Linear Regression (no regularization)')
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    results_lr = evaluate_model(lr, X_train, X_test, y_train, y_test)
    print('LinearRegression results:')
    print(results_lr)

    # Exercise 2: Ridge
    print('\nExercise 2: Ridge Regression')
    alphas = [0.01, 0.1, 1, 10, 100]
    ridge_coefs = []
    ridge_results = {}
    for a in alphas:
        model = Ridge(alpha=a)
        model.fit(X_train, y_train)
        ridge_coefs.append(model.coef_)
        ridge_results[a] = evaluate_model(model, X_train, X_test, y_train, y_test)
    print('Ridge results (alpha -> metrics):')
    for a in alphas:
        print(a, ridge_results[a])
    plot_coefficients(alphas, ridge_coefs, 'Ridge coefficients vs alpha', outdir / 'ridge_coefs.png')

    # Exercise 3: Lasso
    print('\nExercise 3: Lasso Regression')
    lasso_alphas = alphas
    lasso_coefs = []
    lasso_results = {}
    nonzero_counts = []
    for a in lasso_alphas:
        model = Lasso(alpha=a, max_iter=10000)
        model.fit(X_train, y_train)
        lasso_coefs.append(model.coef_)
        nonzero_counts.append(np.sum(model.coef_ != 0))
        lasso_results[a] = evaluate_model(model, X_train, X_test, y_train, y_test)
    print('Lasso results (alpha -> metrics):')
    for a in lasso_alphas:
        print(a, lasso_results[a], 'nonzero_coeffs=', int((np.sum(lasso_coefs[lasso_alphas.index(a)] != 0))))
    plot_coefficients(lasso_alphas, lasso_coefs, 'Lasso coefficients vs alpha', outdir / 'lasso_coefs.png')

    # plot number of non-zero coefficients vs alpha
    plt.figure()
    plt.plot(lasso_alphas, nonzero_counts, marker='o')
    plt.xscale('log')
    plt.xlabel('alpha')
    plt.ylabel('number of non-zero coefficients')
    plt.title('Lasso: non-zero coefficients vs alpha')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outdir / 'lasso_nonzero_counts.png')
    plt.close()

    # Exercise 4: ElasticNet
    print('\nExercise 4: ElasticNet')
    en_alphas = [0.01, 0.1, 1]
    l1_ratios = [0.2, 0.5, 0.8]
    en_results = {}
    for lr_ratio in l1_ratios:
        coefs = []
        for a in en_alphas:
            model = ElasticNet(alpha=a, l1_ratio=lr_ratio, max_iter=10000)
            model.fit(X_train, y_train)
            coefs.append(model.coef_)
            en_results[(a, lr_ratio)] = evaluate_model(model, X_train, X_test, y_train, y_test)
        plot_coefficients(en_alphas, coefs, f'ElasticNet coefs l1_ratio={lr_ratio}', outdir / f'en_coefs_l1_{lr_ratio}.png')

    print('ElasticNet sample results (alpha,l1_ratio) -> metrics:')
    for k, v in en_results.items():
        print(k, v)

    # Bonus: GridSearchCV for ElasticNet
    print('\nBonus: GridSearchCV for ElasticNet')
    param_grid = {'alpha': [0.001, 0.01, 0.1, 1], 'l1_ratio': [0.2, 0.5, 0.8]}
    gs = GridSearchCV(ElasticNet(max_iter=10000), param_grid, cv=5, scoring='r2')
    gs.fit(X_train, y_train)
    print('Best params (ElasticNet):', gs.best_params_)
    best_en = gs.best_estimator_
    best_en_metrics = evaluate_model(best_en, X_train, X_test, y_train, y_test)
    print('ElasticNet (grid-search) metrics:', best_en_metrics)

    print('\nCompare best models on test set:')
    # Evaluate best Ridge/Lasso found earlier by picking alpha with best test_r2
    best_ridge_alpha = max(ridge_results.keys(), key=lambda a: ridge_results[a]['test_r2'])
    best_lasso_alpha = max(lasso_results.keys(), key=lambda a: lasso_results[a]['test_r2'])
    print('Best ridge alpha:', best_ridge_alpha, 'metrics:', ridge_results[best_ridge_alpha])
    print('Best lasso alpha:', best_lasso_alpha, 'metrics:', lasso_results[best_lasso_alpha])
    print('Best elastic net (gridsearch):', best_en_metrics)

    print('\nOutputs (plots) saved to:', outdir)


if __name__ == '__main__':
    main()
