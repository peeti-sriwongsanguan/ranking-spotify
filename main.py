import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from src.data_preprocessing import preprocess_data, save_plot
from src.model import get_models, train_model
import torch
import torch.nn as nn
import torch.optim as optim


def tune_random_forest(X_train, y_train):
    param_dist = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [10, 20, 30, 40, 50, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    rf = RandomForestRegressor(random_state=42)
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=100, cv=3, random_state=42,
                                   n_jobs=-1)
    rf_random.fit(X_train, y_train)
    return rf_random.best_estimator_


def tune_gradient_boosting(X_train, y_train):
    param_dist = {
        'n_estimators': [100, 200, 300, 400, 500],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.8, 0.9, 1.0]
    }
    gb = GradientBoostingRegressor(random_state=42)
    gb_random = RandomizedSearchCV(estimator=gb, param_distributions=param_dist, n_iter=100, cv=3, random_state=42,
                                   n_jobs=-1)
    gb_random.fit(X_train, y_train)
    return gb_random.best_estimator_


def tune_knn(X_train, y_train):
    param_dist = {
        'n_neighbors': list(range(1, 31)),
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }
    knn = KNeighborsRegressor()
    knn_random = RandomizedSearchCV(estimator=knn, param_distributions=param_dist, n_iter=100, cv=3, random_state=42,
                                    n_jobs=-1)
    knn_random.fit(X_train, y_train)
    return knn_random.best_estimator_


class ImprovedNN(nn.Module):
    def __init__(self, input_size):
        super(ImprovedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        return self.fc4(x)


def main():
    # Data Preprocessing
    zip_filepath = r'data/Most Streamed Spotify Songs 2024.csv.zip'
    csv_filename = 'Most Streamed Spotify Songs 2024.csv'

    X, y, features = preprocess_data(zip_filepath, csv_filename)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Tune and train models
    results = {}

    print("Tuning Random Forest...")
    rf_tuned = tune_random_forest(X_train, y_train)
    rf_pred = rf_tuned.predict(X_test)
    results['Tuned Random Forest'] = {'MSE': mean_squared_error(y_test, rf_pred), 'R2': r2_score(y_test, rf_pred)}

    print("Tuning Gradient Boosting...")
    gb_tuned = tune_gradient_boosting(X_train, y_train)
    gb_pred = gb_tuned.predict(X_test)
    results['Tuned Gradient Boosting'] = {'MSE': mean_squared_error(y_test, gb_pred), 'R2': r2_score(y_test, gb_pred)}

    print("Tuning KNN...")
    knn_tuned = tune_knn(X_train, y_train)
    knn_pred = knn_tuned.predict(X_test)
    results['Tuned KNN'] = {'MSE': mean_squared_error(y_test, knn_pred), 'R2': r2_score(y_test, knn_pred)}

    print("Training Improved NN...")
    improved_nn = ImprovedNN(X_train.shape[1])
    mse, r2 = train_model(improved_nn, X_train, y_train, X_test, y_test, epochs=200, batch_size=32)
    results['Improved NN'] = {'MSE': mse, 'R2': r2}

    # Plot results
    plt.figure(figsize=(12, 6))
    mse_values = [result['MSE'] for result in results.values()]
    r2_values = [result['R2'] for result in results.values()]

    plt.subplot(1, 2, 1)
    plt.bar(results.keys(), mse_values)
    plt.title('Mean Squared Error')
    plt.xticks(rotation=45, ha='right')

    plt.subplot(1, 2, 2)
    plt.bar(results.keys(), r2_values)
    plt.title('R-squared Score')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    save_plot("tuned_model_comparison.png")

    # Print results
    for name, metrics in results.items():
        print(f"{name}:")
        print(f"  MSE: {metrics['MSE']:.4f}")
        print(f"  R2: {metrics['R2']:.4f}")
        print()

    # Determine the best performing model
    best_model = min(results, key=lambda x: results[x]['MSE'])
    print(f"The best performing model is: {best_model}")


if __name__ == "__main__":
    main()