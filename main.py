import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler,PowerTransformer
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from src.data_preprocessing import preprocess_data, save_plot
from src.model import *
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
# from src.model import get_models, train_model,SVR, ElasticNet, KNeighborsRegressor, SimpleNN
import src.model as mdl
import time
from sklearn.exceptions import ConvergenceWarning
import warnings
from functools import partial

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_polynomial_features(X, degree=2):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    return poly.fit_transform(X)


def handle_outliers(X, y, contamination=0.1):
    from sklearn.ensemble import IsolationForest
    clf = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    outlier_labels = clf.fit_predict(X)
    X_clean = X[outlier_labels != -1]
    y_clean = y[outlier_labels != -1]
    return X_clean, y_clean


def create_interaction_terms(X):
    n_features = X.shape[1]
    interactions = []
    for i in range(n_features):
        for j in range(i+1, n_features):
            interactions.append(X[:, i] * X[:, j])
    return np.column_stack(interactions)

def apply_non_linear_transformations(X):
    X_transformed = np.column_stack([
        np.log1p(np.abs(X)),
        np.sqrt(np.abs(X)),
        np.square(X)
    ])
    return X_transformed


def feature_engineering(X):
    # Create interaction terms
    X_interact = create_interaction_terms(X)

    # Apply non-linear transformations
    X_non_linear = apply_non_linear_transformations(X)

    # Combine all features
    X_engineered = np.hstack((X, X_interact, X_non_linear))

    # Remove constant and highly correlated features
    selector = VarianceThreshold(threshold=0.01)
    X_engineered = selector.fit_transform(X_engineered)

    # Remove highly correlated features
    corr_matrix = np.abs(np.corrcoef(X_engineered.T))
    upper_tri = corr_matrix[np.triu_indices(corr_matrix.shape[0], k=1)]
    to_drop = [column for column in range(X_engineered.shape[1])
               if any(corr_matrix[:, column] > 0.95) and corr_matrix[column, column] != 1]
    X_engineered = np.delete(X_engineered, to_drop, axis=1)

    return X_engineered


class ImprovedNN(nn.Module):
    def __init__(self, input_size):
        super(ImprovedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.fc5 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        return self.fc5(x)


def train_improved_nn(model, X_train, y_train, X_test, y_test, epochs=300, batch_size=64):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test.values).unsqueeze(1)

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            test_loss = criterion(model(X_test_tensor), y_test_tensor)
        scheduler.step(test_loss)

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor)
        mse = mean_squared_error(y_test, y_pred.numpy())
        r2 = r2_score(y_test, y_pred.numpy())

    return mse, r2
def tune_random_forest(X_train, y_train):
    param_dist = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [10, 20, 30, 40, 50, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestRegressor(random_state=42)
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=100, cv=3, random_state=42, n_jobs=-1)
    rf_random.fit(X_train, y_train)
    return rf_random.best_estimator_


def tune_xgboost(X_train, y_train):
    param_dist = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.3],
        'max_depth': [3, 4, 5, 6],
        'min_child_weight': [1, 2, 3],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
    xgb_random = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_dist, n_iter=20, cv=3,
                                    random_state=42, n_jobs=-1)
    xgb_random.fit(X_train, y_train)
    return xgb_random.best_estimator_


def tune_gradient_boosting(X_train, y_train):
    param_dist = {
        'n_estimators': [100, 200, 300, 400, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'max_depth': [3, 4, 5, 6, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    gb = GradientBoostingRegressor(random_state=42)
    gb_random = RandomizedSearchCV(estimator=gb, param_distributions=param_dist, n_iter=100, cv=3, random_state=42,
                                   n_jobs=-1)
    gb_random.fit(X_train, y_train)
    return gb_random.best_estimator_


def tune_svr(X_train, y_train):
    param_dist = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 1],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    svr = mdl.SVR()
    svr_random = RandomizedSearchCV(estimator=svr, param_distributions=param_dist, n_iter=50, cv=3, random_state=42,
                                    n_jobs=-1)
    svr_random.fit(X_train, y_train)
    return svr_random.best_estimator_


def tune_elastic_net(X_train, y_train):
    param_dist = {
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
        'l1_ratio': np.arange(0.0, 1.0, 0.1)
    }
    en = mdl.ElasticNet(random_state=42)
    en_random = RandomizedSearchCV(estimator=en, param_distributions=param_dist, n_iter=50, cv=3, random_state=42,
                                   n_jobs=-1)
    en_random.fit(X_train, y_train)
    return en_random.best_estimator_


def tune_knn(X_train, y_train):
    param_dist = {
        'n_neighbors': list(range(1, 31)),
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }
    knn = mdl.KNeighborsRegressor()
    knn_random = RandomizedSearchCV(estimator=knn, param_distributions=param_dist, n_iter=50, cv=3, random_state=42,
                                    n_jobs=-1)
    knn_random.fit(X_train, y_train)
    return knn_random.best_estimator_


def tune_nn(model_class, input_size, X_train, y_train, X_test, y_test):
    best_mse = float('inf')
    best_model = None
    best_r2 = -float('inf')
    learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [32, 64, 128]
    epochs_list = [100, 200, 300]

    for lr in learning_rates:
        for batch_size in batch_sizes:
            for epochs in epochs_list:
                model = model_class(input_size)  # Create a new instance of the model
                mse, r2 = train_model(model, X_train, y_train, X_test, y_test, epochs=epochs, batch_size=batch_size)
                if mse < best_mse:
                    best_mse = mse
                    best_r2 = r2
                    best_model = model

    return best_model, best_mse, best_r2
def tune_lightgbm(X_train, y_train):
    param_dist = {
        'num_leaves': [15, 31, 50],
        'learning_rate': [0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'min_child_samples': [20, 30, 50],
        'min_split_gain': [0.01, 0.1, 0.3],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0, 0.1, 0.5]
    }

    def fit_with_timeout(estimator, X, y, timeout=300):
        start_time = time.time()
        estimator.fit(X, y)
        if time.time() - start_time > timeout:
            raise TimeoutError("LightGBM training exceeded the time limit.")
        return estimator

    lgb_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1, min_data_in_leaf=5, max_depth=10)
    fit_with_timeout_partial = partial(fit_with_timeout, timeout=300)  # 5 minutes timeout

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        lgb_random = RandomizedSearchCV(
            estimator=lgb_model,
            param_distributions=param_dist,
            n_iter=20,
            cv=3,
            random_state=42,
            n_jobs=-1,
            error_score='raise'
        )
        try:
            lgb_random.fit(X_train, y_train)
            return lgb_random.best_estimator_
        except Exception as e:
            logging.warning(f"LightGBM tuning failed: {str(e)}")
            return None


def main():
    start_time = time.time()
    logging.info("Starting the main process...")

    # Data Preprocessing
    zip_filepath = r'data/Most Streamed Spotify Songs 2024.csv.zip'
    csv_filename = 'Most Streamed Spotify Songs 2024.csv'

    X, y, features = preprocess_data(zip_filepath, csv_filename)
    logging.info(f"Data preprocessed. Shape of X: {X.shape}, Shape of y: {y.shape}")

    # Handle outliers
    X, y = handle_outliers(X, y)
    logging.info(f"Outliers handled. New shape of X: {X.shape}, New shape of y: {y.shape}")

    # Create polynomial features
    X_poly = create_polynomial_features(X)
    logging.info(f"Polynomial features created. New shape of X: {X_poly.shape}")

    # Feature engineering
    X_engineered = feature_engineering(X_poly)
    logging.info(f"Feature engineering completed. New shape of X: {X_engineered.shape}")

    pt = PowerTransformer(method='yeo-johnson')
    y_transformed = pt.fit_transform(y.reshape(-1, 1)).ravel()

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_engineered, y_transformed, test_size=0.2, random_state=42)
    logging.info("Data split into train and test sets")


    # Feature selection
    selector = SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    logging.info(f"Feature selection completed. New shape of X_train: {X_train_selected.shape}")

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)

    # Convert data to float32
    X_train_scaled = X_train_scaled.astype(np.float32)
    X_test_scaled = X_test_scaled.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

    # Get all models
    input_size = X_train_scaled.shape[1]
    models = mdl.get_models(input_size)

    # Train and evaluate models
    results = {'Non-tuned': {}, 'Tuned': {}}

    # Get all models
    input_size = X_train_scaled.shape[1]
    models = get_models(input_size)

    # Train and evaluate non-tuned models
    for name, model in models.items():
        logging.info(f"Training non-tuned {name}...")
        try:
            if name in ['Random Forest', 'Gradient Boosting', 'SVR', 'Elastic Net', 'KNN', 'XGBoost', 'LightGBM']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
            else:
                mse, r2 = train_model(model, X_train_scaled, y_train, X_test_scaled, y_test)
            results['Non-tuned'][name] = {'MSE': mse, 'R2': r2}
            logging.info(f"Non-tuned {name} training completed. MSE: {mse:.4f}, R2: {r2:.4f}")
        except Exception as e:
            logging.error(f"Error training non-tuned {name}: {str(e)}")
            continue

    # Tune and evaluate models
    tuning_functions = {
        'Random Forest': tune_random_forest,
        'XGBoost': tune_xgboost,
        'Gradient Boosting': tune_gradient_boosting,
        'SVR': tune_svr,
        'Elastic Net': tune_elastic_net,
        'KNN': tune_knn,
        'LightGBM': tune_lightgbm
    }

    for name, tune_func in tuning_functions.items():
        logging.info(f"Tuning {name}...")
        try:
            tuned_model = tune_func(X_train_scaled, y_train)
            y_pred = tuned_model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            results['Tuned'][name] = {'MSE': mse, 'R2': r2}
            logging.info(f"Tuned {name} completed. MSE: {mse:.4f}, R2: {r2:.4f}")
        except Exception as e:
            logging.error(f"Error tuning {name}: {str(e)}")
            continue

    # Tune and evaluate neural network models
    nn_models = {
        'Simple NN': SimpleNN,
        'Deep NN': DeepNN,
        'Residual NN': ResidualNN,
        'LSTM': LSTMModel,
        'CNN': CNNModel,
        'Hybrid Model': HybridModel
    }

    for name, model_class in nn_models.items():
        logging.info(f"Tuning {name}...")
        try:
            best_model, mse, r2 = tune_nn(model_class, input_size, X_train_scaled, y_train, X_test_scaled, y_test)
            results['Tuned'][name] = {'MSE': mse, 'R2': r2}
            logging.info(f"Tuned {name} completed. MSE: {mse:.4f}, R2: {r2:.4f}")
        except Exception as e:
            logging.error(f"Error tuning {name}: {str(e)}")
            continue

    # Plot results
    plt.figure(figsize=(20, 15))

    def plot_metric(metric, subplot):
        plt.subplot(subplot)
        non_tuned_models = list(results['Non-tuned'].keys())
        tuned_models = list(results['Tuned'].keys())
        all_models = list(set(non_tuned_models + tuned_models))

        non_tuned_values = [results['Non-tuned'].get(model, {}).get(metric, 0) for model in all_models]
        tuned_values = [results['Tuned'].get(model, {}).get(metric, 0) for model in all_models]

        x = range(len(all_models))
        width = 0.35
        plt.bar([i - width / 2 for i in x], non_tuned_values, width, label='Non-tuned', color='blue', alpha=0.7)
        plt.bar([i + width / 2 for i in x], tuned_values, width, label='Tuned', color='red', alpha=0.7)
        plt.title(f'{metric} Comparison')
        plt.xticks(x, all_models, rotation=45, ha='right')
        plt.legend()

    plot_metric('MSE', 211)
    plot_metric('R2', 212)

    plt.tight_layout()
    save_plot("tuned_vs_non_tuned_comparison.png")
    logging.info("Results plotted and saved")

    # Print results
    for category in ['Non-tuned', 'Tuned']:
        print(f"\n{category} Models:")
        for name, metrics in results[category].items():
            print(f"  {name}:")
            print(f"    MSE: {metrics['MSE']:.4f}")
            print(f"    R2: {metrics['R2']:.4f}")

    # Determine the best performing model overall
    best_model = max(
        [(cat, model, metrics['R2']) for cat in results for model, metrics in results[cat].items()],
        key=lambda x: x[2]
    )
    print(f"\nThe best performing model is: {best_model[0]} {best_model[1]} with R2 score: {best_model[2]:.4f}")

    end_time = time.time()
    logging.info(f"Total runtime: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()