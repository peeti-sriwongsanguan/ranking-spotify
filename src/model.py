import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pennylane as qml
from sklearn.model_selection import  RandomizedSearchCV
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DeepNN(nn.Module):
    def __init__(self, input_size):
        super(DeepNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.fc5(x)

class ResidualNN(nn.Module):
    def __init__(self, input_size):
        super(ResidualNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.projection = nn.Linear(input_size, 64)

    def forward(self, x):
        identity = self.projection(x)
        out = torch.relu(self.fc1(x))
        out = torch.relu(self.fc2(out) + identity)
        return self.fc3(out)

class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        return self.fc(out.squeeze(1))

class CNNModel(nn.Module):
    def __init__(self, input_size):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * input_size, 1)
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(32)
    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.bn1(torch.relu(self.conv1(x)))
        x = self.bn2(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    inputs = inputs[:n_qubits] #if len(inputs) > n_qubits else inputs + [0] * (n_qubits - len(inputs))
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

class HybridModel(nn.Module):
    def __init__(self, input_size):
        super(HybridModel, self).__init__()
        self.pre_net = nn.Linear(input_size, n_qubits)
        self.q_params = nn.Parameter(torch.randn(3, n_qubits))
        self.post_net = nn.Linear(n_qubits, 1)

    def forward(self, x):
        x = torch.relu(self.pre_net(x))
        q_out = torch.stack([torch.tensor(quantum_circuit(x_item.tolist(), self.q_params.tolist())) for x_item in x])
        x = self.post_net(q_out)
        return x


class DeepHuberNN(nn.Module):
    def __init__(self, input_size):
        super(DeepHuberNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 216)
        self.fc3 = nn.Linear(216, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    abs_error = torch.abs(error)
    quadratic = torch.min(abs_error, torch.tensor(delta))
    linear = abs_error - quadratic
    return torch.mean(0.5 * quadratic.pow(2) + delta * linear)

def train_model(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=32, use_huber=False):
    if use_huber:
        criterion = lambda y_pred, y_true: huber_loss(y_true, y_pred)
    else:
        criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Convert to tensors and handle NaN values
    # X_train_tensor = torch.FloatTensor(np.nan_to_num(X_train, nan=0.0))
    # y_train_tensor = torch.FloatTensor(np.nan_to_num(y_train, nan=0.0)).unsqueeze(1)
    # X_test_tensor = torch.FloatTensor(np.nan_to_num(X_test, nan=0.0))
    # y_test_tensor = torch.FloatTensor(np.nan_to_num(y_test, nan=0.0)).unsqueeze(1)
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.float()

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor)
        mse = mean_squared_error(y_test, y_pred.cpu().numpy())
        r2 = r2_score(y_test, y_pred.cpu().numpy())

    print(f"Final MSE: {mse:.4f}, R2: {r2:.4f}")
    return mse, r2

def get_models(input_size):
    return {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf'),
        'Elastic Net': ElasticNet(random_state=42),
        'KNN': KNeighborsRegressor(n_neighbors=5),
        'XGBoost': xgb.XGBRegressor(random_state=42),
        'LightGBM': lgb.LGBMRegressor(random_state=42),
        'CatBoost': cb.CatBoostRegressor(random_state=42, verbose=False),
        'Simple NN': SimpleNN(input_size),
        'Deep NN': DeepNN(input_size),
        'Residual NN': ResidualNN(input_size),
        'LSTM': LSTMModel(input_size),
        'CNN': CNNModel(input_size),
        'Quantum NN': HybridModel(input_size),
        'Deep Huber NN': DeepHuberNN(input_size)
    }
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
    xgb_model = xgb.XGBRegressor(random_state=42)
    xgb_random = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_dist, n_iter=20, cv=3, random_state=42, n_jobs=-1)
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
    gb_random = RandomizedSearchCV(estimator=gb, param_distributions=param_dist, n_iter=100, cv=3, random_state=42, n_jobs=-1)
    gb_random.fit(X_train, y_train)
    return gb_random.best_estimator_

def tune_svr(X_train, y_train):
    param_dist = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 1],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    svr = SVR()
    svr_random = RandomizedSearchCV(estimator=svr, param_distributions=param_dist, n_iter=50, cv=3, random_state=42, n_jobs=-1)
    svr_random.fit(X_train, y_train)
    return svr_random.best_estimator_

def tune_elastic_net(X_train, y_train):
    param_dist = {
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
        'l1_ratio': np.arange(0.0, 1.0, 0.1)
    }
    en = ElasticNet(random_state=42)
    en_random = RandomizedSearchCV(estimator=en, param_distributions=param_dist, n_iter=50, cv=3, random_state=42, n_jobs=-1)
    en_random.fit(X_train, y_train)
    return en_random.best_estimator_


def tune_knn(X_train, y_train):
    param_dist = {
        'n_neighbors': list(range(1, 31)),
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }
    knn = KNeighborsRegressor()

    # Calculate total number of combinations
    n_combinations = len(param_dist['n_neighbors']) * len(param_dist['weights']) * len(param_dist['p'])

    # Use min of n_combinations and 50 for n_iter
    n_iter = min(n_combinations, 50)

    knn_random = RandomizedSearchCV(estimator=knn, param_distributions=param_dist, n_iter=n_iter, cv=3, random_state=42,
                                    n_jobs=-1)
    knn_random.fit(X_train, y_train)
    return knn_random.best_estimator_

def tune_lightgbm(X_train, y_train):
    param_dist = {
        'num_leaves': [31, 50, 70],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'min_child_samples': [20, 30, 50]
    }
    lgb_model = lgb.LGBMRegressor(random_state=42)
    lgb_random = RandomizedSearchCV(estimator=lgb_model, param_distributions=param_dist, n_iter=20, cv=3, random_state=42, n_jobs=-1)
    lgb_random.fit(X_train, y_train)
    return lgb_random.best_estimator_


def tune_catboost(X_train, y_train):
    param_dist = {
        'depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'iterations': [100, 200, 300],
        'l2_leaf_reg': [1, 3, 5, 7, 9],
        'border_count': [32, 64, 128]
    }
    catboost_model = cb.CatBoostRegressor(random_state=42, verbose=False)
    catboost_random = RandomizedSearchCV(estimator=catboost_model, param_distributions=param_dist, n_iter=20, cv=3, random_state=42, n_jobs=-1)
    catboost_random.fit(X_train, y_train)
    return catboost_random.best_estimator_