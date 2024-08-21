import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from data_preprocessing import read_zip_file, clean_numeric_columns, clean_data
from sklearn.neural_network import MLPRegressor


class RankNet(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(RankNet, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def ranknet_loss(y_pred, y_true):
    diff_matrix = y_pred - y_pred.t()
    true_diff_matrix = y_true - y_true.t()
    loss = torch.mean(torch.log(1 + torch.exp(-true_diff_matrix * diff_matrix)))
    return loss

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class AdvancedSpotifySongAnalysis:
    def __init__(self, zip_filepath, csv_filename):
        self.zip_filepath = zip_filepath
        self.csv_filename = csv_filename
        self.df = None
        self.df_cleaned = None
        self.X = None
        self.y = None
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4
        self.scaler = StandardScaler()
        self.device = self.get_device()
        print(f"Using device: {self.device}")

    def get_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def load_and_preprocess_data(self):
        self.df = read_zip_file(self.zip_filepath, self.csv_filename)
        numeric_columns = [
            'Spotify_Streams', 'Spotify_Playlist_Count', 'Spotify_Playlist_Reach',
            'YouTube_Views', 'YouTube_Likes', 'TikTok_Posts', 'TikTok_Likes',
            'TikTok_Views', 'YouTube_Playlist_Reach', 'AirPlay_Spins', 'Deezer_Playlist_Reach',
            'Pandora_Streams', 'Pandora_Track_Stations', 'Shazam_Counts', 'Track_Score'
        ]
        self.df = clean_numeric_columns(self.df, numeric_columns)
        self.df['All_Time_Rank'] = self.df['All_Time_Rank'].str.replace(',', '').astype('int')
        self.df_cleaned = clean_data(self.df, numeric_columns + ['All_Time_Rank'])

    def prepare_features_and_target(self):
        self.X = self.df_cleaned[['YouTube_Views', 'YouTube_Likes', 'Spotify_Streams', 'TikTok_Views', 'TikTok_Likes',
                                  'Spotify_Playlist_Count']]
        self.y = self.df_cleaned['All_Time_Rank']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                random_state=42)
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

    def plot_metric_correlations(self):
        metrics = ['YouTube_Views', 'YouTube_Likes', 'Spotify_Streams', 'TikTok_Views']
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=metric, y='All_Time_Rank', data=self.df)
            plt.title(f'All Time Rank vs {metric}', fontsize=16, pad=20)
            plt.xlabel(metric, fontsize=12)
            plt.ylabel('All Time Rank', fontsize=12)
            plt.yscale('log')
            plt.xscale('log')
            plt.tight_layout()
            plt.show()

    def plot_feature_distributions(self):
        features = ['YouTube_Views', 'YouTube_Likes', 'Spotify_Streams', 'TikTok_Views', 'TikTok_Likes',
                    'Spotify_Playlist_Count']
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Feature Distributions', fontsize=20, y=1.02)
        for ax, feature in zip(axes.ravel(), features):
            sns.histplot(self.df[feature], kde=True, ax=ax)
            ax.set_title(f'Distribution of {feature}', fontsize=14)
            ax.set_xlabel(feature, fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_xscale('log')
        plt.tight_layout()
        plt.show()

    def train_ranknet(self, X_train, y_train, X_test, y_test, hidden_sizes=(100, 50), lr=0.001, num_epochs=1000,
                      batch_size=64, patience=10):
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train.values).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test.values).to(self.device)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model = RankNet(X_train.shape[1], hidden_sizes).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_val_loss = float('inf')
        best_model = None
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = ranknet_loss(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test_tensor)
                val_loss = ranknet_loss(val_outputs.squeeze(), y_test_tensor)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = model.state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve == patience:
                    print(f'Early stopping triggered at epoch {epoch + 1}')
                    break

            if (epoch + 1) % 100 == 0:
                print(
                    f'RankNet Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

        # Load the best model
        model.load_state_dict(best_model)

        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_tensor).cpu().numpy().flatten()

        return y_pred

    def train_mlp_ranknet(self, X_train, y_train, X_test, y_test, hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42):
        model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return y_pred

    def train_lstm(self, X_train, y_train, X_test, y_test, hidden_size=50, num_layers=1, lr=0.001, num_epochs=50,
                   batch_size=64, patience=10):
        X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train.values).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test.values).to(self.device)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model = LSTMModel(input_size=X_train.shape[1], hidden_size=hidden_size, num_layers=num_layers,
                          output_size=1).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_val_loss = float('inf')
        best_model = None
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test_tensor)
                val_loss = criterion(val_outputs.squeeze(), y_test_tensor)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = model.state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve == patience:
                    print(f'Early stopping triggered at epoch {epoch + 1}')
                    break

            if (epoch + 1) % 10 == 0:
                print(
                    f'LSTM Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

        # Load the best model
        model.load_state_dict(best_model)

        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_tensor).cpu().numpy().flatten()

        return y_pred

    def tune_lstm(self):
        hidden_sizes = [32, 64, 128]
        num_layers_list = [1, 2]
        learning_rates = [0.001, 0.01, 0.1]
        num_epochs_list = [100, 200, 300]  # Increased max epochs since we have early stopping
        batch_sizes = [32, 64, 128]
        patience_values = [5, 10, 20]

        best_spearman = -1
        best_params = {}

        for hidden_size in hidden_sizes:
            for num_layers in num_layers_list:
                for lr in learning_rates:
                    for num_epochs in num_epochs_list:
                        for batch_size in batch_sizes:
                            for patience in patience_values:
                                print(f"Tuning LSTM with hidden_size={hidden_size}, num_layers={num_layers}, lr={lr}, "
                                      f"num_epochs={num_epochs}, batch_size={batch_size}, patience={patience}")
                                y_pred = self.train_lstm(self.X_train_scaled, self.y_train, self.X_test_scaled,
                                                         self.y_test,
                                                         hidden_size=hidden_size, num_layers=num_layers, lr=lr,
                                                         num_epochs=num_epochs, batch_size=batch_size,
                                                         patience=patience)
                                spearman_corr, _ = spearmanr(self.y_test, y_pred)

                                if spearman_corr > best_spearman:
                                    best_spearman = spearman_corr
                                    best_params = {'hidden_size': hidden_size, 'num_layers': num_layers,
                                                   'lr': lr, 'num_epochs': num_epochs, 'batch_size': batch_size,
                                                   'patience': patience}

        print(f"Best LSTM parameters: {best_params}")
        return best_params
    def train_and_evaluate_models(self):
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
            'CatBoost': CatBoostRegressor(iterations=100, random_state=42, verbose=False),
        }

        results = {}
        for name, model in models.items():
            model.fit(self.X_train_scaled, self.y_train)
            y_pred = model.predict(self.X_test_scaled)
            mse = mean_squared_error(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            spearman_corr, _ = spearmanr(self.y_test, y_pred)
            results[name] = {'MSE': mse, 'MAE': mae, 'Spearman Correlation': spearman_corr}

        print("Training RankNet...")
        ranknet_pred = self.train_ranknet(self.X_train_scaled, self.y_train, self.X_test_scaled, self.y_test)
        ranknet_mse = mean_squared_error(self.y_test, ranknet_pred)
        ranknet_mae = mean_absolute_error(self.y_test, ranknet_pred)
        ranknet_spearman, _ = spearmanr(self.y_test, ranknet_pred)
        results['RankNet'] = {'MSE': ranknet_mse, 'MAE': ranknet_mae, 'Spearman Correlation': ranknet_spearman}

        print("Training LSTM...")
        lstm_pred = self.train_lstm(self.X_train_scaled, self.y_train, self.X_test_scaled, self.y_test)
        lstm_mse = mean_squared_error(self.y_test, lstm_pred)
        lstm_mae = mean_absolute_error(self.y_test, lstm_pred)
        lstm_spearman, _ = spearmanr(self.y_test, lstm_pred)
        results['LSTM'] = {'MSE': lstm_mse, 'MAE': lstm_mae, 'Spearman Correlation': lstm_spearman}

        return results, models

    def hyperparameter_tuning(self, model, param_grid):
        search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=3, random_state=42, n_jobs=-1)
        search.fit(self.X_train_scaled, self.y_train)
        return search.best_estimator_

    def tune_ranknet(self):
        hidden_sizes_list = [(64,), (128,), (64, 32), (128, 64)]
        learning_rates = [0.001, 0.01, 0.1]
        num_epochs_list = [500, 1000, 1500]
        batch_sizes = [32, 64, 128]
        patience_values = [5, 10, 20]

        best_spearman = -1
        best_params = {}

        for hidden_sizes in hidden_sizes_list:
            for lr in learning_rates:
                for num_epochs in num_epochs_list:
                    for batch_size in batch_sizes:
                        for patience in patience_values:
                            print(f"Tuning RankNet with hidden_sizes={hidden_sizes}, lr={lr}, "
                                  f"num_epochs={num_epochs}, batch_size={batch_size}, patience={patience}")
                            y_pred = self.train_ranknet(self.X_train_scaled, self.y_train, self.X_test_scaled,
                                                        self.y_test,
                                                        hidden_sizes=hidden_sizes, lr=lr, num_epochs=num_epochs,
                                                        batch_size=batch_size, patience=patience)
                            spearman_corr, _ = spearmanr(self.y_test, y_pred)

                            if spearman_corr > best_spearman:
                                best_spearman = spearman_corr
                                best_params = {'hidden_sizes': hidden_sizes, 'lr': lr, 'num_epochs': num_epochs,
                                               'batch_size': batch_size, 'patience': patience}

        print(f"Best RankNet parameters: {best_params}")
        return best_params

    def ensemble_prediction(self, tuned_models, ranknet_params, lstm_params):
        predictions = []
        for name, model in tuned_models.items():
            pred = model.predict(self.X_test_scaled)
            predictions.append(pred)

        # Custom PyTorch RankNet prediction
        ranknet_pred = self.train_ranknet(self.X_train_scaled, self.y_train, self.X_test_scaled, self.y_test,
                                          hidden_sizes=ranknet_params['hidden_sizes'],
                                          lr=ranknet_params['lr'],
                                          num_epochs=ranknet_params['num_epochs'],
                                          batch_size=ranknet_params['batch_size'],
                                          patience=ranknet_params['patience'])
        predictions.append(ranknet_pred)

        # MLP RankNet prediction
        mlp_ranknet_pred = self.train_mlp_ranknet(self.X_train_scaled, self.y_train, self.X_test_scaled, self.y_test)
        predictions.append(mlp_ranknet_pred)

        # LSTM prediction
        lstm_pred = self.train_lstm(self.X_train_scaled, self.y_train, self.X_test_scaled, self.y_test,
                                    hidden_size=lstm_params['hidden_size'],
                                    num_layers=lstm_params['num_layers'],
                                    lr=lstm_params['lr'],
                                    num_epochs=lstm_params['num_epochs'],
                                    batch_size=lstm_params['batch_size'],
                                    patience=lstm_params['patience'])
        predictions.append(lstm_pred)

        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred

    def run_analysis(self):
        self.load_and_preprocess_data()
        self.prepare_features_and_target()

        print("Plotting metric correlations and feature distributions...")
        self.plot_metric_correlations()
        self.plot_feature_distributions()

        print("\nTraining and evaluating initial models...")
        model_results, initial_models = self.train_and_evaluate_models()

        print("Initial Model Evaluation Results:")
        for model, metrics in model_results.items():
            print(f"\n{model}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")

        print("\nPerforming hyperparameter tuning...")
        rf_params = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20, 30]}
        xgb_params = {'n_estimators': [100, 200, 300], 'max_depth': [3, 4, 5], 'learning_rate': [0.01, 0.1, 0.3]}
        cb_params = {'iterations': [100, 200, 300], 'depth': [4, 6, 8], 'learning_rate': [0.01, 0.1, 0.3]}

        tuned_models = {
            'Random Forest': self.hyperparameter_tuning(RandomForestRegressor(random_state=42), rf_params),
            'XGBoost': self.hyperparameter_tuning(XGBRegressor(random_state=42), xgb_params),
            'CatBoost': self.hyperparameter_tuning(CatBoostRegressor(random_state=42, verbose=False), cb_params)
        }

        print("\nTuning RankNet...")
        best_ranknet_params = self.tune_ranknet()



        print("\nTuning LSTM...")
        best_lstm_params = self.tune_lstm()

        print("\nEvaluating tuned models...")
        tuned_results = {}
        for name, model in tuned_models.items():
            y_pred = model.predict(self.X_test_scaled)
            mse = mean_squared_error(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            spearman_corr, _ = spearmanr(self.y_test, y_pred)
            tuned_results[name] = {'MSE': mse, 'MAE': mae, 'Spearman Correlation': spearman_corr}

        print("\nTraining MLP RankNet...")
        mlp_ranknet_pred = self.train_mlp_ranknet(self.X_train_scaled, self.y_train, self.X_test_scaled, self.y_test)
        mlp_ranknet_mse = mean_squared_error(self.y_test, mlp_ranknet_pred)
        mlp_ranknet_mae = mean_absolute_error(self.y_test, mlp_ranknet_pred)
        mlp_ranknet_spearman, _ = spearmanr(self.y_test, mlp_ranknet_pred)
        tuned_results['MLP RankNet'] = {'MSE': mlp_ranknet_mse, 'MAE': mlp_ranknet_mae,
                                        'Spearman Correlation': mlp_ranknet_spearman}

        # Evaluate tuned custom RankNet
        ranknet_pred = self.train_ranknet(self.X_train_scaled, self.y_train, self.X_test_scaled, self.y_test,
                                          **best_ranknet_params)
        ranknet_mse = mean_squared_error(self.y_test, ranknet_pred)
        ranknet_mae = mean_absolute_error(self.y_test, ranknet_pred)
        ranknet_spearman, _ = spearmanr(self.y_test, ranknet_pred)
        tuned_results['Custom RankNet'] = {'MSE': ranknet_mse, 'MAE': ranknet_mae,
                                           'Spearman Correlation': ranknet_spearman}

        # Evaluate tuned LSTM
        lstm_pred = self.train_lstm(self.X_train_scaled, self.y_train, self.X_test_scaled, self.y_test,
                                    **best_lstm_params)
        lstm_mse = mean_squared_error(self.y_test, lstm_pred)
        lstm_mae = mean_absolute_error(self.y_test, lstm_pred)
        lstm_spearman, _ = spearmanr(self.y_test, lstm_pred)
        tuned_results['LSTM'] = {'MSE': lstm_mse, 'MAE': lstm_mae, 'Spearman Correlation': lstm_spearman}

        print("Tuned Model Evaluation Results:")
        for model, metrics in tuned_results.items():
            print(f"\n{model}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")

        print("\nPerforming ensemble prediction...")
        ensemble_pred = self.ensemble_prediction(tuned_models, best_ranknet_params, best_lstm_params)
        ensemble_mse = mean_squared_error(self.y_test, ensemble_pred)
        ensemble_mae = mean_absolute_error(self.y_test, ensemble_pred)
        ensemble_spearman, _ = spearmanr(self.y_test, ensemble_pred)

        print("\nEnsemble Model Results:")
        print(f"  MSE: {ensemble_mse:.4f}")
        print(f"  MAE: {ensemble_mae:.4f}")
        print(f"  Spearman Correlation: {ensemble_spearman:.4f}")

        # comparison of all models including the ensemble
        all_results = {**tuned_results, 'Ensemble': {'MSE': ensemble_mse, 'MAE': ensemble_mae,
                                                     'Spearman Correlation': ensemble_spearman}}

        print("\nFinal Comparison of All Models:")
        for model, metrics in all_results.items():
            print(f"\n{model}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")

        return all_results

if __name__ == "__main__":
    zip_filepath = r'data/Most Streamed Spotify Songs 2024.csv.zip'
    csv_filename = 'Most Streamed Spotify Songs 2024.csv'

    analysis = AdvancedSpotifySongAnalysis(zip_filepath, csv_filename)
    analysis.run_analysis()