import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
from data_preprocessing import read_zip_file, clean_numeric_columns, clean_data


class SpotifySongAnalysis:
    def __init__(self, zip_filepath, csv_filename):
        self.zip_filepath = zip_filepath
        self.csv_filename = csv_filename
        self.df = None
        self.df_cleaned = None
        self.X = None
        self.y = None
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4

    def load_and_preprocess_data(self):
        # Read and preprocess data
        self.df = read_zip_file(self.zip_filepath, self.csv_filename)
        numeric_columns = [
            'Spotify_Streams', 'Spotify_Playlist_Count', 'Spotify_Playlist_Reach',
            'YouTube_Views', 'YouTube_Likes', 'TikTok_Posts', 'TikTok_Likes',
            'TikTok_Views', 'YouTube_Playlist_Reach', 'AirPlay_Spins', 'Deezer_Playlist_Reach',
            'Pandora_Streams', 'Pandora_Track_Stations', 'Shazam_Counts', 'Track_Score'
        ]
        self.df = clean_numeric_columns(self.df, numeric_columns)
        self.df['All_Time_Rank'] = self.df['All_Time_Rank'].str.replace(',', '').astype('int')

        self.df_cleaned = clean_data(self.df, [
            'YouTube_Playlist_Reach', 'YouTube_Views', 'YouTube_Likes', 'Track_Score',
            'Spotify_Streams', 'Spotify_Playlist_Count', 'Spotify_Playlist_Reach',
            'TikTok_Posts', 'TikTok_Likes', 'TikTok_Views', 'All_Time_Rank'
        ])

    def prepare_features_and_target(self):
        # Prepare features and target
        self.X = self.df_cleaned[['YouTube_Views', 'YouTube_Likes', 'YouTube_Playlist_Reach','Spotify_Streams'
                                  , 'Spotify_Playlist_Count', 'Spotify_Playlist_Reach', 'TikTok_Posts','TikTok_Views']]
        self.y = self.df_cleaned['All_Time_Rank']

        # Split data and scale features
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                random_state=42)
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)

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
        features = ['YouTube_Views', 'YouTube_Likes', 'Spotify_Streams', 'TikTok_Views']

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Feature Distributions', fontsize=20, y=1.02)

        for ax, feature in zip(axes.ravel(), features):
            sns.histplot(self.df[feature], kde=True, ax=ax)
            ax.set_title(f'Distribution of {feature}', fontsize=14)
            ax.set_xlabel(feature, fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_xscale('log')

        plt.tight_layout()
        plt.show()

    def train_and_evaluate_models(self):
        models = {
            'Ranking SVM': LinearSVR(random_state=42),
            'RankNet': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        }

        results = {}
        for name, model in models.items():
            model.fit(self.X_train_scaled, self.y_train)
            y_pred = model.predict(self.X_test_scaled)
            mse = mean_squared_error(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            spearman_corr, _ = spearmanr(self.y_test, y_pred)
            results[name] = {'MSE': mse, 'MAE': mae, 'Spearman Correlation': spearman_corr}

        return results

    def predict_top_n_efficiently(self, n=100):
        # Train on a subset of data for efficiency
        X_subset = self.X_train_scaled[:10000]  # Adjust based on your computational resources
        y_subset = self.y_train[:10000]

        model = LinearSVR(random_state=42)
        model.fit(X_subset, y_subset)

        # Predict on the entire dataset
        all_predictions = model.predict(self.X_test_scaled)

        # Get top N predictions
        top_n_indices = np.argsort(all_predictions)[:n]
        top_n_predictions = all_predictions[top_n_indices]
        top_n_actual = self.y_test.iloc[top_n_indices]

        return top_n_predictions, top_n_actual

    def run_analysis(self):
        self.load_and_preprocess_data()
        self.prepare_features_and_target()

        print("Plotting metric correlations...")
        self.plot_metric_correlations()

        print("Plotting feature distributions...")
        self.plot_feature_distributions()

        print("Training and evaluating models...")
        model_results = self.train_and_evaluate_models()

        print("Model Evaluation Results:")
        for model, metrics in model_results.items():
            print(f"\n{model}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")

        print("\nPredicting top 100 rankings...")
        top_100_pred, top_100_actual = self.predict_top_n_efficiently(100)
        print("\nTop 100 Predicted vs Actual Rankings:")
        for pred, actual in zip(top_100_pred, top_100_actual):
            print(f"Predicted: {pred:.0f}, Actual: {actual}")


if __name__ == "__main__":
    zip_filepath = r'data/Most Streamed Spotify Songs 2024.csv.zip'
    csv_filename = 'Most Streamed Spotify Songs 2024.csv'

    analysis = SpotifySongAnalysis(zip_filepath, csv_filename)
    analysis.run_analysis()