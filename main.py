import src.data_preprocessing as dp
from src.model import get_models, train_model

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Data Preprocessing
    zip_filepath = 'data/Most Streamed Spotify Songs 2024.csv.zip'
    csv_filename = 'Most Streamed Spotify Songs 2024.csv'

    df = dp.preprocess_data(zip_filepath, csv_filename)

    # Feature Selection and Target Variable
    features = ['Spotify_Streams', 'Spotify_Playlist_Count', 'Spotify_Playlist_Reach', 'Spotify_Popularity',
                'YouTube_Views', 'YouTube_Likes', 'TikTok_Posts', 'TikTok_Likes', 'TikTok_Views']
    target = 'All_Time_Rank'

    X_scaled, y = dp.prepare_features(df, features, target)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Get models
    models = get_models(X_train.shape[1])

    results = {}

    for name, model in models.items():
        if name in ['Random Forest', 'Gradient Boosting', 'SVR', 'Elastic Net', 'KNN']:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
        else:
            mse, r2 = train_model(model, X_train, y_train, X_test, y_test)

        results[name] = {'MSE': mse, 'R2': r2}

    # Plot results
    # plt.figure(figsize=(12, 6))
    # mse_values = [result['MSE'] for result in results.values()]
    # r2_values = [result['R2'] for result in results.values()]

    # plt.subplot(1, 2, 1)
    # plt.bar(results.keys(), mse_values)
    # plt.title('Mean Squared Error')
    # plt.xticks(rotation=45, ha='right')
    #
    # plt.subplot(1, 2, 2)
    # plt.bar(results.keys(), r2_values)
    # plt.title('R-squared Score')
    # plt.xticks(rotation=45, ha='right')
    #
    # plt.tight_layout()
    # dp.save_plot("model_comparison.png")


if __name__ == '__main__':
    df = main()


