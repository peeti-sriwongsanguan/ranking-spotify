import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from src.data_preprocessing import preprocess_data, save_plot
from src.model import get_models, train_model


def main():
    # Data Preprocessing
    zip_filepath = r'data/Most Streamed Spotify Songs 2024.csv.zip'
    csv_filename = 'Most Streamed Spotify Songs 2024.csv'

    X, y, features = preprocess_data(zip_filepath, csv_filename)

    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)
    print("Sample of X:", X[:5])
    print("Sample of y:", y[:5])

    # Check for NaN or infinity values
    if np.isnan(X).any() or np.isinf(X).any():
        print("Warning: X contains NaN or infinity values")
    if np.isnan(y).any() or np.isinf(y).any():
        print("Warning: y contains NaN or infinity values")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Get models
    models = get_models(X_train.shape[1])

    # Train and evaluate models
    results = {}

    for name, model in models.items():
        print(f"Training {name}...")
        try:
            if name in ['Random Forest', 'Gradient Boosting', 'SVR', 'Elastic Net', 'KNN']:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
            else:
                mse, r2 = train_model(model, X_train, y_train, X_test, y_test)

            results[name] = {'MSE': mse, 'R2': r2}
            print(f"{name} trained successfully. MSE: {mse:.4f}, R2: {r2:.4f}")
        except Exception as e:
            print(f"Error training {name}: {str(e)}")

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
    save_plot("model_comparison.png")

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