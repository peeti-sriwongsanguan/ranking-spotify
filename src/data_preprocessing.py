import pandas as pd
import zipfile
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.preprocessing import StandardScaler


def read_zip_file(zip_filepath, csv_filename, encoding='ISO-8859-1'):
    with zipfile.ZipFile(zip_filepath, 'r') as myZip:
        with myZip.open(csv_filename) as myZipCsv:
            df = pd.read_csv(myZipCsv, encoding=encoding)
            df.columns = df.columns.str.replace(' ', '_')
            return df


def clean_numeric_columns(df, columns):
    for column in columns:
        df[column] = df[column].str.replace(',', '')
        df[column] = pd.to_numeric(df[column], errors='coerce')
    return df


def plot_n_drop_missing(df, missing_value=40):
    missing_percent = df.isnull().mean() * 100
    missing_percent = missing_percent.sort_values(ascending=False)

    plt.figure(figsize=(10, 8))
    bars = missing_percent.plot(kind='barh')
    plt.title('Percentage of Missing Values by Column')
    plt.xlabel('Percentage of Missing Values')
    plt.ylabel('Columns')

    for bar in bars.patches:
        plt.text(bar.get_width() + 0.5,
                 bar.get_y() + bar.get_height() / 2,
                 f'{bar.get_width():.2f}%',
                 va='center')

    save_plot("missing_values_plot.png")

    columns_to_drop = missing_percent[missing_percent > missing_value].index

    if not columns_to_drop.empty:
        for column in columns_to_drop:
            print(f"Dropping column '{column}' with {missing_percent[column]:.2f}% missing values.")

    df = df.drop(columns=columns_to_drop)

    if df.duplicated().sum() == 0:
        print('\nThere are no duplicate records')
    else:
        print(f'There are {df.duplicated().sum()} duplicate records')
        df.drop_duplicates(inplace=True)
        print('Duplicate records were removed')

    return df


def plot_correlation_matrix(df, filename='correlation_matrix.png'):
    plt.figure(figsize=(15, 15))
    correlation_matrix = df.select_dtypes(exclude='object').corr().round(decimals=2)
    sns.heatmap(correlation_matrix,
                annot=True,
                cmap='coolwarm',
                fmt='.2f',
                square=True,
                cbar_kws={"shrink": .8},
                linewidths=.5,
                linecolor='white')
    plt.title('Correlation Matrix')
    save_plot(filename)


def save_plot(filename):
    directory = "image"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
    plt.savefig(filepath)
    print(f"Plot saved to {filepath}")


def find_highly_correlated_features(df, threshold=0.70):
    numeric_df = df.select_dtypes(include=[float, int])
    correlation = numeric_df.corr().stack().reset_index()
    correlation.columns = ['feature1', 'feature2', 'corr']
    correlation['corr'] = correlation['corr'].round(3)
    highest_corr = correlation[(correlation['corr'] > threshold) & (correlation['corr'] < 1) & (
            correlation['feature1'] != correlation['feature2'])]
    highest_corr = highest_corr.sort_values('corr', ascending=False).drop_duplicates(['corr'])
    print("Features with correlation greater than", threshold)
    print(highest_corr)
    return highest_corr


def preprocess_data(zip_filepath, csv_filename):
    df = read_zip_file(zip_filepath, csv_filename)

    numeric_columns = ['Spotify_Streams', 'Spotify_Playlist_Count', 'Spotify_Playlist_Reach',
                       'YouTube_Views', 'YouTube_Likes', 'TikTok_Posts', 'TikTok_Likes',
                       'TikTok_Views', 'YouTube_Playlist_Reach', 'AirPlay_Spins', 'Deezer_Playlist_Reach',
                       'Pandora_Streams', 'Pandora_Track_Stations', 'Shazam_Counts']

    df = clean_numeric_columns(df.copy(), numeric_columns)
    pd.set_option('display.float_format', lambda x: '%.f' % x)
    df = plot_n_drop_missing(df)
    df["Release_Year"] = pd.to_datetime(df["Release_Date"]).dt.year
    plot_correlation_matrix(df)
    find_highly_correlated_features(df, threshold=0.70)

    return df


def prepare_features(df, features, target):
    X = df[features]
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y