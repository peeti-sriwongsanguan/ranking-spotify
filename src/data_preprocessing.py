import pandas as pd
import zipfile
import matplotlib.pyplot as plt
import os
import seaborn as sns


#open and read the file
def read_zip_file(zip_filepath, csv_filename, encoding='ISO-8859-1'):
    with zipfile.ZipFile(zip_filepath, 'r') as myZip:
        with myZip.open(csv_filename) as myZipCsv:
            df = pd.read_csv(myZipCsv, encoding=encoding)
            # Replace spaces with underscores in column names
            df.columns = df.columns.str.replace(' ', '_')
            return df

#start EDA process

# clean numeric columns
def clean_numeric_columns(df, columns):
    for column in columns:
        # Remove commas and convert to numeric
        df[column] = df[column].str.replace(',', '')
        df[column] = pd.to_numeric(df[column], errors='coerce')
    return df

# Plotting missing values and dropping columns with >40% missing
def plot_n_drop_missing(df, missing_value = 40):
    # Calculate the percentage of missing values in each column
    missing_percent = df.isnull().mean() * 100

    # Sort the missing percentages in descending order
    missing_percent = missing_percent.sort_values(ascending=False)

    # Plot the missing values with horizontal bars
    plt.figure(figsize=(10, 8))
    bars = missing_percent.plot(kind='barh')
    plt.title('Percentage of Missing Values by Column')
    plt.xlabel('Percentage of Missing Values')
    plt.ylabel('Columns')

    # Annotate each bar with the percentage value (aligned horizontally)
    for bar in bars.patches:
        plt.text(bar.get_width() + 0.5,  # Position the text slightly beyond the bar
                 bar.get_y() + bar.get_height() / 2,  # Vertically center the text
                 f'{bar.get_width():.2f}%',
                 va='center')

    # Save the plot
    save_plot("missing_values_plot.png")

    # Identify columns to drop
    columns_to_drop = missing_percent[missing_percent > missing_value].index

    # Print the columns being dropped and the reason
    if not columns_to_drop.empty:
        for column in columns_to_drop:
            print(f"Dropping column '{column}' with {missing_percent[column]:.2f}% missing values.")

    # Drop columns with more than 40% missing values
    df = df.drop(columns=columns_to_drop)

    #check if there is any duplicate record
    if df.duplicated().sum() == 0:
        print('\nThere is no duplicate records')
    else:
        print(f'There are {df.duplicated().sum()} duplicate records')
        df.drop_duplicates(inplace=True)
        print('Now duplicate records were removed')


    return df

# plot and save the correlation matrix
def plot_correlation_matrix(df, filename='correlation_matrix.png'):
    plt.figure(figsize=(15, 15))
    correlation_matrix = df.select_dtypes(exclude='object').corr().round(decimals=2)
    sns.heatmap(correlation_matrix,
                annot=True,
                cmap='coolwarm',
                fmt='.2f',
                square=True,
                cbar_kws={"shrink": .8},
                linewidths = .5,
                linecolor='white')
    plt.title('Correlation Matrix')

    # Save the plot
    save_plot(filename)


# Function to handle saving the plot to the "image" folder
def save_plot(filename):
    # Define the directory
    directory = "image"

    # Create the directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Define the full path
    filepath = os.path.join(directory, filename)

    # If the file exists, remove it
    if os.path.exists(filepath):
        os.remove(filepath)

    # Save the figure
    plt.savefig(filepath)
    print(f"Plot saved to {filepath}")

# find and display highly correlated features
def find_highly_correlated_features(df, threshold=0.70):
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=[float, int])

    correlation = numeric_df.corr().stack().reset_index()
    correlation.columns = ['feature1', 'feature2', 'corr']

    # Round correlation values to 3 decimal places
    correlation['corr'] = correlation['corr'].round(3)

    # Find pairs of features with correlation above the threshold, excluding self-correlations and correlations of exactly 1
    highest_corr = correlation[(correlation['corr'] > threshold) & (correlation['corr'] < 1) & (
                correlation['feature1'] != correlation['feature2'])]

    # Drop duplicate correlations (e.g., feature1-feature2 and feature2-feature1 are the same)
    highest_corr = highest_corr.sort_values('corr', ascending=False).drop_duplicates(['corr'])

    print("Features with correlation greater than", threshold)
    print(highest_corr)

    return highest_corr


pd.set_option('display.max_columns', None)
def main():
    zfile = r'data/Most Streamed Spotify Songs 2024.csv.zip'
    csv_filename = 'Most Streamed Spotify Songs 2024.csv'
    # Step 1: load data
    df = read_zip_file(zfile, csv_filename)
    # Step 2: Clean numeric columns
    numeric_columns = ['Spotify_Streams', 'Spotify_Playlist_Count', 'Spotify_Playlist_Reach',
                       'YouTube_Views', 'YouTube_Likes', 'TikTok_Posts', 'TikTok_Likes',
                       'TikTok_Views', 'YouTube_Playlist_Reach', 'AirPlay_Spins', 'Deezer_Playlist_Reach',
                       'Pandora_Streams', 'Pandora_Track_Stations', 'Shazam_Counts']

    df = clean_numeric_columns(df.copy(), numeric_columns)

    # Disable scientific notation for display
    pd.set_option('display.float_format', lambda x: '%.f' % x)
    # Step 3: remove missing value
    df = plot_n_drop_missing(df)


    # Step 4: Extract year from the release date
    df["Release_Year"] = pd.to_datetime(df["Release_Date"]).dt.year

    # Step 5: Plot the correlation matrix
    plot_correlation_matrix(df)

    # Step 6: Find and display highly correlated features with a threshold of 0.70
    find_highly_correlated_features(df, threshold=0.70)


    return df


if __name__ == '__main__':
    df = main()

# print(df.head())
# print(df.info())