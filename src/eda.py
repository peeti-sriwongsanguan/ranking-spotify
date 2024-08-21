from data_preprocessing import read_zip_file, clean_numeric_columns, clean_data
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


zip_filepath = r'data/Most Streamed Spotify Songs 2024.csv.zip'
csv_filename = 'Most Streamed Spotify Songs 2024.csv'

df = read_zip_file(zip_filepath, csv_filename)

numeric_columns = ['Spotify_Streams', 'Spotify_Playlist_Count', 'Spotify_Playlist_Reach',
                   'YouTube_Views', 'YouTube_Likes', 'TikTok_Posts', 'TikTok_Likes',
                   'TikTok_Views', 'YouTube_Playlist_Reach', 'AirPlay_Spins', 'Deezer_Playlist_Reach',
                   'Pandora_Streams', 'Pandora_Track_Stations', 'Shazam_Counts', 'Track_Score']

df = clean_numeric_columns(df.copy(), numeric_columns)


pd.set_option('display.float_format', lambda x: '%.f' % x)



df['All_Time_Rank'] = df['All_Time_Rank'].str.replace(',', '').astype('int')


df_cleaned = clean_data(df, ['YouTube_Playlist_Reach','YouTube_Views', 'YouTube_Likes','Track_Score',
                             'Spotify_Streams', 'Spotify_Playlist_Count', 'Spotify_Playlist_Reach',
                             'TikTok_Posts', 'TikTok_Likes','TikTok_Views','All_Time_Rank'])
# df2 = pd.concat([df_cleaned,df[['All_Time_Rank']]], axis=0)
print(df_cleaned.head())
# # Define features and target
# X = df[['YouTube_Views', 'YouTube_Likes']]
# y = df['Track_Score']
#
# # Fit the k-NN model
# knn = KNeighborsRegressor(n_neighbors=3)
# knn.fit(X, y)
#
# # Generate predictions for plotting
# x_grid = np.linspace(X['YouTube_Views'].min(), X['YouTube_Views'].max(), 100)
# y_grid = np.linspace(X['YouTube_Likes'].min(), X['YouTube_Likes'].max(), 100)
# X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
# Z_grid = knn.predict(np.c_[X_grid.ravel(), Y_grid.ravel()]).reshape(X_grid.shape)
#
# # Plotting
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X['YouTube_Views'], X['YouTube_Likes'], y, color='red', label='Data Points')
# ax.plot_surface(X_grid, Y_grid, Z_grid, color='blue', alpha=0.5)
#
# ax.set_xlabel('YouTube Views')
# ax.set_ylabel('YouTube Likes')
# ax.set_zlabel('Track Score')
# ax.set_title('k-NN Regression: Track Score vs YouTube Views & Likes')
# plt.legend()
# plt.show()


# Scatter plot for Track_Score vs All_Time_Rank
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
sns.scatterplot(x='TikTok_Posts', y='Track_Score', data=df)
plt.title('Track Score vs TikTok_Posts')

# Scatter plot for YouTube_Views vs All_Time_Rank
plt.subplot(1, 3, 2)
sns.scatterplot(x='TikTok_Views', y='Track_Score', data=df)
plt.title('Track Score vs TikTok Views')


# Scatter plot for YouTube_Likes vs All_Time_Rank
plt.subplot(1, 3, 3)
sns.scatterplot(x='TikTok_Likes', y='Track_Score', data=df)
plt.title('Track Score vs TikTok Likes')

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
sns.scatterplot(x='YouTube_Playlist_Reach', y='Track_Score', data=df)
plt.title('Track Score vs YouTube_Playlist_Reach')

# Scatter plot for YouTube_Views vs All_Time_Rank
plt.subplot(1, 3, 2)
sns.scatterplot(x='YouTube_Views', y='Track_Score', data=df)
plt.title('Track Score vs YouTube Views')


# Scatter plot for YouTube_Likes vs All_Time_Rank
plt.subplot(1, 3, 3)
sns.scatterplot(x='YouTube_Likes', y='Track_Score', data=df)
plt.title('Track Score vs YouTube Likes')

plt.tight_layout()
plt.show()

# Scatter plot for Track_Score vs All_Time_Rank
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
sns.scatterplot(x='Spotify_Streams', y='Track_Score', data=df)
plt.title('Track Score vs Spotify_Streams')

# Scatter plot for YouTube_Views vs All_Time_Rank
plt.subplot(1, 3, 2)
sns.scatterplot(x='Spotify_Playlist_Count', y='Track_Score', data=df)
plt.title('Track Score vs Spotify_Playlist_Count')


# Scatter plot for YouTube_Likes vs All_Time_Rank
plt.subplot(1, 3, 3)
sns.scatterplot(x='Spotify_Playlist_Reach', y='Track_Score', data=df)
plt.title('Track Score vs Spotify_Playlist_Reach Likes')

plt.tight_layout()
plt.show()

# df.head().to_clipboard()
# df["Release_Year"] = pd.to_datetime(df["Release_Date"]).dt.year
#
#
# features = ['Spotify_Streams', 'Spotify_Playlist_Count', 'Spotify_Playlist_Reach', 'Spotify_Popularity',
#             'YouTube_Views', 'YouTube_Likes', 'TikTok_Posts', 'TikTok_Likes', 'TikTok_Views']

import pandas as pd
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel

# Preparing the data for regression
X = df_cleaned[['YouTube_Views', 'YouTube_Likes']]
y = df_cleaned['All_Time_Rank']

# Fit the ordinal regression model (without adding a constant)
model = OrderedModel(y, X, distr='logit')
result = model.fit(method='bfgs')

# Summary of the model
print(result.summary())

# Predicting the All_Time_Rank
predicted_ranks = result.predict(X)
print(predicted_ranks)