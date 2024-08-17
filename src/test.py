numeric_columns = ['Spotify Streams','Spotify Playlist Count','Spotify Playlist Reach',
                       'YouTube Views', 'YouTube Likes','TikTok Posts', 'TikTok Likes',
                       'TikTok Views','YouTube Playlist Reach','AirPlay Spins','Deezer Playlist Reach',
                       'Pandora Streams','Pandora Track Stations','Shazam Counts']

numeric_col = [i.replace(' ','_') for i in numeric_columns]
print(numeric_col)