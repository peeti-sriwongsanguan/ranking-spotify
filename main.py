from src.data_preprocessing import read_zip_file

zfile = r'data/Most Streamed Spotify Songs 2024.csv.zip'
# Assuming the CSV inside the ZIP is named 'Most Streamed Spotify Songs 2024.csv'
csv_filename = 'Most Streamed Spotify Songs 2024.csv'


def main():
    df = read_zip_file(zfile, csv_filename)

    return df


if __name__ == '__main__':
    df = main()



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
