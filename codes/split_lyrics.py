import numpy as np
import pandas as pd

def split_lyrics(csv_path):	
	df = pd.read_csv(csv_path)

	df = df.iloc[:,1:]

	result = []
	for genre in ['R&B/Soul']:

		result.append(df[df['genre'] == genre])

	# rock = df[df['genre'] == "Rock"]

	# randr = df[df['genre'] == "Rock & Roll"]
	# randr.loc[:,'genre'] = 'Rock'

	# r = pd.concat([rock, randr])
	# result.append(r)

	return result



if __name__ == "__main__":
	df_female = split_lyrics('../csv_files/all_female_artist_lyrics.csv')
	df_male = split_lyrics('../csv_files/all_male_artist_lyrics.csv')

	for d, f in zip(df_female,df_male):
		genre = d.iloc[0,1]
		genre = genre.replace(" ", "_").replace("/", "_")
		df = pd.concat([d,f])
		df.to_csv('../csv_files/lyrics_' + genre +".csv", index = False)
