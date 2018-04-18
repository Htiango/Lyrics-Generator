import pandas as pd
import lyricwikia as ly


#request lyric song by song
#row by row in the dataframe
def getLyrics(songs):
	i = -1
	for index, row in songs.iterrows():
		i += 1
		if i==100:
			break

		song = row['track_name']
		#print(song)
		artist = row['artist']
		try:
			lyric = ly.get_lyrics(artist, song, linesep='\n', timeout=None)
			songs.loc[index,'lyric'] = lyric
		except ly.LyricsNotFound:
			continue	
		#print(lyric)
	return songs


def run(oriFile, newFile):
	songs = pd.read_csv(oriFile)
	songs = getLyrics(songs)
	songs = songs.dropna()
	#print(songs)
	songs.to_csv(newFile)



#run('artist_genre_track.csv','lyrics.csv')
