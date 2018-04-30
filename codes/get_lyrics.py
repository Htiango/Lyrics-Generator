import pandas as pd
import lyricwikia as ly
import argparse

#request lyric song by song
#row by row in the dataframe
def getLyrics(songs):
    i = -1
    print("Total songs number:" + str(songs.shape[0]))
    for index, row in songs.iterrows():
        i += 1
        if i%100 == 0:
            print("Processing song [" + str(i) + "]")

        song = row['track_name']
        #print(song)
        artist = row['artist']
        try:
            lyric = ly.get_lyrics(artist, song, linesep='\n', timeout=None)
            songs.loc[index,'lyric'] = lyric
        except:
            continue    
        #print(lyric)
    return songs


def run(oriFile, newFile):
    songs = pd.read_csv(oriFile, encoding = "ISO-8859-1")
    songs = getLyrics(songs)
    songs = songs.dropna()
    #print(songs)
    songs.to_csv(newFile)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, 
        required=True, help='input the path of the csv file of the lyric track file.')
    parser.add_argument('-o', '--output', type=str, 
        required=True, help='the output path of the lyric file.')

    args = parser.parse_args()

    run(args.filename, args.output)

if __name__ == "__main__":
    main()


#run('artist_genre_track.csv','lyrics.csv')
