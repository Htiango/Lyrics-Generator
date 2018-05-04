from lyric_processing import tokenize, remove_stopwords, seperate
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.contrib.keras as kr
import argparse
import pickle

filename_rock = "../csv_files/lyrics_Rock.csv"
filename_pop  = "../csv_files/lyrics_Pop.csv"
filename_rap  = "../csv_files/lyrics_Hip_Hop_Rap.csv"
doc_num = 5000
max_length = 800
categories = ['pop','rock', 'rap']
cat_to_id = dict(zip(categories, range(len(categories))))


def load_lyrics(filename):
    df = pd.read_csv(filename)
    docs = df['lyric'].values
    return docs

def get_raw_data():
    # --------------- load and select data -------------
    lyric_rock = load_lyrics(filename_rock)
    lyric_pop = load_lyrics(filename_pop)
    lyric_rap = load_lyrics(filename_rap)
    P_rap = np.random.permutation(lyric_rap.shape[0])[:doc_num]
    P_rock = np.random.permutation(lyric_rock.shape[0])[:doc_num]
    P_pop = np.random.permutation(lyric_pop.shape[0])[:doc_num]

    lyric_pop_chosen = lyric_pop[P_pop]
    lyric_rap_chosen = lyric_rap[P_rap]
    lyric_rock_chosen = lyric_rock[P_rock]
    lyrics = np.concatenate((lyric_pop_chosen, lyric_rock_chosen, lyric_rap_chosen))

    y_pop = np.array([cat_to_id['pop'] for _ in lyric_pop_chosen])
    y_rock = np.array([cat_to_id['rock'] for _ in lyric_rock_chosen])
    y_rap = np.array([cat_to_id['rap'] for _ in lyric_rap_chosen])
    y = np.concatenate((y_pop, y_rock, y_rap))

    return lyrics, y


def process(args):
    lyrics, y = get_raw_data()

    lyricDocs = seperate(lyrics, False)
    print("Totally %d lyrics."%len(lyricDocs))
    allWords = {}
    for lyricDoc in lyricDocs:
        for word in lyricDoc:
            if word not in allWords:
                allWords[word] = 1
            else:
                allWords[word] += 1

    wordPairs = sorted(allWords.items(), key = lambda x: -x[1])
    words, a= zip(*wordPairs)
    words += (" ", )
    words = ['<PAD>'] + list(words)
    wordToID = dict(zip(words, range(len(words)))) #word to ID
    wordTOIDFun = lambda A: wordToID.get(A, len(words))

    lyricVector = [([wordTOIDFun(word) for word in lyricDoc]) for lyricDoc in lyricDocs]

    x_pad = kr.preprocessing.sequence.pad_sequences(lyricVector, max_length)
    y_pad = kr.utils.to_categorical(y, num_classes=len(cat_to_id))

    data = {'X': x_pad, 'Y': y_pad, 'wordToID': wordToID, 'seq_length': max_length}

    param_saving_path = args.output

    with open(param_saving_path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    
    print('Finish!')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, 
        required=True, help='output path of param.')

    args = parser.parse_args()
    process(args)


if __name__ == "__main__":
    main()

