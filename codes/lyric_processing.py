import nltk
from collections import Counter
import numpy as np
import string
import pandas as pd
import argparse

START_MARK = "["
END_MARK = "]"

def seperate(docs_ls):
    docs_raw = [tokenize(START_MARK+doc+END_MARK) for doc in docs_ls]
    docs = remove_stopwords(docs_raw)
    print(" ".join(docs[0]))
    return docs

def remove_stopwords(docs):
    # stopwords=nltk.corpus.stopwords.words('english')
    # stopwords = tokenize(' '.join(stopwords))
    stopwords = get_rare_words(docs)
    stopwords = set(stopwords)
    res = [[word for word in doc if word not in stopwords ] for doc in docs]
    return res


def tokenize(text, lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()):
    """ Normalizes case and handles punctuation
    Inputs:
        text: str: raw text
        lemmatizer: an instance of a class implementing the lemmatize() method
                    (the default argument is of type nltk.stem.wordnet.WordNetLemmatizer)
    Outputs:
        list(str): tokenized text
    """
    text = text.strip()
    text = text.lower()
    # text = text.replace("'s", "")
    text = text.replace("'", "")
    # text = text.replace("\n", ".\n")
    text = text.replace("\t", " ")
    
    punc = string.punctuation
    for c in punc:
        if c in text:
            text = text.replace(c, ' '+c+' ')
    
    tokens = nltk.word_tokenize(text)
    # print(tokens)
    res = []
    
    for token in tokens:
        try:
            word = lemmatizer.lemmatize(token)
            res.append(str(word))
            # if len(word)>1:
            # try:
            #     int(word)
            # except:
            #     res.append(str(word))
        except:
            continue
    docs=nltk.word_tokenize(" ".join(res))
    return res

def get_rare_words(tokens_ls):
    """ use the word count information across all tweets in training data to come up with a feature list
    Inputs:
        processed_tweets: pd.DataFrame: the output of process_all() function
    Outputs:
        list(str): list of rare words, sorted alphabetically.
    """
    counter = Counter([])
    for tokens in tokens_ls:
        counter.update(tokens)
    
    rare_tokes = [k for k,v in counter.items() if v==1]
    rare_tokes.sort()
    return rare_tokes


def process(lyrics, batchSize=10):
    """
    It will change lyrics to vetors as well as build the
    features and labels for LSTM

    lyric: list of str. all of the lyrics
    return: (X, Y, vocab_size, vocab_ID, vocab)
    """

    lyricDocs = seperate(lyrics)
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
    #print(words)
    words += (" ", )
    wordToID = dict(zip(words, range(len(words)))) #word to ID
    wordTOIDFun = lambda A: wordToID.get(A, len(words))

    lyricVector = [([wordTOIDFun(word) for word in lyricDoc]) for lyricDoc in lyricDocs] 

    batchNum = (len(lyrics) - 1) // batchSize 

    X = []
    Y = []

    for i in range(batchNum):
        batchVec = lyricVector[i*batchSize: (i+1)*batchSize]

        maxLen = max([len(vector) for vector in batchVec])

        temp = np.full((batchSize, maxLen), wordTOIDFun(" "),np.int32)

        for j in range(batchSize):
            temp[j, :len(batchVec[j])] = batchVec[j]

        X.append(temp)

        temp_copy = np.copy(temp)
        temp_copy[:, :-1] = temp[:, 1:]

        Y.append(temp_copy)

    return X, Y, len(words) + 1, wordToID, words


def generate_feature(args):
    filename = args.filename
    ouput_path = args.output

    df = pd.read_csv(filename)

    docs = df['lyric'].values.tolist()[:100]
    print(docs[0])
    print()
    print()
    X, Y, size, wordToId, words = process(docs)
    print(size)
    print(X[0][0].shape)
    # print(type(docs)) 

def pretreatment(filename):
    df = pd.read_csv(filename)
    docs = df['lyric'].values
    P = np.random.permutation(len(docs))
    docs = docs[P]
    # docs = df['lyric'].values.tolist()
    return process(docs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, 
        required=True, help='input the path of the csv file of the lyric.')
    parser.add_argument('-o', '--output', type=str, 
        required=True, help='the output path of the all the required parameter.')

    args = parser.parse_args()

    generate_feature(args)

if __name__ == "__main__":
    main()

