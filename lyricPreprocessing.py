import nltk
from collections import Counter
import numpy as np
import string

START_MARK = "["
END_MARK = "]"

def seperate(docs_ls):
    docs_raw = [tokenize(doc) for doc in docs_ls]
    docs = remove_stopwords(docs_raw)
    add_mark(docs)
    return docs

def remove_stopwords(docs):
    stopwords=nltk.corpus.stopwords.words('english')
    stopwords = tokenize(' '.join(stopwords))
    # stopwords.extend(get_rare_words(docs))
    stopwords = set(stopwords)
    res = [[word for word in doc if word not in stopwords ] for doc in docs]
    return res

def add_mark(docs):
    for doc in docs:
        doc.append(END_MARK)
        doc.insert(0, START_MARK)

def tokenize(text, lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()):
    """ Normalizes case and handles punctuation
    Inputs:
        text: str: raw text
        lemmatizer: an instance of a class implementing the lemmatize() method
                    (the default argument is of type nltk.stem.wordnet.WordNetLemmatizer)
    Outputs:
        list(str): tokenized text
    """
    text = text.lower()
    text = text.replace("'s", '')
    text = text.replace("'", '')
    
    punc = string.punctuation
    for c in punc:
        if c in text:
            text = text.replace(c, ' ')
    
    tokens = nltk.word_tokenize(text)
#     print(tokens)
    res = []
    
    for token in tokens:
        try:
            word = lemmatizer.lemmatize(token)
            if len(word)>1:
                try:
                    int(word)
                except:
                    res.append(str(word))
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


def process(lyrics, batchSize=5):
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

docs = """ I'm hurting, baby, I'm broken down
I need your loving, loving, I need it now
When I'm without you
I'm something weak
You got me begging
Begging, I'm on my knees
I don't wanna be needing your love
I just wanna be deep in your love
And it's killing me when you're away
Ooh, baby,
'Cause I really don't care where you are
I just wanna be there where you are
And I gotta get one little taste
Your sugar
Yes, please
Won't you come and put it down on me
I'm right here, 'cause I need
Little love and little sympathy
Yeah you show me good loving
Make it alright
Need a little sweetness in my life
Your sugar
Yes, please
Won't you come and put it down on me
My broken pieces
You pick them up
Don't leave me hanging, hanging
Come give me some
When I'm without ya
I'm so insecure
You are the one thing
The one thing, I'm living for
I don't wanna be needing your love
I just wanna be deep in your love
And it's killing me when you're away
Ooh, baby,
'Cause I really don't care where you are
I just wanna be there where you are
And I gotta get one little taste
Your sugar
Yes, please
Won't you come and put it down on me
I'm right here, 'cause I need
Little love and little sympathy
Yeah you show me good loving
Make it alright
Need a little sweetness in my life
Your sugar (your sugar)
Yes, please (yes, please)
Won't you come and put it down on me
Yeah
I want that red velvet
I want that sugar sweet
Don't let nobody touch it
Unless that somebody's me
I gotta be a man
There ain't no other way
'Cause girl you're hotter than southern California Bay
I don't wanna play no games
I don't gotta be afraid
Don't give all that shy shit
No make up on, that's my
Sugar
Yes, please
Won't you come and put it down on me (down on me)
Oh, right here (right here),
'Cause I need (I need)
Little love and little sympathy
Yeah you show me good loving
Make it alright
Need a little sweetness in my life
Your sugar (sugar)
Yes, please (yes, please)
Won't you come and put it down on me
Your sugar
Yes, please
Won't you come and put it down on me
I'm right here, 'cause I need
Little love and little sympathy
Yeah you show me good loving
Make it alright
Need a little sweetness in my life
Your sugar
Yes, please
Won't you come and put it down on me
"""

print(process(docs.split('\n')))

