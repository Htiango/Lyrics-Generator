# Lyrics generator
The purpose of the project is to automatically generate a lyric based on the genre you choose. The generated lyric will be verified by our lyric classification model. You can also get a sentiment analysis of the lyric right after its generated.

The method we use to generate lyrics is LSTM model. 

And the method we use to classify lyric into genre is CNN model.


## Generate lyrics
Go to directory `codes/`, run script `generator.py` with the following command:

```bash
python3 generator.py -g [pop/rock/rap]
```

Then the generated lyric will be print on the console.

And our classification model will verify the genre of the generated lyric.

## Project Steps
(If you want to start from the beginning) 


### Step 1. Data collection
#### Part 1. Singer and song collection
In order to train the lyric model, the first step is to collect lyrics by genre. We collected the male and female artists' names from [music.163.com ](https://music.163.com/#/discover/artist/cat?id=2001)by copying the information on the webpage and saved them as csv files.

After getting artists' names, we use the [musixmatch](http://api.musixmatch.com/ws/1.1/) api to collect the genre and name of songs of the artists. The results are exported as csv file so that we can count the most frequent genres among all the songs we collected. 

We already provide the lyric track files in `./csv/`

#### Part 2. Lyrics collection via *lyricwikia*

With all the song names, we use [lyricwikia](https://github.com/enricobacis/lyricwikia) package in Python to collect the lyrics. The package can be installed with pip.

```python
pip3 install lyricwikia
```

You can run in command line in `./codes/` directory:

```bash
python3 get_lyrics.py -h
```
to choose the csv file of the lyric track and customize the output path of the lyric file. 

From the csv file of songs and their genres, we found the top 3 genres are:

* Pop
* Hip Hop/Rap
* Rock

We will train the model based on these three genres. Therefore, we will extract and generate the dataset of lyrics of each genre.


We are now ready to train the model with 3 datasets consisting of lyrics of different genres. 

And the number of lyrics in each generes are:

| genre | lyric number |
|---|---|
| rap | 5039 |
| pop | 21998 |
| rock | 6503 | 

*The singer, song and lyric files are all stored in `./csv_files/` directory*

### Step 2. Data Preprocessing

Later we'll apply two different deep learning methods:
+ LSTM model to generate chosen genre of lyrics
+ CNN model to classify a lyric into specific genre

Since these two methods need different preprocessing, here we divide data preprocessing part into 2 part. 


#### Part 1. Preprocessing for LSTM model:
For LSTM model, it is important to know the start and the end of a sentense. So here in the preprocessing stage, we manually add a start mark and an end mark to each lyric. And then use `nltk` package to tokenize lyrics into words and stem them. Then remove all the rare words.   <br>
The most important method below is the `process` method, which generates all the features needed by the LSTM model. The returning X represents the word id sequences in each batch size lyrics. The Y is almost the same as X, except it is actually X moving 1 word to the right. And we also need a word to Id dict so that we can transform generated ids into words in the test stage. 

Since the processing is a little bit time-consuming, we save the result into a pickle file to make it easier for us to testing LSTM model. We use the following code to save preprocessed LSTM data.

You can run in command line in `./codes/` directory:
```bash
python3 save_data.py -h
```
to choose the input raw data and customize your output file name. 

**Please Remember:**<br>
The saved data is too big to upload to github, the pickle files in `./codes/generate-param/` is only used for testing. If you want to do the training, you have to run `save_data.py` to generate needed pickle files. 


#### Part 2. Preprocessing for CNN model.
Unlike the LSTM model, the start and the end is not important in CNN model. <br>
We use the same method in Part 1. to tokenize lyrics and generate a vocabulary. And then we pad each lyrics into the longest (or our chosen) length. (Here we pad using the mark `<PAD>`) Then we save parameters into a pickle file. 


You can run in command line in `./codes/` directory:
```bash
python3 classification_preprocess.py -h
```
to customize your output parameter pickle file name and path.  

**Please Remember:**<br>
The saved data is too big to upload to github, the pickle files in `./codes/generate-param/` is only used for testing. If you want to do the training, you have to run `classification_preprocess.py` to generate needed pickle files. 


### Step 3. Training Models
#### Part 1. LSTM model
Here we use LSTM model to generate lyrics for different genres. [This blog](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) clearly states the knowledge of LSTM. <br>
For each genres, we do preprocessing and save the needed parameters into pickle files. Then we load the specific pickle file and train the LSTM model for the genre. <br>
The word IDs will be embedded into a dense representation before feeding to the LSTM, which is called embedding layer. Here we use 2 layers of LSTM to process the data, followed by softmax representing each word's appearing probability. <br>
In the training stage, we do 100 epoches. Since the data we use is very large and LSTM model is very slow to train. Here we use AWS to train 3 models for 3 genres. Even on AWS GPU server, it took nearly 40 hours to train. (40+ hours to train for pop, 18+ hours to train for rap and 10+ hours to train for rock). <br>
The models are saved in:
+ rap model:  `./codes/checkpoints/rap/` 
+ rock model: `./codes/checkpoints/rock/`
+ pop model:  `./codes/checkpoints/pop/`


You can also run in command line in `./codes/` directory:
```bash
python3 main.py -h
```
to choose training or testing the LSTM mode.



#### Part 2. CNN model
Here we use the method mentioned in [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) <br>
The architecture of the model is listed as below, which is taken from the above article. 
![CNN model](https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2017/08/Example-of-a-CNN-Filter-and-Polling-Architecture-for-Natural-Language-Processing.png)

In this project, I set the word embedding dimension to be 600 and each sequence length to be 800. (If not satisfied we add `<PAD>` in the front. ) We choose 256 convolution filters and each size is 5 followed by a max-over-time polling. Then we use a fully connected layers with drop out and ReLU. And finally use softmax to do the classification. (Here we do 3-class classification: the 3 genres mentioned above).

The training processing is in `classification.ipynb`, where prints out the details in training. <br>
You can also run in command line in `./codes/` directory:
```bash
python3 classification_model.py -h
```
to choose training or testing mode. 

The training loss figure is:
![Screen Shot 2018-05-07 at 9.25.54 PM](https://oh1ulkf4j.qnssl.com/Screen%20Shot%202018-05-07%20at%209.25.54%20PM.png)

And the training accuracy figure is:
![Screen Shot 2018-05-07 at 9.25.40 PM](https://oh1ulkf4j.qnssl.com/Screen%20Shot%202018-05-07%20at%209.25.40%20PM.png)

The above figures are recorded by tensorboard.

The saving model's accuracy on testing dataset is 77.40%.


The model is saved in `./codes/checkpoints/textcnn/`

### Step 4. Sentiment analysis

To analyze the sentiment of some text, do an HTTP POST to http://text-processing.com/api/sentiment/ with form encoded data containing the text we want to analyze. (1000 requests per IP a day)

You can run in command line in `./codes/` directory:
```bash
python3 sentiment_analysis.py -h 
```
to test an input sentense's sentiment.

### Steps 5. Display Result (It's show time!)

We use AWS server to train 3 LSTM lyric generator models for 3 genres and train the CNN classification model locally. With those saving models, now we can use our LSTM model to generate lyric in chosen genre. And then use our CNN classification model to test the result. Finally do a sentiment analysis for the generated lyric.

In order to get lyric in random, instead of selecting the word with the highest probability, I map the probability to an interval and randomly sample one. See in `probsToWord` method in Step3 part1. (Of course each lyric starts with the starting mark `[`) 

You are recommended to run in command line in `./codes/` directory:
```bash
python3 generator.py -g [pop/rock/rap]
```
to generate a chosen genre lyric and verify in our classification model as well as getting a sentiment analysis.


