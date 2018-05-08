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

## Result:
Here are some of the results:<br>

When generating pop lyric:

```
in black vision in the sunshine  
never noticed the end  
 
everything that will come  
it incredible , complete  
it getting with you like a summer song  
i never saw it bloom , i want it  
higher than living until the light is gone  
the way i find a way to shine  
still watch them fade away  
never felt so wrong  
( and i let the sun shine on ( yeh )  
so touch my heart  
( ad - lib )  
 
yeah , i don't wan na lose  
and the one and more  
it an love , make it strong  
you see , but it got me so good  
youre my doubt  
 
we had a price tag darlin  
up against the door  
were down on my knee  
built forever on the chain  
hold your head  
diamond in our heart  
a night , with your dream  
youre taking the breath ( of my body )  
im wicked , yeah  
 
cause i love my way in love  
reaching out to all these stone tonight  
all alone and a trail of the promise  
for what i do  
 
what you can see  
a mirror of me  
complete me  
i am a love i  
livin love  
im on a mission  
like just a better love  
 
im your freak off the light , fire  
going down again  
im gon na dance to a show  
 
gim me what this love bout  
i want to be my only sin  
i can tell  
cause i can feel it happening  
 
it like an ocean liner  
and we are in it cup  
every one working fun , it slow  
i know it too late at night  
it summer a long time  
to the wind on - a figure in sea  
heartbreak getting road to meet  
only sleep tonight  
im a young girl  
i get tongue around on ,  
cause i got better for you  
it got ta never bring back time , ooh  
it over now  
it torture on front  
the end of my mind and it my  
you can be over  
always on my mind  
yeah yeah  
 
( gettin close , under my sheet )  
now im calling out  
if my back so you can't see  
it look like i miss it  
said it o - o - no  
didnt take , doing it but getting back  
never seen letting go ( go )  
how im gon be so early before )  
 
im hooked my name  
but it a natural fact  
im there a much for an last just debris  
it time too much to work out  
 
im going on , yeah  
yeah im gon na turn to you  
to the way it real  
it the rush , my body , now  
i got to get to lose it  
thats the end of my head  
 
walking on broken glass , rocket road to fall  
and then im in my own  
don't ever mess alone  
youre alone with nothing left behind  
and i got thunder on the floor  
found a hero  
but i feel like a living fed my heart  
that make me up and shake  
a i im  
caught up in my foot  
nothing came to me , im in my hand  
 
could come out off you  
oh , oh , woah - oh - oh - honey  
uh - ah  
came over , right now  
 
i got that walkin back of the one  
i got nothing but a soldier with a taste  
whats yesterday any time , im able  
i wasnt really like  
im gon na be in the moment like  
weak and my , my middle  
all for my time , where my mind feel  
im on an island  
slowly , oh , im building  
tomorrow  
 
whoa , i got ta get it back  
i know that i got ta go  
youd never be the same  
im going home  
cause it is hard to forgive my boo  
i got ta make a move , no say ive got  
yeah , the beat drop  
and i don't know about you  
 
but somehow im a sign  
i know there one little strength to beat myself  
but im not the type i feel lost with love  
im twenty foot to my bed , woah  
oh , oh , oh  
 
i don't want praise , i can't stop this really be  
this time if i take it down  
im just takin the breath can you bring my time ``  
cause im all go on a mission , now  
im gon na play this best friend  
and never let this go today  
cause im never gon na go , go and move  
like a bird of a summer night  
and it time it grow high  
and it get harder till we take the sun  
somehow won't make it through  
baby , come back now  
ooh , see i walk on and on the floor  
on my lip and im on top of me  
 
got ta get on this club  
keep , cause he got that  
ooh oh 
```

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
The architecture of the LSTM model is shown below:![LSTM-model-2](https://oh1ulkf4j.qnssl.com/LSTM-model-2.jpg)




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


