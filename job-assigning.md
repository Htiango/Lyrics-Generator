# Task

## Lyric collecting
 Steps:
 
 + collect singer, song name, genre (W)
 + based on singer and song to require lyric from API (Y)
 + processing (H)

## Sentiment analysis
Exploring:

To analyze the sentiment of some text, do an HTTP POST to http://text-processing.com/api/sentiment/ with form encoded data containing the text we want to analyze. (1000 requests per IP a day)

## Generator
Use LSTM model to generate a random lyric. Each time apply a gaussian distribution to choose the next word. (Until it's to the end.)

## Summary
Summary the generated lyrics. (Using genesis library to do the summary.)

## Classificator
Do a classification to an input lyric to a specific genre. 

  

