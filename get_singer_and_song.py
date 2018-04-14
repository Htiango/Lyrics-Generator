#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-04-14 13:31:00
# @Author  : Huijun Wang 


'''
15688 final project
lyric generator
'''
import os
import json
import requests
import pandas as pd
from bs4 import BeautifulSoup

api = "2d1c084b949c9033f589d692764998b4"

root = "http://api.musixmatch.com/ws/1.1/"

def get_singer(pageNum:int, page_size=100, country = "us"):
	result = []
	all_genres = set()
	for i in range(pageNum):
		param = {
			"apikey":api,
			"country": "country",
			"page": i+1,
			"page_size": page_size,
			"format": "json"
		}

		singers = requests.get(root + "chart.artists.get?", params = param)
		response = json.loads(singers.content)
		artist_list = response.get("message").get("body").get("artist_list")
		
		for artist in artist_list:
			name = artist.get("artist").get('artist_name')
			genres = artist.get("artist").get("primary_genres").get("music_genre_list")
			for g in genres:
				genre = g.get("music_genre").get("music_genre_name")
				genre_id = g.get("music_genre").get("music_genre_id")
				all_genres.add(genre)
				result.append({"artist":name, "genre":genre, "genre_id":genre_id})
	
	df = pd.DataFrame(result)
	df = df.loc[:, ["artist", "genre", "genre_id"]]
	return df, all_genres

def get_songs(artist_df, page_size = 100):

	result = []

	for i, row in artist_df.iterrows(): 
		param = {
				"apikey":api,
				"q_artist": row['artist'],
				"f_music_genre_id": row['genre_id'],
				"f_has_lyrics":"True",
				"page": 1,
				"page_size": page_size
			}

		singer = requests.get(root + "track.search?", params = param)
		response = json.loads(singer.content)
		song_list = response.get("message").get("body").get("track_list")
		for song in song_list:
	
			track_name = song.get("track").get("track_name")
			result.append(
				{
				"artist":row["artist"], 
				"genre":row["genre"], 
				"genre_id":row["genre_id"],
				"track_name":track_name
				})

	df = pd.DataFrame(result)
	df = df.loc[:, ["artist", "genre","genre_id", "track_name"]]
	return df


if __name__ == "__main__":
	# df, all_genres = get_singer(2)
	# df.to_csv("artist_genre.csv", index = False)
	df = pd.read_csv("artist_genre.csv")
	song_df = get_songs(df)
	song_df.to_csv("artist_genre_track.csv", index = False)
	# print(df.head())
	



