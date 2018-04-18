#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-04-14 13:31:00
# @Author  : Huijun Wang 


'''
15688 final project - lyric generator

data collection

retrieve the artists, genres and tracks and export to csv file

API used: musixmatch Developer
documentation: https://developer.musixmatch.com/documentation

'''
import os
import json
import requests
import pandas as pd

with open("../musicmatch_api.key",'r') as f:
		api = f.read()

root = "http://api.musixmatch.com/ws/1.1/"

def get_artist(api, pageNum, page_size=100, country = "us"):

	'''
	getting top artists and their genres
	Args:
		api: API key
		pageNum: the page number for paginated results
		page_size: the page size for paginated results. Range is 1 to 100
		country: country of the artist ranking
	Return:
		df: a pandas dataframe containing artists, genres and genre id
		all_genres: a set of all genres related to the artists found
	'''
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


def get_artist_genre(api, all_artist_list):

	'''
	getting the artists and their genres of given list

	Args:
		api: API key
		all_artist_list: list of all the artists
	Return:
		df: a pandas dataframe containing artists, genres and genre id
		all_genres: a set of all genres related to the artists found

	'''
	result = []
	all_genres = set()
	param = {
			"apikey":api,
			"page":1,
			"page_size":10
		}
	for artist in all_artist_list:
		param["q_artist"] = artist

		search_result = requests.get(root + "artist.search?", params = param)
		response = json.loads(search_result.content)

		artist_list = response.get("message").get("body").get("artist_list")

		if not artist_list:
			continue
		artist_item = artist_list[0]
		
		name = artist_item.get("artist").get('artist_name')
		genres = artist_item.get("artist").get("primary_genres").get("music_genre_list")
		for g in genres:
			genre = g.get("music_genre").get("music_genre_name")
			genre_id = g.get("music_genre").get("music_genre_id")
			all_genres.add(genre)
			result.append({"artist":name, "genre":genre, "genre_id":genre_id})
	
	df = pd.DataFrame(result)

	if not df.empty:
		df = df.loc[:, ["artist", "genre", "genre_id"]]
	else:
		print("result is an empty dataframe")
	return df, all_genres

def get_songs(api, artist_df, page_size = 100):


	'''
	getting track names by artists and genre id

	Args:
		api: API key
		artist_df: dataframe with columns of artist, genre and genre id
		page_size: the page size for paginated results. Range is 1 to 100
	Return:
		df: a pandas dataframe containing artists, genres, genre id and the top
		100 tracks with lyrics under that genre by the artist
		
	'''

	result = []

	for i, row in artist_df.iterrows(): 
		param = {
				"apikey":api,
				"q_artist": row['artist'],
				"f_music_genre_id": row['genre_id'], # filter by genre id
				"f_has_lyrics":"True", # only get tracks with lyrics
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
	# load all artists from csv file
	# artist_df = pd.read_csv("artists.csv", header = None)[:50]
	# artists_list = []
	# for col in artist_df.columns.values:
	# 	artists_list += list(artist_df[col])

	# artist_genre_df, all_genres = get_artist_genre(api, artists_list)
	# artist_genre_df.to_csv("all_artist_genre.csv",index = False)

	artist_df = pd.read_csv("all_artist_genre.csv")[1000:]
	print(artist_df.shape)
	song_df = get_songs(api, artist_df)
	song_df.to_csv("all_artist_genre_tracks.csv", index = False)




