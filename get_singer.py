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

def get_singer(pageNum, page_size=100, country = "us"):
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

		singer = requests.get(root + "chart.artists.get?", params = param)
		response = json.loads(singer.content)
		artist_list = response.get("message").get("body").get("artist_list")
		
		for artist in artist_list:
			name = artist.get("artist").get('artist_name')
			genres = artist.get("artist").get("primary_genres").get("music_genre_list")
			for g in genres:
				genre = g.get("music_genre").get("music_genre_name")
				all_genres.add(genre)
				result.append({"artist":name, "genre":genre})
	
	print(len(result))
	df = pd.DataFrame(result)
	df = df.loc[:, ["artist", "genre"]]
	return df, all_genres

# print(response.keys())

if __name__ == "__main__":
	df, all_genres = get_singer(2)

	print(df.shape)








