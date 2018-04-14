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
from bs4 import BeautifulSoup

api = "2d1c084b949c9033f589d692764998b4"

root = "http://api.musixmatch.com/ws/1.1/"

param = {
	"apikey":api,
	"country": "us",
	"page": 1,
	"page-size": 20,
	"format": "json"
}

singer = requests.get(root + "chart.artists.get?", params = param)
response = json.loads(singer.content)
artist_list = response.get("message").get("body").get("artist_list")
print(artist_list[0].get("artist"))
# print(response.keys())










