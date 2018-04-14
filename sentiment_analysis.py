import requests
import argparse
import json

"""
This script is a demo of how to get sentiment analysis from an online API.
Remember each IP can only request 1000 times per day.
If you want to exceed the limitation, you might need to automatic change 
the IP in case of failure.
"""

URL = "http://text-processing.com/api/sentiment/"
prefix = "text="

def get_sentiment(args):
	text = args.text
	param = prefix + text
	response = requests.post(URL, param)
	res = json.dumps(response.json())
	print(res)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--text', type=str, 
        required=True, help='input text you want to do sentiment analysis.')  

    args = parser.parse_args()

    get_sentiment(args)

if __name__ == "__main__":
    main()