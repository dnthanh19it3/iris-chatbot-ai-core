import urllib.request, json
import os
from dotenv import load_dotenv, find_dotenv, dotenv_values

#GET ENV VARIABLE
load_dotenv('.env')
BASE_URL = os.environ.get("BASE_URL")

def getJson(project_id='1'):
    urlIntent = "{}console/get-dataset/{}".format(BASE_URL, project_id)
    with urllib.request.urlopen(urlIntent) as url:
        data = json.loads(url.read().decode())
        return data

def getStaticJson(project_id='1'):
    with open('static_intent/{}.json'.format(project_id), 'r', encoding='utf8') as json_data:
        data = json.load(json_data)
        return data