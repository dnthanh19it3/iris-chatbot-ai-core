import urllib.request, json
def getJson(project_id):
    with urllib.request.urlopen("https://irisbot-ai.iristech.live/console/get-dataset/" + project_id) as url:
    # with urllib.request.urlopen("http://localhost/console/get-dataset/" + project_id) as url:
        data = json.loads(url.read().decode())
        print("Got intent")
        with open('static_intent/' + project_id + '.json', 'w',  encoding='utf8') as f:
            json.dump(data, f)
        return data

def getStaticJson(project_id):
    with open('static_intent/' + project_id + '.json', 'r', encoding='utf8') as json_data:
        data = json.load(json_data)
        return data