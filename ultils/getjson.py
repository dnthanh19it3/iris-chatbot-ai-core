import urllib.request, json
def getJson(project_id = 0):
    with urllib.request.urlopen("http://127.0.0.1/console/get-dataset/" + project_id) as url:
        data = json.loads(url.read().decode())
        return data
    # with open('static_intent/intents.json', 'r', encoding='utf8') as json_data:
    #     data = json.load(json_data)
    #     return data