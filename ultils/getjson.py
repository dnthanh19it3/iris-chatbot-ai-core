import urllib.request, json
def getJson(project_id = 0):
    # with urllib.request.urlopen("http://localhost/zalo-chatbot/export") as url:
    #     data = json.loads(url.read().decode())
    #     return data

    with open('static_intent/intents.json', 'r', encoding='utf8') as json_data:
        data = json.load(json_data)
        return data