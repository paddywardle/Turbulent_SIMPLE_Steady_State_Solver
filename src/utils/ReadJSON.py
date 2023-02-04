import json

def ReadJSON(filename):

    with open(filename, 'r') as f:
        return json.load(f)
