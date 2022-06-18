import requests

BASE_ENDPOINT = "http://api.semanticscholar.org/"

search_query = "graph/v1/paper/search?query=literature+graph"

r = requests.get(BASE_ENDPOINT+search_query)

data = r.json()

print(data)