import requests

text = "Sardi khokla tap ghamoriya"
resp = requests.post('http://127.0.0.1:8000/predict', json={'text': text})
print('Status', resp.status_code)
print('Response escaped', resp.text.encode('unicode_escape'))
