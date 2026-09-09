import os, requests
from dotenv import load_dotenv
load_dotenv(override=True)
r = requests.get('https://openrouter.ai/api/v1/models', headers={'Authorization': f"Bearer {os.getenv('OPENROUTER_API_KEY')}"})
models = r.json()['data']
for m in models:
    if 'liquid' in m['id'].lower() or 'lfm' in m['id'].lower():
        print(m['id'])
