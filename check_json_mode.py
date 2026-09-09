import os, json
from dotenv import load_dotenv
load_dotenv(override=True)
from openai import OpenAI

client = OpenAI(base_url='https://openrouter.ai/api/v1', api_key=os.getenv('OPENROUTER_API_KEY'))
try:
    resp = client.chat.completions.create(
        model='nvidia/nemotron-3-super-120b-a12b:free',
        messages=[{'role': 'user', 'content': 'Return JSON: {\"greeting\": \"hello\"}'}],
        response_format={'type': 'json_object'}
    )
    print('SUCCESS:', resp.choices[0].message.content)
except Exception as e:
    print('ERROR TYPE:', type(e).__name__)
    print('ERROR:', str(e))
