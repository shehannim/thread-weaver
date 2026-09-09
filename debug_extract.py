import json, os
from dotenv import load_dotenv
load_dotenv(override=True)
import sys
sys.path.insert(0, '.')
from graph.extract import extract_from_chunk

chunks = [json.loads(l) for l in open('data/processed/chunks.jsonl', encoding='utf-8')]
test_chunk = chunks[0]
print('Testing chunk:', test_chunk['chunk_id'])
try:
    result = extract_from_chunk(test_chunk)
    print('SUCCESS:', result)
except Exception as e:
    import traceback
    traceback.print_exc()
