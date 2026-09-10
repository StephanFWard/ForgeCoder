import asyncio
from core.inference.client import InferenceClient

async def main():
    c = InferenceClient("http://127.0.0.1:8080")
    grammar = open("runtime/grammars/create.gbnf", encoding="utf-8").read()
    messages = [
        {"role": "system", "content": "You are ForgeCoder. Respond ONLY with JSON."},
        {"role": "user", "content": 'Create a small tic-tac-toe HTML page. Respond ONLY with JSON: {"message": "...", "creates": {"index.html": "complete file content"}}'},
    ]
    # Shape A: dict form (current client)
    try:
        raw = await c.chat(messages, temperature=0.1, max_tokens=300, grammar=grammar)
        print("DICT-FORM RAW >>>", raw[:400])
    except Exception as e:
        print("DICT-FORM ERROR:", e)

asyncio.run(main())
