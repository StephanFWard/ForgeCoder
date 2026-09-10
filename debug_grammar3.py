import asyncio, httpx, json

SCHEMA = {
    "type": "object",
    "properties": {
        "message": {"type": "string"},
        "creates": {"type": "object", "additionalProperties": {"type": "string"}},
    },
    "required": ["message", "creates"],
}

async def main():
    payload = {
        "model": "forgecoder",
        "messages": [{"role": "user", "content": 'Create a tiny tic-tac-toe HTML page. JSON only: {"message": "...", "creates": {"index.html": "content"}}'}],
        "temperature": 0.1, "max_tokens": 120,
        "response_format": {"type": "json_schema", "json_schema": {"schema": SCHEMA}},
    }
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post("http://127.0.0.1:8080/v1/chat/completions", json=payload)
        print("STATUS", r.status_code)
        if r.status_code == 200:
            txt = r.json()["choices"][0]["message"]["content"]
            print("RAW >>>", txt[:300])
            json.loads(txt)
            print("CLEAN JSON: YES")

asyncio.run(main())
