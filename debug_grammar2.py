import asyncio, httpx

async def main():
    grammar = open("runtime/grammars/create.gbnf", encoding="utf-8").read()
    base = {"model": "forgecoder", "messages": [
        {"role": "user", "content": 'Create a tiny tic-tac-toe HTML page. JSON only: {"message": "...", "creates": {"index.html": "content"}}'}],
        "temperature": 0.1, "max_tokens": 120}
    async with httpx.AsyncClient(timeout=60) as c:
        # Shape B: top-level string
        p = dict(base); p["grammar"] = grammar
        r = await c.post("http://127.0.0.1:8080/v1/chat/completions", json=p)
        print("TOP-LEVEL STATUS", r.status_code)
        txt = r.json()["choices"][0]["message"]["content"]
        print("TOP-LEVEL RAW >>>", txt[:300])
        # Shape C: response_format grammar
        p = dict(base); p["response_format"] = {"type": "grammar", "value": grammar}
        r = await c.post("http://127.0.0.1:8080/v1/chat/completions", json=p)
        print("RF STATUS", r.status_code)
        txt = r.json()["choices"][0]["message"]["content"]
        print("RF RAW >>>", txt[:300])

asyncio.run(main())
