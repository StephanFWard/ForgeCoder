# System One: deterministic weighted decisions (free by default)

ForgeCoder's decision layer answers typed questions about a state and returns
**weighted, calibrated answers** — probability distributions plus a confidence —
instead of prose. Code branches on numbers; nothing has to be parsed out of a
sentence.

The design follows **Jev**, TypeSafe AI's "System One" model: no text
generation, just `choice` / `score` / `noul` questions answered against a
state, fast enough to call from code. ForgeCoder copies the request/response
contract exactly, but ships it with **free, offline backends by default** — the
hosted, billed Jev API is opt-in only and can never be selected implicitly.

## Why this shape

- **Deterministic answers.** The default backend has no model, no clock and no
  randomness: the same `(state, questions)` pair always produces the same
  probabilities, so weights can be thresholded and pinned in unit tests.
- **Free only, until you ask.** `deterministic` (default) and `local` backends
  cost nothing; `cost_usd` is always `0.0` in their responses.
- **Portable contract.** Anything built on these shapes ports to the hosted
  Jev API by swapping the backend — no schema work required.

## Package layout (`core/system_one/`)

| Module          | Contents                                                          |
| --------------- | ----------------------------------------------------------------- |
| `primitives.py` | Request/response shapes, Jev limits (255 choices, 2–10 score levels) |
| `backends.py`   | `deterministic` (default), `local`, `jev` backends + selection     |
| `decide.py`     | `Decider` facade + composition helpers (`combine`, `gate`, ...)    |
| `lexical.py`    | Tokenisation, IDF weighting, softmax — the deterministic arithmetic |
| `jev_compat.py` | Upstream `jev` decorator API (`@fn`, `decide`, `BaseModel`) for Python 3.10+ |
| `rerank.py`     | Weighted retrieval reranking with batched `score` questions       |
| `cli.py`        | `forge-decide` — one decision from a shell or a script            |

## Backends

| Backend        | Cost | What it does                                              | Needs                              |
| -------------- | ---- | --------------------------------------------------------- | ---------------------------------- |
| `deterministic`| free | IDF-weighted lexical overlap → softmax distributions      | Nothing (no model, no network)     |
| `local`        | free | 1.5B llama.cpp verdicts blended into the same shape; degrades to the deterministic answer if the server is down | llama-server on `:8080` |
| `jev`          | paid | The hosted TypeSafe-compatible API, identical contract    | `JEV_API_KEY` **and** `FORGECODER_SYSTEM_ONE_ALLOW_PAID=1` |

Selection is explicit: the `backend` field on a request, the
`FORGECODER_SYSTEM_ONE_BACKEND` environment variable, or `Decider(backend=...)`.
`"auto"` prefers `local` when the model is up and falls back to
`deterministic`; it never reaches the network. List what is available with
`GET /v1/decide/backends` or `forge-decide --backends`.

**Known limit, stated plainly:** the deterministic backend has no world
knowledge — a state that means "billing" without saying it ("I was charged
twice") cannot be routed to a `billing` option lexically. Use `local` (free,
on-device) for meaning-level verdicts and `deterministic` for gates over the
text in front of it.

## Question types

```python
from core.system_one import ChoiceQuestion, ScoreQuestion, NoulQuestion

choice = ChoiceQuestion(
    instructions="Which pipeline stage failed?",
    criteria={"build": "compilation or packaging error",
              "test": "a test assertion failed",
              "deploy": "release step failed"},
)
score = ScoreQuestion(
    instructions="How confident are we in this patch?",
    criteria=["guess", "plausible", "verified"],
)
noul = NoulQuestion(
    instructions="Should the change be blocked?",
    criteria={"true": "a block-severity rule fired",
              "false": "no blocking evidence"},
)
```

Answers carry the winning value, a confidence, and the full distribution:

```python
{"choice": "test", "confidence": 0.61,
 "probabilities": {"build": 0.17, "test": 0.61, "deploy": 0.22}}
{"score": 2, "confidence": 0.48, "probabilities": {"guess": 0.09, ...}}
{"noul": 0.83}
```

## Calling it from code

```python
from core.system_one import Decider, combine, gate

decider = Decider()  # backend="deterministic" by default: free, offline
decision = decider.decide(
    {"state": "build failed with a null pointer in UserService:45",
     "log": "AssertionError in test_user_lookup"},
    {"stage": choice, "risky": noul},
)

stage, confidence = decider.top_choice(decision.answers["stage"])
if gate(decision.answers["risky"], threshold=0.7):
    raise Blocked()

# Weighted composition — the weights are in *your* code, not in a prompt:
combined = combine(decision.answers, weights={"stage": 2.0, "risky": 1.0})
print(combined.total)  # 0..1 weighted mean of every answer
```

`Decider` also has async twins (`adecide`) for FastAPI paths.

## The upstream decorator API (Python 3.10+)

The PyPI `jev` package requires Python ≥ 3.14 and a paid key, so
`core/system_one/jev_compat.py` re-implements its public surface on the free
backends. The upstream usage works unchanged, and if the real package is
installed (Python 3.14+), the `jev` backend hands the same questions to it:

```python
from enum import Enum
from core.system_one.jev_compat import BaseModel, fn

class Stage(Enum):
    build = "build"
    test = "test"

class Triage(BaseModel):
    stage: Stage                    # Literal / Enum -> choice question
    urgent: bool                    # bool -> noul question
    confidence: int = 0             # 0..N -> score question

@fn(model=Triage)
def triage(build_log: str) -> Triage:
    """Triage this build log: {{ build_log }}"""

triage("AssertionError in test_user_lookup")   # -> Triage instance
```

Field annotations compile to questions at decoration time; unsupported ones
raise `TypeError` immediately rather than at the first call.

## Weighted rerank (`POST /v1/search` with `"weighted": true`)

`core/system_one/rerank.py` adds one **batched** decision on top of the
heuristic ranker: every candidate becomes a `score` question ("adding questions
barely changes the response time"), each answer returns an expected level plus
probabilities and a confidence, and the final order is an explicit blend:

```
weighted = (1 - w) * heuristic_normalised + w * expected_level_normalised
```

with `w = system_one_weight` (default `0.35`). Each row gains `weighted_score`,
`system_one`, and `system_one_confidence` so the ordering is inspectable.
With the default backend the whole rerank is offline arithmetic — the same
candidates and query always produce the same order.

## Endpoints and CLI

| Surface                         | Purpose                                        |
| ------------------------------- | ---------------------------------------------- |
| `POST /v1/decide`               | Jev-shaped request in, Jev-shaped answers out (+ `backend`, `free`) |
| `GET /v1/decide/backends`       | Backends, their cost, and what each needs      |
| `forge-decide --backends`       | Same catalog from a shell                      |
| `forge-decide --input req.json` | One decision from a file or stdin              |

Environment variables:

- `FORGECODER_SYSTEM_ONE_BACKEND` — default backend (`deterministic`).
- `FORGECODER_SYSTEM_ONE_ALLOW_PAID` — set to `1` to permit the billed `jev`
  backend; without it, a request for `jev` is refused, never billed.
- `JEV_API_KEY` / `TYPESAFE_API_KEY` — only consulted by the paid backend.
  Keys are also read from `.env.local` (gitignored; see `.env.example`).
- `JEV_API_URL` — override the endpoint. The key prefix picks it by default:
  `jv_live_...` keys go to the managed wrapper
  (`https://jevtypesafeai.com/api/v1/decide`); any other key is treated as an
  official TypeSafe key and goes to `https://api.typesafe.ai/v1/systemone`.
- `JEV_MODEL` — model name sent to the API (default `jev-latest`; the official
  endpoint requires it). Pin a version like `jev-1.13.0` in production so
  calibrated thresholds do not shift.
