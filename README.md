# llm-cassette

**Your agent tests should not cost money.**

Record one real LLM session through a local proxy, commit the cassette, and every later test run replays byte-identical responses with no network, no tokens, and no flakiness.

```bash
llm-cassette record --cassette tests/cassettes/support.json --upstream https://api.openai.com &
OPENAI_BASE_URL=http://127.0.0.1:8787/v1 pytest tests/

llm-cassette replay --cassette tests/cassettes/support.json &
OPENAI_BASE_URL=http://127.0.0.1:8787/v1 pytest tests/   # offline, deterministic, free
```

One file, standard library only, Python 3.9+. No SDK to wrap, no decorator to learn: it is a proxy, so it works with the official SDKs, with raw `requests`, and with whatever your agent framework uses underneath.

## The part that matters when a test breaks

A generic VCR tool tells you `request not found`. That is useless when your agent has thirty messages and one of them changed. llm-cassette compares the incoming request against every recording and names the first message that diverged:

```
{
  "error": "llm-cassette miss: this request is not in the cassette",
  "hint": "re-record with --update, or fix the code path that changed",
  "detail": "closest recording is entry #0 (POST /v1/chat/completions)
they first differ at message 1:
  recorded: user: Where is order 4471?
  received: user: Where is order 4472?"
}
```

That output is produced by `examples/make_demo.py` against the committed demo cassette, not written by hand.

## Record and replay

**Record** against the real API, once:

```bash
llm-cassette record --cassette tests/cassettes/support.json \
                    --upstream https://api.openai.com --port 8787
```

Then point the client at the proxy for that run:

```bash
OPENAI_BASE_URL=http://127.0.0.1:8787/v1 pytest tests/test_support_agent.py
ANTHROPIC_BASE_URL=http://127.0.0.1:8787 pytest tests/test_support_agent.py
```

Press Ctrl-C. The cassette is written, and the request is reviewed like any other file in the diff.

**Replay** forever after:

```bash
llm-cassette replay --cassette tests/cassettes/support.json --port 8787
```

Any request the cassette does not have answers `501` with the divergence report above. That is deliberate: a silent fallback to the real API is how a test suite quietly starts costing money again.

**Update** deliberately, when the prompt changed on purpose:

```bash
llm-cassette replay --cassette tests/cassettes/support.json --update \
                    --upstream https://api.openai.com --port 8787
```

Misses are forwarded to the real API and appended; everything already recorded still replays. New entries show up as a diff, which is the review you want on a prompt change.

## In a test suite

No global process, no port juggling:

```python
import os
from llm_cassette import Server

def test_refund_flow():
    with Server("tests/cassettes/refund.json", mode="replay") as cassette:
        os.environ["OPENAI_BASE_URL"] = cassette.base_url + "/v1"
        result = run_agent("refund order 4471")
    assert "refunded" in result
```

`Server` binds a free port by default, so tests can run in parallel.

## What is in a cassette

Plain JSON, pretty-printed, sorted at the entry level, one file per scenario:

```json
{
  "version": 1,
  "generator": "llm-cassette 0.1.0",
  "upstream": "stub://demo-model",
  "created_at": "2026-11-19T08:31:07Z",
  "entries": [
    {
      "key": "6a1f...",
      "method": "POST",
      "path": "/v1/chat/completions",
      "request": { "headers": { "content-type": "application/json" }, "body": "{\"model\": \"gpt-5\", ...}" },
      "response": { "status": 200, "headers": { "Content-Type": "application/json" }, "body": "{...}", "stream": false },
      "meta": { "model": "gpt-5", "messages": 2, "recorded_at": "2026-11-19T08:31:07Z" }
    }
  ]
}
```

Three properties make it reviewable:

- **The key is content, not a timestamp.** Method, path and the canonical JSON body are hashed. Key order and whitespace do not matter, so a reformatted request still matches.
- **Identical requests replay in order.** An agent that retries the same call gets the first response, then the second, the way the real API behaved. Exhaust the sequence and you get a miss, not a loop.
- **Secrets are removed on the way in.** `Authorization`, `x-api-key`, cookies and five common token shapes are stripped or replaced with `[REDACTED]` before anything touches the disk.

Streaming responses are recorded as the raw `text/event-stream` body and replayed event by event, so an SSE parser sees the same frames it saw live.

## What it does not do

- It is not a mock. It replays what a model actually returned, including the unhelpful answers. A cassette that says something wrong keeps saying it.
- It cannot intercept a client that hardcodes the production URL. You have to point the base URL at the proxy, which every major SDK supports.
- It records wall-clock timing only as a property of a single exchange, not the delays inside a stream. If your code depends on token timing, this is the wrong tool.
- It does not shrink cassettes or deduplicate them beyond the key ordering. A long agent run makes a large file; that is the honest cost of replaying exactly what happened.

## Why not just mock the SDK

A mock tests your own understanding of the API, and it drifts. A cassette tests the API as it actually answered, and when the recording goes stale you see a diff rather than a silent lie. The failure mode you want in six months is a clear miss pointing at message 1, not a green suite that no longer reflects reality.

## License

MIT.

Built by [Srinikhil Chakilam](https://srinikhilchakilam.com). If a cassette catches something in your suite, I want to hear about it.
