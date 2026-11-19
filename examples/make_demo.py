#!/usr/bin/env python3
"""Build examples/demo.cassette.json and print the miss message used in the README.

    python3 examples/make_demo.py

The upstream here is a local stub, so the demo cassette is honest: it was
recorded through the same proxy path a real run uses, just against a fake model.
"""

import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import llm_cassette as lc  # noqa: E402

REPLY = "Your order 4471 left the warehouse this morning and arrives Thursday."


class StubModel(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        request = json.loads(self.rfile.read(length)) if length else {}
        last_user = ""
        for message in request.get("messages", []):
            if message.get("role") == "user":
                last_user = str(message.get("content"))
        payload = json.dumps(
            {
                "id": "chatcmpl-demo",
                "object": "chat.completion",
                "model": request.get("model", "gpt-5"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": REPLY if "4471" in last_user else "I do not have that order."},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 42, "completion_tokens": 18, "total_tokens": 60},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def post(url, payload):
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST")
    request.add_header("Content-Type", "application/json")
    request.add_header("Authorization", "Bearer sk-demo-should-never-be-recorded")
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def main() -> int:
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), StubModel)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    upstream_url = f"http://127.0.0.1:{upstream.server_address[1]}"

    cassette_path = HERE / "demo.cassette.json"
    if cassette_path.exists():
        cassette_path.unlink()

    server = lc.Server(cassette_path, mode="record", upstream=upstream_url, port=0).start()
    base = server.base_url
    try:
        post(base + "/v1/chat/completions", {
            "model": "gpt-5",
            "messages": [
                {"role": "system", "content": "You are a support agent for an online store. Answer in one sentence."},
                {"role": "user", "content": "Where is order 4471?"},
            ],
        })
        post(base + "/v1/chat/completions", {
            "model": "gpt-5",
            "messages": [
                {"role": "system", "content": "You are a support agent for an online store. Answer in one sentence."},
                {"role": "user", "content": "Where is order 4471?"},
                {"role": "assistant", "content": REPLY},
                {"role": "user", "content": "Can I change the delivery address?"},
            ],
        })
    finally:
        server.cassette.upstream = "stub://demo-model"
        server.stop()
        upstream.shutdown()
        upstream.server_close()

    print(f"wrote {cassette_path} with {len(lc.Cassette.load(cassette_path).entries)} exchanges")
    print(f"size: {cassette_path.stat().st_size} bytes")

    replay = lc.Server(cassette_path, mode="replay", port=0).start()
    try:
        request = urllib.request.Request(
            replay.base_url + "/v1/chat/completions",
            data=json.dumps({
                "model": "gpt-5",
                "messages": [
                    {"role": "system", "content": "You are a support agent for an online store. Answer in one sentence."},
                    {"role": "user", "content": "Where is order 4472?"},
                ],
            }).encode(),
            method="POST",
        )
        request.add_header("Content-Type", "application/json")
        try:
            urllib.request.urlopen(request, timeout=10)
        except urllib.error.HTTPError as error:
            print("\n--- miss response ---")
            print(error.read().decode())
    finally:
        replay.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
