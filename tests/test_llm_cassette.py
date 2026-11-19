#!/usr/bin/env python3
"""Tests for llm_cassette. Run: python3 -m unittest discover -s tests -v"""

import json
import sys
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import llm_cassette as lc  # noqa: E402

CHAT_BODY = {
    "model": "gpt-5",
    "messages": [
        {"role": "system", "content": "You are a support agent."},
        {"role": "user", "content": "Where is order 4471?"},
    ],
}
CHAT_RESPONSE = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "It ships tomorrow."}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 31, "completion_tokens": 9, "total_tokens": 40},
}
SSE_STREAM = (
    'data: {"id":"chatcmpl-2","choices":[{"delta":{"content":"It "}}]}\n\n'
    'data: {"id":"chatcmpl-2","choices":[{"delta":{"content":"ships tomorrow."}}]}\n\n'
    "data: [DONE]\n\n"
)


class FakeUpstream(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    hits = []
    responses = {}

    def log_message(self, *args):
        pass

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        FakeUpstream.hits.append({"path": self.path, "body": body, "headers": dict(self.headers)})

        if self.path.endswith("/stream"):
            payload = SSE_STREAM.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if self.path.endswith("/error"):
            payload = json.dumps({"error": {"message": "rate limited"}}).encode()
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        payload = json.dumps(CHAT_RESPONSE).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def start_upstream():
    FakeUpstream.hits = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeUpstream)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def post(url, payload, headers=None):
    data = json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, method="POST")
    request.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.status, response.read().decode()


class RecordTests(unittest.TestCase):
    def setUp(self):
        self.upstream, upstream_url = start_upstream()
        self.tmp = Path(self._testMethodName + ".cassette.json")
        if self.tmp.exists():
            self.tmp.unlink()
        self.server = lc.Server(self.tmp, mode="record", upstream=upstream_url, port=0)
        self.server.start()

    def tearDown(self):
        self.server.stop()
        self.upstream.shutdown()
        self.upstream.server_close()
        if self.tmp.exists():
            self.tmp.unlink()

    def test_record_writes_one_entry(self):
        status, body = post(self.server.base_url + "/v1/chat/completions", CHAT_BODY)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["choices"][0]["message"]["content"], "It ships tomorrow.")
        self.assertEqual(len(self.server.cassette.entries), 1)
        entry = self.server.cassette.entries[0]
        self.assertEqual(entry["method"], "POST")
        self.assertEqual(entry["path"], "/v1/chat/completions")
        self.assertEqual(entry["meta"]["model"], "gpt-5")
        self.assertEqual(entry["meta"]["messages"], 2)

    def test_record_forwards_the_body_unchanged(self):
        post(self.server.base_url + "/v1/chat/completions", CHAT_BODY)
        self.assertEqual(len(FakeUpstream.hits), 1)
        forwarded = json.loads(FakeUpstream.hits[0]["body"])
        self.assertEqual(forwarded["model"], "gpt-5")

    def test_record_drops_auth_headers(self):
        post(self.server.base_url + "/v1/chat/completions", CHAT_BODY, {"Authorization": "Bearer sk-secret-value-1234567890"})
        entry = self.server.cassette.entries[0]
        self.assertNotIn("Authorization", entry["request"]["headers"])

    def test_record_redacts_secrets_in_the_body(self):
        payload = dict(CHAT_BODY)
        payload["api_key"] = "sk-live-abcdefghijklmnopqrstuvwxyz"
        payload["messages"] = [{"role": "user", "content": "my key is sk-live-abcdefghijklmnopqrstuvwxyz"}]
        post(self.server.base_url + "/v1/chat/completions", payload)
        self.server.stop()
        written = self.tmp.read_text()
        self.assertNotIn("sk-live-abcdefghijklmnopqrstuvwxyz", written)
        self.assertIn(lc.REDACTED, written)

    def test_record_preserves_error_status(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            post(self.server.base_url + "/v1/error", CHAT_BODY)
        self.assertEqual(caught.exception.code, 429)
        self.assertIn("rate limited", caught.exception.read().decode())
        self.assertEqual(self.server.cassette.entries[0]["response"]["status"], 429)

    def test_stream_is_flagged(self):
        status, body = post(self.server.base_url + "/v1/stream", CHAT_BODY)
        self.assertEqual(status, 200)
        self.assertIn("[DONE]", body)
        entry = self.server.cassette.entries[0]
        self.assertTrue(entry["response"]["stream"])
        self.assertIn("ships tomorrow", entry["response"]["body"])


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self._testMethodName + ".cassette.json")
        cassette = lc.Cassette(upstream="http://example.invalid")
        body = lc.canonical_body(json.dumps(CHAT_BODY).encode()).encode()
        key = lc.request_key("POST", "/v1/chat/completions", body)
        cassette.add(
            {
                "key": key,
                "method": "POST",
                "path": "/v1/chat/completions",
                "request": {"headers": {}, "body": json.dumps(CHAT_BODY)},
                "response": {
                    "status": 200,
                    "headers": {"Content-Type": "application/json"},
                    "body": json.dumps(CHAT_RESPONSE),
                    "stream": False,
                },
                "meta": {"model": "gpt-5", "messages": 2},
            }
        )
        cassette.save(self.tmp)
        self.server = lc.Server(self.tmp, mode="replay", port=0)
        self.server.start()

    def tearDown(self):
        self.server.stop()
        if self.tmp.exists():
            self.tmp.unlink()

    def test_replay_returns_the_recorded_body(self):
        status, body = post(self.server.base_url + "/v1/chat/completions", CHAT_BODY)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), CHAT_RESPONSE)

    def test_replay_accepts_an_equivalent_request(self):
        reordered = {"messages": CHAT_BODY["messages"], "model": "gpt-5"}
        status, body = post(self.server.base_url + "/v1/chat/completions", reordered)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), CHAT_RESPONSE)

    def test_miss_returns_501_and_explains_the_divergence(self):
        changed = {"model": "gpt-5", "messages": CHAT_BODY["messages"] + [{"role": "assistant", "content": "It ships tomorrow."}, {"role": "user", "content": "And order 9001?"}]}
        with self.assertRaises(urllib.error.HTTPError) as caught:
            post(self.server.base_url + "/v1/chat/completions", changed)
        self.assertEqual(caught.exception.code, 501)
        payload = json.loads(caught.exception.read().decode())
        self.assertIn("miss", payload["error"])
        self.assertIn("closest recording", payload["detail"])

    def test_missing_entry_reports_empty_cassette(self):
        empty = Path("empty.cassette.json")
        empty.write_text(json.dumps({"version": 1, "entries": [], "upstream": ""}))
        server = lc.Server(empty, mode="replay", port=0).start()
        try:
            with self.assertRaises(urllib.error.HTTPError) as caught:
                post(server.base_url + "/v1/chat/completions", CHAT_BODY)
            payload = json.loads(caught.exception.read().decode())
            self.assertIn("record a run first", payload["detail"])
        finally:
            server.stop()
            empty.unlink()


class SequenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(self._testMethodName + ".cassette.json")
        body = lc.canonical_body(json.dumps(CHAT_BODY).encode()).encode()
        key = lc.request_key("POST", "/v1/chat/completions", body)
        cassette = lc.Cassette()
        for index in range(2):
            cassette.add(
                {
                    "key": key,
                    "method": "POST",
                    "path": "/v1/chat/completions",
                    "request": {"headers": {}, "body": json.dumps(CHAT_BODY)},
                    "response": {"status": 200, "headers": {}, "body": json.dumps({"n": index}), "stream": False},
                    "meta": {},
                }
            )
        cassette.save(self.tmp)
        self.server = lc.Server(self.tmp, mode="replay", port=0).start()

    def tearDown(self):
        self.server.stop()
        self.tmp.unlink()

    def test_identical_requests_replay_in_order(self):
        first = json.loads(post(self.server.base_url + "/v1/chat/completions", CHAT_BODY)[1])
        second = json.loads(post(self.server.base_url + "/v1/chat/completions", CHAT_BODY)[1])
        self.assertEqual((first["n"], second["n"]), (0, 1))

    def test_exhausted_sequence_is_a_clean_miss(self):
        post(self.server.base_url + "/v1/chat/completions", CHAT_BODY)
        post(self.server.base_url + "/v1/chat/completions", CHAT_BODY)
        with self.assertRaises(urllib.error.HTTPError) as caught:
            post(self.server.base_url + "/v1/chat/completions", CHAT_BODY)
        self.assertEqual(caught.exception.code, 501)


class UpdateTests(unittest.TestCase):
    def test_update_records_a_miss_into_the_cassette(self):
        upstream, upstream_url = start_upstream()
        path = Path("update.cassette.json")
        if path.exists():
            path.unlink()
        lc.Cassette(upstream=upstream_url).save(path)
        server = lc.Server(path, mode="replay", upstream=upstream_url, port=0, update=True).start()
        try:
            status, body = post(server.base_url + "/v1/chat/completions", CHAT_BODY)
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body), CHAT_RESPONSE)
            self.assertEqual(len(FakeUpstream.hits), 1)
        finally:
            server.stop()
            upstream.shutdown()
            upstream.server_close()
        try:
            reloaded = lc.Cassette.load(path)
            self.assertEqual(len(reloaded.entries), 1)
        finally:
            path.unlink()


class StreamingTests(unittest.TestCase):
    def test_stream_roundtrip(self):
        upstream, upstream_url = start_upstream()
        path = Path("stream.cassette.json")
        if path.exists():
            path.unlink()
        recorder = lc.Server(path, mode="record", upstream=upstream_url, port=0).start()
        try:
            status, body = post(recorder.base_url + "/v1/stream", CHAT_BODY)
            self.assertEqual(status, 200)
            self.assertEqual(body, SSE_STREAM)
        finally:
            recorder.stop()
            upstream.shutdown()
            upstream.server_close()

        replayer = lc.Server(path, mode="replay", port=0).start()
        try:
            status, replayed = post(replayer.base_url + "/v1/stream", CHAT_BODY)
            self.assertEqual(status, 200)
            self.assertEqual(replayed, SSE_STREAM)
        finally:
            replayer.stop()
            path.unlink()


class FormatTests(unittest.TestCase):
    def test_cassette_is_readable_json(self):
        path = Path("format.cassette.json")
        cassette = lc.Cassette(upstream="https://api.anthropic.com")
        cassette.add({"key": "abc", "method": "POST", "path": "/v1/messages", "request": {"headers": {}, "body": "{}"}, "response": {"status": 200, "headers": {}, "body": "{}", "stream": False}, "meta": {}})
        cassette.save(path)
        try:
            text = path.read_text()
            self.assertIn('"version": 1', text)
            self.assertIn('"upstream"', text)
            self.assertTrue(len(text.strip().splitlines()) > 5)
            reloaded = lc.Cassette.load(path)
            self.assertEqual(len(reloaded.entries), 1)
        finally:
            path.unlink()

    def test_unsupported_version_is_refused(self):
        path = Path("old.cassette.json")
        path.write_text(json.dumps({"version": 99, "entries": []}))
        try:
            with self.assertRaises(ValueError):
                lc.Cassette.load(path)
        finally:
            path.unlink()

    def test_key_ignores_whitespace_and_key_order(self):
        left = lc.request_key("POST", "/v1/x", json.dumps({"b": 1, "a": 2}).encode())
        right = lc.request_key("POST", "/v1/x", b'{"a": 2,   "b": 1}')
        self.assertEqual(left, right)

    def test_key_changes_with_content(self):
        left = lc.request_key("POST", "/v1/x", b'{"a": 1}')
        right = lc.request_key("POST", "/v1/x", b'{"a": 2}')
        self.assertNotEqual(left, right)


class RedactionTests(unittest.TestCase):
    def test_known_secret_shapes_are_removed(self):
        cases = [
            "sk-live-abcdefghijklmnopqrstuvwxyz",
            "sk-ant-api03-abcdefghijklmnopqrstuvwxyz",
            "AIzaSyA1234567890abcdefghijklmnopqrstu",
            "Bearer abcdefghijklmnopqrstuvwxyz123456",
            "ghp_abcdefghijklmnopqrstuvwxyz123456",
        ]
        for secret in cases:
            self.assertNotIn(secret, lc.redact(f"header: {secret} end"))

    def test_scrub_keys_by_name(self):
        cleaned = lc.scrub({"api_key": "anything", "nested": {"token": "x"}, "model": "gpt-5"})
        self.assertEqual(cleaned["api_key"], lc.REDACTED)
        self.assertEqual(cleaned["nested"]["token"], lc.REDACTED)
        self.assertEqual(cleaned["model"], "gpt-5")


class ServerTests(unittest.TestCase):
    def test_context_manager_gives_a_free_port(self):
        path = Path("ctx.cassette.json")
        lc.Cassette().save(path)
        with lc.Server(path, mode="replay", port=0) as server:
            self.assertTrue(server.base_url.startswith("http://127.0.0.1:"))
            self.assertGreater(server.httpd.server_address[1], 1024)
        path.unlink()

    def test_inspect_cli_lists_entries(self):
        path = Path("inspect.cassette.json")
        cassette = lc.Cassette(upstream="https://api.openai.com")
        cassette.add({"key": "k", "method": "POST", "path": "/v1/chat/completions", "request": {"headers": {}, "body": "{}"}, "response": {"status": 200, "headers": {}, "body": '{"ok":true}', "stream": False}, "meta": {"model": "gpt-5", "messages": 2}})
        cassette.save(path)
        try:
            self.assertEqual(lc.main(["inspect", str(path)]), 0)
        finally:
            path.unlink()


if __name__ == "__main__":
    unittest.main()
