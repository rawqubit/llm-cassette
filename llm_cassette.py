#!/usr/bin/env python3
"""
llm-cassette - record LLM API traffic once, replay it forever.

Your agent tests either hit a real model on every run (slow, flaky, expensive,
nondeterministic) or mock the SDK (and drift from what the API actually does).
This is the third option: point the SDK's base URL at a local proxy, run the
test once against the real API, and commit the cassette. Every later run replays
byte-identical responses with no network and no tokens.

    llm-cassette record --cassette tests/cassettes/support.json \
                        --upstream https://api.openai.com --port 8787 &
    OPENAI_BASE_URL=http://127.0.0.1:8787/v1 pytest tests/

    llm-cassette replay --cassette tests/cassettes/support.json --port 8787

Cassettes are plain JSON, so a prompt change shows up as a reviewable diff, and
a miss tells you exactly which message diverged from the recording.

One file, standard library only, Python 3.9+.

MIT licensed.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import socket
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

__version__ = "0.1.0"
CASSETTE_VERSION = 1

REDACTED = "[REDACTED]"
SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b"),
    re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{16,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}\b"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{16,}"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
)
DROP_REQUEST_HEADERS = {"authorization", "x-api-key", "api-key", "cookie", "proxy-authorization"}
DROP_RESPONSE_HEADERS = {"transfer-encoding", "connection", "set-cookie", "content-length"}


# ---------------------------------------------------------------------------
# cassette model
# ---------------------------------------------------------------------------


def canonical_body(body: bytes) -> str:
    """A stable string for hashing: JSON is normalised, anything else is raw."""
    if not body:
        return ""
    try:
        parsed = json.loads(body.decode("utf-8"))
    except Exception:
        return body.decode("utf-8", "replace")
    return json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def request_key(method: str, path: str, body: bytes) -> str:
    payload = f"{method.upper()} {path}\n{canonical_body(body)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def redact(text: str) -> str:
    for pattern in SECRET_PATTERNS:
        text = pattern.sub(REDACTED, text)
    return text


def scrub(value, key_hint: str = ""):
    """Redact secrets in a decoded JSON structure, without changing its shape."""
    if isinstance(value, dict):
        return {k: scrub(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(item, key_hint) for item in value]
    if isinstance(value, str):
        if key_hint.lower() in ("api_key", "apikey", "authorization", "token", "secret"):
            return REDACTED
        return redact(value)
    return value


def summarise(body: bytes) -> dict:
    """What the human needs at a glance: model and message count."""
    try:
        parsed = json.loads(body.decode("utf-8"))
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    info = {}
    for key in ("model", "stream", "temperature", "max_tokens", "tool_choice"):
        if key in parsed:
            info[key] = parsed[key]
    for key in ("messages", "input", "contents"):
        if isinstance(parsed.get(key), list):
            info["messages"] = len(parsed[key])
            break
    if isinstance(parsed.get("tools"), list):
        info["tools"] = len(parsed["tools"])
    return info


class Cassette:
    """A list of recorded exchanges, keyed by request, with per-key ordering."""

    def __init__(self, entries: list = None, upstream: str = "", created_at: str = ""):
        self.entries = entries or []
        self.upstream = upstream
        self.created_at = created_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # -- io ---------------------------------------------------------------
    @classmethod
    def load(cls, path) -> "Cassette":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        version = data.get("version")
        if version != CASSETTE_VERSION:
            raise ValueError(f"cassette version {version} is not supported (expected {CASSETTE_VERSION})")
        return cls(data.get("entries") or [], data.get("upstream", ""), data.get("created_at", ""))

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CASSETTE_VERSION,
            "generator": f"llm-cassette {__version__}",
            "upstream": self.upstream,
            "created_at": self.created_at,
            "entries": self.entries,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n", encoding="utf-8")

    # -- lookup -----------------------------------------------------------
    def add(self, entry: dict) -> None:
        self.entries.append(entry)

    def counts(self) -> dict:
        counts = {}
        for entry in self.entries:
            counts[entry["key"]] = counts.get(entry["key"], 0) + 1
        return counts

    def find(self, key: str, occurrence: int) -> dict:
        seen = 0
        for entry in self.entries:
            if entry["key"] != key:
                continue
            if seen == occurrence:
                return entry
            seen += 1
        return None


# ---------------------------------------------------------------------------
# miss diagnostics
# ---------------------------------------------------------------------------


def _message_texts(body: bytes) -> list:
    try:
        parsed = json.loads(body.decode("utf-8"))
    except Exception:
        return []
    if not isinstance(parsed, dict):
        return []
    for key in ("messages", "input", "contents"):
        items = parsed.get(key)
        if isinstance(items, list):
            out = []
            for item in items:
                if isinstance(item, dict):
                    content = item.get("content")
                    if isinstance(content, list):
                        content = " ".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
                    out.append(f"{item.get('role', '?')}: {content}")
                else:
                    out.append(str(item))
            return out
    return []


def _first_divergence(left: list, right: list) -> int:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right))


def explain_miss(cassette: Cassette, body: bytes) -> str:
    """Say what the closest recording expected, and where the request diverged."""
    incoming = _message_texts(body)
    best = None
    for index, entry in enumerate(cassette.entries):
        recorded = _message_texts(entry.get("request", {}).get("body", "").encode("utf-8"))
        divergence = _first_divergence(recorded, incoming)
        score = divergence - abs(len(recorded) - len(incoming))
        if best is None or score > best[0]:
            best = (score, index, entry, recorded, divergence)

    if not best:
        return "cassette is empty: record a run first"

    _, index, entry, recorded, divergence = best
    lines = [f"closest recording is entry #{index} ({entry.get('method')} {entry.get('path')})"]
    if divergence < min(len(recorded), len(incoming)):
        lines.append(f"they first differ at message {divergence}:")
        lines.append(f"  recorded: {recorded[divergence][:160]}")
        lines.append(f"  received: {incoming[divergence][:160]}")
    else:
        lines.append(f"the recorded request has {len(recorded)} messages, the incoming one has {len(incoming)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# proxy
# ---------------------------------------------------------------------------


def _entry_for(method: str, path: str, key: str, headers, body: bytes, response: dict) -> dict:
    return {
        "key": key,
        "method": method,
        "path": path,
        "request": {
            "headers": {k: v for k, v in dict(headers).items() if k.lower() not in DROP_REQUEST_HEADERS},
            "body": redact(body.decode("utf-8", "replace")),
        },
        "response": response,
        "meta": {**summarise(body), "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
    }


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    cassette: Cassette = None
    mode: str = "replay"
    upstream: str = ""
    update: bool = False
    counts: dict = None
    lock = threading.Lock()
    verbose: bool = False

    def log_message(self, fmt, *args):  # noqa: A003
        if self.verbose:
            sys.stderr.write("llm-cassette: " + (fmt % args) + "\n")

    # -- helpers ----------------------------------------------------------
    def _read_request(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            return self.rfile.read(length)
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            chunks = []
            while True:
                line = self.rfile.readline().strip()
                size = int(line.split(b";")[0] or b"0", 16)
                if size == 0:
                    self.rfile.readline()
                    break
                chunks.append(self.rfile.read(size))
                self.rfile.readline()
            return b"".join(chunks)
        return b""

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _forward(self, path: str, body: bytes):
        parsed = urllib.parse.urlsplit(self.upstream)
        connection_class = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        connection = connection_class(parsed.netloc, timeout=120)
        target = parsed.path.rstrip("/") + path
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in ("host", "content-length", "connection")
        }
        connection.request(self.command, target, body=body or None, headers=headers)
        response = connection.getresponse()
        data = response.read()
        result = {
            "status": response.status,
            "headers": {key: value for key, value in response.getheaders()},
            "body": data.decode("utf-8", "replace"),
            "stream": "text/event-stream" in (response.getheader("Content-Type") or ""),
        }
        connection.close()
        return result

    # -- verbs ------------------------------------------------------------
    def do_POST(self):  # noqa: N802
        self._handle()

    def do_GET(self):  # noqa: N802
        self._handle()

    def _handle(self):
        body = self._read_request()
        key = request_key(self.command, self.path, body)

        if self.mode == "record":
            try:
                response = self._forward(self.path, body)
            except Exception as error:  # noqa: BLE001
                self._send_json(502, {"error": "llm-cassette could not reach the upstream", "detail": str(error)})
                return
            entry = _entry_for(self.command, self.path, key, self.headers, body, response)
            with self.lock:
                self.cassette.add(entry)
            self._send_replay(response)
            return

        with self.lock:
            occurrence = self.counts.get(key, 0)
            self.counts[key] = occurrence + 1
        entry = self.cassette.find(key, occurrence)
        if entry is None and self.update and self.upstream:
            try:
                response = self._forward(self.path, body)
            except Exception as error:  # noqa: BLE001
                self._send_json(502, {"error": "llm-cassette could not reach the upstream", "detail": str(error)})
                return
            with self.lock:
                self.cassette.add(_entry_for(self.command, self.path, key, self.headers, body, response))
            self._send_replay(response)
            return
        if entry is None:
            reason = explain_miss(self.cassette, body)
            self._send_json(
                501,
                {
                    "error": "llm-cassette miss: this request is not in the cassette",
                    "hint": "re-record with --update, or fix the code path that changed",
                    "detail": reason,
                },
            )
            return
        self._send_replay(entry["response"])

    def _send_replay(self, response: dict):
        body = (response.get("body") or "").encode("utf-8")
        self.send_response(int(response.get("status", 200)))
        headers = response.get("headers") or {}
        for key, value in headers.items():
            if key.lower() in DROP_RESPONSE_HEADERS:
                continue
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()

        if response.get("stream") and body:
            for chunk in body.split(b"\n\n"):
                if not chunk:
                    continue
                self.wfile.write(chunk + b"\n\n")
                self.wfile.flush()
        elif body:
            self.wfile.write(body)


class Server:
    """A cassette-backed proxy that can be used as a context manager."""

    def __init__(self, cassette_path, mode="replay", upstream="https://api.openai.com", port=0, update=False, verbose=False):
        self.cassette_path = Path(cassette_path)
        self.mode = mode
        self.upstream = upstream
        self.port = port
        self.update = update
        self.verbose = verbose

        self.cassette = Cassette.load(self.cassette_path) if self.cassette_path.exists() else Cassette(upstream=upstream)
        if self.mode == "record" and not self.cassette.upstream:
            self.cassette.upstream = upstream

        handler = type("Handler", (_Handler,), {
            "cassette": self.cassette,
            "mode": mode,
            "upstream": self.cassette.upstream or upstream,
            "update": update,
            "counts": {},
            "lock": threading.Lock(),
            "verbose": verbose,
        })
        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
        self.httpd.daemon_threads = True
        self.thread = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def start(self):
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return self

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        if self.mode == "record" or (self.mode == "replay" and self.update):
            self.cassette.save(self.cassette_path)


def _default_cassette() -> str:
    return os.environ.get("LLM_CASSETTE", "cassette.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_record(args) -> int:
    cassette_path = Path(args.cassette)
    server = Server(cassette_path, mode="record", upstream=args.upstream, port=args.port, verbose=args.verbose)
    base = server.base_url
    print(f"recording to {cassette_path}")
    print(f"upstream   {args.upstream}")
    print(f"base url   {base}")
    print("point the SDK at the base url and run the test, then press Ctrl-C")
    server.start()
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        print(f"\nwrote {len(server.cassette.entries)} exchanges to {cassette_path}")
    return 0


def cmd_replay(args) -> int:
    cassette_path = Path(args.cassette)
    if not cassette_path.exists() and not args.update:
        print(f"llm-cassette: no cassette at {cassette_path}", file=sys.stderr)
        return 2
    server = Server(cassette_path, mode="replay", port=args.port, update=args.update, verbose=args.verbose)
    print(f"replaying {cassette_path} ({len(server.cassette.entries)} exchanges)")
    print(f"base url  {server.base_url}")
    server.start()
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return 0


def cmd_inspect(args) -> int:
    cassette = Cassette.load(args.cassette)
    if not cassette.entries:
        print("cassette is empty")
        return 1
    print(f"{args.cassette}: {len(cassette.entries)} exchanges from {cassette.upstream or 'unknown upstream'}")
    print(f"recorded {cassette.created_at}\n")
    print(f"{'#':>3}  {'STATUS':>6}  {'BYTES':>7}  {'MODEL':<22}  {'MSGS':>4}  PATH")
    for index, entry in enumerate(cassette.entries):
        meta = entry.get("meta") or {}
        response_body = (entry.get("response") or {}).get("body") or ""
        size = len(response_body)
        print(
            f"{index:>3}  {entry.get('response', {}).get('status', '?'):>6}  {size:>7}  "
            f"{str(meta.get('model') or '-')[:22]:<22}  {str(meta.get('messages') or '-'):>4}  {entry.get('path')}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-cassette",
        description="Record LLM API traffic once and replay it forever: deterministic, offline, free tests.",
    )
    parser.add_argument("--version", action="version", version=f"llm-cassette {__version__}")
    sub = parser.add_subparsers(dest="command")

    record = sub.add_parser("record", help="proxy to the real API and write a cassette")
    record.add_argument("--cassette", default=_default_cassette())
    record.add_argument("--upstream", default=os.environ.get("LLM_CASSETTE_UPSTREAM", "https://api.openai.com"))
    record.add_argument("--port", type=int, default=8787)
    record.add_argument("--verbose", action="store_true")
    record.set_defaults(func=cmd_record)

    replay = sub.add_parser("replay", help="serve responses from a cassette, no network")
    replay.add_argument("--cassette", default=_default_cassette())
    replay.add_argument("--port", type=int, default=8787)
    replay.add_argument("--update", action="store_true", help="record any request the cassette does not have")
    replay.add_argument("--upstream", default=os.environ.get("LLM_CASSETTE_UPSTREAM", ""))
    replay.add_argument("--verbose", action="store_true")
    replay.set_defaults(func=cmd_replay)

    inspect = sub.add_parser("inspect", help="list what a cassette contains")
    inspect.add_argument("cassette")
    inspect.set_defaults(func=cmd_inspect)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
