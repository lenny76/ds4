#!/usr/bin/env python3
"""Small LAN web UI and reverse proxy for DwarfStar's HTTP server."""

from __future__ import annotations

import argparse
import http.client
import json
import mimetypes
import os
import secrets
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


STATIC_DIR = Path(__file__).resolve().parent / "static"
HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
}


class AppServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, upstream: str, api_key: str | None):
        super().__init__(address, handler)
        parsed = urlsplit(upstream)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("--upstream must be an http(s) URL")
        self.upstream_scheme = parsed.scheme
        self.upstream_host = parsed.hostname
        self.upstream_port = parsed.port or (443 if parsed.scheme == "https" else 80)
        self.upstream_prefix = parsed.path.rstrip("/")
        self.api_key = api_key


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"ok": True})
        elif self.path.startswith("/v1/"):
            self._proxy()
        else:
            self._static()

    def do_POST(self):
        if self.path.startswith("/v1/"):
            self._proxy()
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        if not self.path.startswith("/v1/"):
            self.send_error(404)
            return
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, X-API-Key")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _authorized(self) -> bool:
        expected = self.server.api_key
        if not expected:
            return True
        auth = self.headers.get("Authorization", "")
        supplied = auth[7:] if auth.lower().startswith("bearer ") else self.headers.get("X-API-Key", "")
        return secrets.compare_digest(supplied, expected)

    def _proxy(self):
        if not self._authorized():
            self._json(401, {"error": {"message": "Invalid or missing API key"}})
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else None
        headers = {
            key: value for key, value in self.headers.items()
            if key.lower() not in HOP_BY_HOP | {"host", "authorization", "x-api-key", "content-length"}
        }
        if body is not None:
            headers["Content-Length"] = str(len(body))
        connection_type = http.client.HTTPSConnection if self.server.upstream_scheme == "https" else http.client.HTTPConnection
        upstream = connection_type(self.server.upstream_host, self.server.upstream_port, timeout=3600)
        try:
            upstream.request(self.command, self.server.upstream_prefix + self.path, body=body, headers=headers)
            response = upstream.getresponse()
            self.send_response(response.status, response.reason)
            for key, value in response.getheaders():
                if key.lower() not in HOP_BY_HOP | {"content-length"}:
                    self.send_header(key, value)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Connection", "close")
            self.end_headers()
            while chunk := response.read1(16 * 1024):
                self.wfile.write(chunk)
                self.wfile.flush()
        except (OSError, http.client.HTTPException) as exc:
            if not self.wfile.closed:
                try:
                    self._json(502, {"error": {"message": f"DwarfStar unavailable: {exc}"}})
                except OSError:
                    pass
        finally:
            self.close_connection = True
            upstream.close()

    def _static(self):
        path = "/index.html" if self.path == "/" else self.path.split("?", 1)[0]
        candidate = (STATIC_DIR / path.lstrip("/")).resolve()
        if STATIC_DIR not in candidate.parents or not candidate.is_file():
            self.send_error(404)
            return
        content = candidate.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(candidate.name)[0] or "application/octet-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _json(self, status: int, value: object):
        content = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def main() -> None:
    parser = argparse.ArgumentParser(description="DwarfStar LAN web UI and API proxy")
    parser.add_argument("--host", default="0.0.0.0", help="UI bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="UI port (default: 8080)")
    parser.add_argument("--upstream", default="http://127.0.0.1:8000", help="DwarfStar server URL")
    parser.add_argument("--api-key", default=os.getenv("DS4_UI_API_KEY"), help="LAN API key or DS4_UI_API_KEY")
    args = parser.parse_args()
    server = AppServer((args.host, args.port), Handler, args.upstream, args.api_key)
    protection = "API key required" if args.api_key else "no API key"
    print(f"DwarfStar UI: http://{args.host}:{args.port} -> {args.upstream} ({protection})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
