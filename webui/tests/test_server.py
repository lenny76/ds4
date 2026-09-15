import json
import sys
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import AppServer, Handler  # noqa: E402


class FakeModel(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"data": [{"id": "test-model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        request = json.loads(self.rfile.read(length))
        chunks = [
            'data: {"choices":[{"delta":{"content":"ciao"}}]}\n\n',
            'data: [DONE]\n\n',
        ]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(chunk.encode())
            self.wfile.flush()
        self.close_connection = True
        self.server.last_request = request

    def log_message(self, *_):
        pass


class WebUITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = ThreadingHTTPServer(("127.0.0.1", 0), FakeModel)
        cls.model_thread = threading.Thread(target=cls.model.serve_forever, daemon=True)
        cls.model_thread.start()
        upstream = f"http://127.0.0.1:{cls.model.server_port}"
        cls.app = AppServer(("127.0.0.1", 0), Handler, upstream, "secret")
        cls.app_thread = threading.Thread(target=cls.app.serve_forever, daemon=True)
        cls.app_thread.start()
        cls.base = f"http://127.0.0.1:{cls.app.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.app.shutdown(); cls.app.server_close()
        cls.model.shutdown(); cls.model.server_close()

    def request(self, path, *, data=None, auth=True):
        headers = {"Content-Type": "application/json"}
        if auth:
            headers["Authorization"] = "Bearer secret"
        request = urllib.request.Request(self.base + path, data=data, headers=headers)
        return urllib.request.urlopen(request, timeout=2)

    def test_static_ui_is_public(self):
        with self.request("/", auth=False) as response:
            self.assertIn(b"DwarfStar Chat", response.read())

    def test_api_requires_key(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request("/v1/models", auth=False)
        self.assertEqual(caught.exception.code, 401)

    def test_models_proxy(self):
        with self.request("/v1/models") as response:
            self.assertEqual(json.load(response)["data"][0]["id"], "test-model")

    def test_streaming_chat_proxy(self):
        payload = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
        with self.request("/v1/chat/completions", data=payload) as response:
            body = response.read().decode()
        self.assertIn('"content":"ciao"', body)
        self.assertEqual(self.model.last_request["messages"][0]["content"], "hi")


if __name__ == "__main__":
    unittest.main()
