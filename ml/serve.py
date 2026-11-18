"""Local server for the course-agnostic engine: serves index.html and two JSON endpoints.
Run (from coursemate-app/): ml/.venv/bin/python ml/serve.py   then open http://127.0.0.1:8765
POST /api/ingest   body = file bytes, header X-Filename -> {chunks, faq, threshold}
POST /api/ask      body = {"q": "..."}                 -> {stamp, answer, section, conf}
GET  /api/health   -> {ok, threshold, bank}"""
import json, sys
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import Engine

ROOT = Path(__file__).resolve().parent.parent
ENGINE = Engine()
MAX_BYTES = 20 * 1024 * 1024


class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k): super().__init__(*a, directory=str(ROOT), **k)

    def log_message(self, *a): pass

    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        if self.path == '/api/health':
            return self._json(200, {'ok': True, 'threshold': ENGINE.t, 'bank': len(ENGINE.bank)})
        return super().do_GET()

    def do_POST(self):
        n = int(self.headers.get('Content-Length') or 0)
        if n > MAX_BYTES: return self._json(413, {'error': 'file too large (20 MB max)'})
        body = self.rfile.read(n)
        try:
            if self.path == '/api/ingest':
                return self._json(200, ENGINE.ingest(self.headers.get('X-Filename') or 'upload.txt', body))
            if self.path == '/api/ask':
                return self._json(200, ENGINE.ask(json.loads(body or b'{}').get('q', '')))
            return self._json(404, {'error': 'not found'})
        except ValueError as e:
            return self._json(400, {'error': str(e)})


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f'CourseMate engine on http://127.0.0.1:{port}  (threshold {ENGINE.t:.3f}, {len(ENGINE.bank)} bank questions)', flush=True)
    ThreadingHTTPServer(('127.0.0.1', port), H).serve_forever()
