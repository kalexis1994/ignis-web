#!/usr/bin/env python3
"""HTTP server with /log endpoint and .env scene path support."""
import http.server, sys, os
from datetime import datetime
from pathlib import Path

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080

# Load .env file
env_path = Path(__file__).parent / '.env'
SCENE_PATH = None
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, val = line.split('=', 1)
            if key.strip() == 'SCENE_PATH':
                SCENE_PATH = val.strip()

if SCENE_PATH:
    print(f'Scene path: {SCENE_PATH}')

with open('client.log', 'w') as f:
    f.write(f'=== {datetime.now()} ===\n')

class H(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def translate_path(self, path):
        # Redirect /scene/* requests to SCENE_PATH if configured
        if SCENE_PATH and path.startswith('/scene/'):
            rel = path[len('/scene/'):]
            return os.path.join(SCENE_PATH, rel)
        return super().translate_path(path)

    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(n).decode()
        line = f'[{datetime.now().strftime("%H:%M:%S")}] {body}'
        print(line, flush=True)
        with open('client.log', 'a') as f:
            f.write(line + '\n')
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'ok')

with http.server.HTTPServer(('0.0.0.0', PORT), H) as s:
    print(f'http://localhost:{PORT}')
    s.serve_forever()
