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

SCENE_ROOT = None
SCENE_FILE = None
SCENE_FILE_ALIAS = None
if SCENE_PATH:
    scene_path = Path(SCENE_PATH)
    if scene_path.is_file() and scene_path.suffix.lower() in ('.gltf', '.glb'):
        SCENE_FILE = str(scene_path)
        SCENE_FILE_ALIAS = f'scene{scene_path.suffix.lower()}'
    SCENE_ROOT = scene_path.parent if scene_path.is_file() else scene_path
    SCENE_ROOT = str(SCENE_ROOT)

if SCENE_PATH:
    print(f'Scene path: {SCENE_PATH}')
if SCENE_ROOT:
    print(f'Scene root: {SCENE_ROOT}')
if SCENE_FILE:
    print(f'Scene file: {SCENE_FILE}')

with open('client.log', 'w', encoding='utf-8') as f:
    f.write(f'=== {datetime.now()} ===\n')

class H(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def do_GET(self):
        if self.path == '/scene/.entry' and SCENE_FILE_ALIAS:
            data = SCENE_FILE_ALIAS.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        super().do_GET()

    def translate_path(self, path):
        # Redirect /scene/* requests to SCENE_PATH if configured
        if SCENE_ROOT and path.startswith('/scene/'):
            rel = path[len('/scene/'):]
            if SCENE_FILE and rel == SCENE_FILE_ALIAS:
                return SCENE_FILE
            return os.path.join(SCENE_ROOT, rel)
        return super().translate_path(path)

    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(n).decode()
        line = f'[{datetime.now().strftime("%H:%M:%S")}] {body}'
        print(line, flush=True)
        with open('client.log', 'a', encoding='utf-8') as f:
            f.write(line + '\n')
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'ok')

with http.server.HTTPServer(('0.0.0.0', PORT), H) as s:
    print(f'http://localhost:{PORT}')
    s.serve_forever()
