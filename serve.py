#!/usr/bin/env python3
"""HTTP server with /log endpoint."""
import http.server, sys
from datetime import datetime

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080

with open('client.log', 'w') as f:
    f.write(f'=== {datetime.now()} ===\n')

class H(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()

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
