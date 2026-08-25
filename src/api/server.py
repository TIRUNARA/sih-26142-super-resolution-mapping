"""
SIH 26142 Web Server & API Gateway
Serves the interactive UI workbench and API telemetry endpoint.
"""

import http.server
import socketserver
import os
import sys

PORT = 8080
UI_DIR = os.path.join(os.path.dirname(__file__), "..", "ui")

class SIHHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=UI_DIR, **kwargs)

    def do_GET(self):
        if self.path == "/api/status":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            response = '{"status": "online", "model": "SatelliteSRMNet", "upscale_factor": 4, "target_psnr": 34.2}'
            self.wfile.write(response.encode())
            return
        return super().do_GET()

if __name__ == "__main__":
    print(f"🛰️ SIH 26142 SRM Server starting on http://localhost:{PORT}")
    with socketserver.TCPServer(("", PORT), SIHHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")
