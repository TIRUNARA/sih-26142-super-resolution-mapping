import http.server
import socketserver
import json
import urllib.parse
import os
import io
import time
import base64
from typing import Dict, Any

import torch
import numpy as np
from PIL import Image

try:
    from src.models.super_resolution import SatelliteSRMNet
    from src.data.synthetic_geospatial import generate_synthetic_satellite_tile
except ImportError:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from src.models.super_resolution import SatelliteSRMNet
    from src.data.synthetic_geospatial import generate_synthetic_satellite_tile

PORT = 8000
UI_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../ui"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
srm_model = SatelliteSRMNet(in_channels=4, out_channels=4, scale_factor=4).to(device)
srm_model.eval()

TILE_CACHE: Dict[str, str] = {}
CACHE_STATS = {"hits": 0, "misses": 0}

def get_super_resolved_tile_b64(tile_key: str) -> str:
    global CACHE_STATS, TILE_CACHE
    if tile_key in TILE_CACHE:
        CACHE_STATS["hits"] += 1
        return TILE_CACHE[tile_key]

    CACHE_STATS["misses"] += 1
    low_res_bands = generate_synthetic_satellite_tile(size=64)
    
    tensor_in = torch.from_numpy(low_res_bands).unsqueeze(0).float().to(device)
    with torch.no_grad():
        sr_output = srm_model(tensor_in).squeeze(0).cpu().numpy()

    rgb = sr_output[:3, :, :]
    rgb_normalized = ((rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8) * 255).astype(np.uint8)
    rgb_transposed = np.transpose(rgb_normalized, (1, 2, 0))

    img = Image.fromarray(rgb_transposed)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    if len(TILE_CACHE) > 100:
        first_key = next(iter(TILE_CACHE))
        del TILE_CACHE[first_key]
    
    TILE_CACHE[tile_key] = b64_str
    return b64_str


class SIHHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=UI_DIR, **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        
        if parsed.path == "/api/v1/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            response = {
                "status": "online",
                "problem": "SIH 26142 - Super Resolution Mapping",
                "model": "SatelliteSRMNet (RCAN + Spatial Attraction)",
                "scale_factor": "4x (10m -> 2.5m)",
                "device": str(device),
                "cache_stats": CACHE_STATS,
                "timestamp": time.time()
            }
            self.wfile.write(json.dumps(response).encode("utf-8"))
            return

        elif parsed.path == "/api/v1/tile":
            query = urllib.parse.parse_qs(parsed.query)
            z = query.get("z", ["14"])[0]
            x = query.get("x", ["0"])[0]
            y = query.get("y", ["0"])[0]
            tile_key = f"{z}_{x}_{y}"

            b64_image = get_super_resolved_tile_b64(tile_key)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            res = {
                "tile_key": tile_key,
                "zoom": z,
                "x": x,
                "y": y,
                "resolution": "2.5m",
                "image_data": f"data:image/png;base64,{b64_image}"
            }
            self.wfile.write(json.dumps(res).encode("utf-8"))
            return

        elif parsed.path == "/api/v1/prefetch":
            query = urllib.parse.parse_qs(parsed.query)
            tiles = query.get("tiles", [])
            prefetched = 0
            for t in tiles:
                if t not in TILE_CACHE:
                    get_super_resolved_tile_b64(t)
                    prefetched += 1

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "prefetched", "count": prefetched}).encode("utf-8"))
            return

        return super().do_GET()

def run_server():
    print(f"🚀 Starting NTRO SIH 26142 API Server on http://localhost:{PORT}")
    with socketserver.TCPServer(("", PORT), SIHHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")

if __name__ == "__main__":
    run_server()
