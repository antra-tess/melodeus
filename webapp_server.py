#!/usr/bin/env python3
"""
HTTP + WebSocket Server for Phone-based Voice Participation
Serves the webapp and handles WebSocket connections for audio.
"""

import asyncio
import websockets
import json
import logging
from pathlib import Path
from typing import Dict, Set, Optional
from datetime import datetime
import tempfile
import subprocess
import argparse
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import os

# Import our voice WebSocket server
from voice_websocket_server import VoiceWebSocketServer, generate_qr_code

class WebAppHTTPHandler(SimpleHTTPRequestHandler):
    """HTTP handler for serving webapp files."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory="webapp", **kwargs)
    
    def end_headers(self):
        # Add CORS headers for local development
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_GET(self):
        # Serve index.html for root path
        if self.path == '/' or self.path == '/webapp/':
            self.path = '/index.html'
        
        return super().do_GET()

class CombinedServer:
    """Combined HTTP and WebSocket server for the voice participation system."""
    
    def __init__(self, host: str = 'localhost', port: int = 8765, http_port: int = 8080):
        self.host = host
        self.port = port
        self.http_port = http_port
        self.http_server = None
        self.http_thread = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def start_http_server(self):
        """Start the HTTP server in a separate thread."""
        def run_http():
            try:
                self.http_server = HTTPServer((self.host, self.http_port), WebAppHTTPHandler)
                print(f"🌐 HTTP server starting on http://{self.host}:{self.http_port}")
                self.http_server.serve_forever()
            except Exception as e:
                print(f"❌ HTTP server error: {e}")
        
        self.http_thread = threading.Thread(target=run_http, daemon=True)
        self.http_thread.start()
    
    def stop_http_server(self):
        """Stop the HTTP server."""
        if self.http_server:
            self.http_server.shutdown()
            self.http_server.server_close()
        
        if self.http_thread:
            self.http_thread.join(timeout=5)
    
    async def start_combined_server(self, public_host: str = None):
        """Start both HTTP and WebSocket servers."""
        
        # Start HTTP server for webapp
        self.start_http_server()
        
        # Start WebSocket server for voice
        voice_server = VoiceWebSocketServer(self.host, self.port)
        websocket_server = await voice_server.start_server()
        
        # Generate QR code
        public_host = public_host or self.host
        webapp_url = f"http://{public_host}:{self.http_port}"
        generate_qr_code(webapp_url)
        
        print(f"\n🎭 Claude Sonnet Funeral Voice System Ready!")
        print(f"📱 Webapp URL: {webapp_url}")
        print(f"🔌 WebSocket URL: ws://{public_host}:{self.port}")
        print(f"📞 People can scan the QR code to join the conversation")
        print(f"🎤 Each person will identify themselves and use push-to-talk")
        print(f"\n🚀 Ready for the memorial event!")
        
        return websocket_server
    
    async def shutdown(self):
        """Shutdown both servers."""
        print("\n🛑 Shutting down servers...")
        self.stop_http_server()
        print("✅ Servers stopped")

async def main():
    """Main function to run the combined server."""
    parser = argparse.ArgumentParser(description='Voice Participation System for Claude Sonnet Funeral')
    parser.add_argument('--host', default='0.0.0.0', help='Server host (default: 0.0.0.0 for external access)')
    parser.add_argument('--port', type=int, default=8765, help='WebSocket port (default: 8765)')
    parser.add_argument('--http-port', type=int, default=8080, help='HTTP port for webapp (default: 8080)')
    parser.add_argument('--public-host', help='Public hostname/IP for QR code (e.g., 192.168.1.100)')
    
    args = parser.parse_args()
    
    # Ensure webapp directory exists
    webapp_dir = Path("webapp")
    webapp_dir.mkdir(exist_ok=True)
    
    if not (webapp_dir / "index.html").exists():
        print("❌ webapp/index.html not found!")
        print("   Make sure the webapp files are in the 'webapp' directory")
        return
    
    # Create and start combined server
    server = CombinedServer(args.host, args.port, args.http_port)
    
    try:
        websocket_server = await server.start_combined_server(args.public_host)
        
        # Keep servers running
        await websocket_server.wait_closed()
        
    except KeyboardInterrupt:
        await server.shutdown()

if __name__ == "__main__":
    asyncio.run(main()) 