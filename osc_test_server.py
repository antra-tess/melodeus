#!/usr/bin/env python3
"""
Test OSC Server with Web UI for Avatar Colors

Run this to test avatar color changes via OSC.
Web UI shows avatars with their current colors.
"""

import asyncio
import json
from aiohttp import web
from pythonosc import dispatcher, osc_server
import threading

# Store avatar colors (name -> [r, g, b])
avatar_colors = {}
websocket_clients = set()

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Avatar Color Test</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            background: #1a1a2e;
            color: white;
            padding: 20px;
            margin: 0;
        }
        h1 {
            color: #eee;
            margin-bottom: 20px;
        }
        .avatars {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
        }
        .avatar {
            background: #16213e;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            min-width: 150px;
            transition: all 0.3s ease;
        }
        .avatar-circle {
            width: 100px;
            height: 100px;
            border-radius: 50%;
            margin: 0 auto 15px;
            transition: background-color 0.3s ease;
            border: 3px solid rgba(255,255,255,0.2);
        }
        .avatar-name {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .avatar-rgb {
            font-size: 12px;
            color: #888;
            font-family: monospace;
        }
        .log {
            margin-top: 30px;
            background: #0f0f1a;
            border-radius: 8px;
            padding: 15px;
            max-height: 200px;
            overflow-y: auto;
            font-family: monospace;
            font-size: 12px;
        }
        .log-entry {
            padding: 3px 0;
            border-bottom: 1px solid #222;
        }
        .status {
            position: fixed;
            top: 10px;
            right: 10px;
            padding: 8px 15px;
            border-radius: 20px;
            font-size: 12px;
        }
        .status.connected { background: #2ecc71; }
        .status.disconnected { background: #e74c3c; }
    </style>
</head>
<body>
    <h1>Avatar Color Test Server</h1>
    <div class="status disconnected" id="status">Disconnected</div>

    <div class="avatars" id="avatars">
        <div class="avatar">
            <div class="avatar-circle" style="background: #444"></div>
            <div class="avatar-name">Waiting for OSC...</div>
            <div class="avatar-rgb">No messages yet</div>
        </div>
    </div>

    <div class="log" id="log"></div>

    <script>
        let ws;
        const avatars = {};

        function connect() {
            ws = new WebSocket(`ws://${location.host}/ws`);

            ws.onopen = () => {
                document.getElementById('status').className = 'status connected';
                document.getElementById('status').textContent = 'Connected';
            };

            ws.onclose = () => {
                document.getElementById('status').className = 'status disconnected';
                document.getElementById('status').textContent = 'Disconnected';
                setTimeout(connect, 1000);
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.type === 'color') {
                    updateAvatar(data.name, data.rgb);
                    addLog(`${data.name}: rgb(${data.rgb.join(', ')})`);
                } else if (data.type === 'init') {
                    // Initialize with existing colors
                    for (const [name, rgb] of Object.entries(data.colors)) {
                        updateAvatar(name, rgb);
                    }
                }
            };
        }

        function updateAvatar(name, rgb) {
            avatars[name] = rgb;
            renderAvatars();
        }

        function renderAvatars() {
            const container = document.getElementById('avatars');
            if (Object.keys(avatars).length === 0) {
                container.innerHTML = `
                    <div class="avatar">
                        <div class="avatar-circle" style="background: #444"></div>
                        <div class="avatar-name">Waiting for OSC...</div>
                        <div class="avatar-rgb">No messages yet</div>
                    </div>
                `;
                return;
            }

            container.innerHTML = Object.entries(avatars).map(([name, rgb]) => `
                <div class="avatar">
                    <div class="avatar-circle" style="background: rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})"></div>
                    <div class="avatar-name">${name}</div>
                    <div class="avatar-rgb">rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})</div>
                </div>
            `).join('');
        }

        function addLog(message) {
            const log = document.getElementById('log');
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            entry.textContent = `${new Date().toLocaleTimeString()} - ${message}`;
            log.insertBefore(entry, log.firstChild);

            // Keep only last 50 entries
            while (log.children.length > 50) {
                log.removeChild(log.lastChild);
            }
        }

        connect();
    </script>
</body>
</html>
"""


async def websocket_handler(request):
    """Handle WebSocket connections."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    websocket_clients.add(ws)
    print(f"🌐 WebSocket client connected ({len(websocket_clients)} total)")

    # Send current state
    await ws.send_json({"type": "init", "colors": avatar_colors})

    try:
        async for msg in ws:
            pass  # We don't expect messages from client
    finally:
        websocket_clients.discard(ws)
        print(f"🌐 WebSocket client disconnected ({len(websocket_clients)} total)")

    return ws


async def index_handler(request):
    """Serve the HTML page."""
    return web.Response(text=HTML_PAGE, content_type='text/html')


async def broadcast_color(name: str, rgb: list):
    """Broadcast color change to all WebSocket clients."""
    message = json.dumps({"type": "color", "name": name, "rgb": rgb})

    dead_clients = set()
    for ws in websocket_clients:
        try:
            await ws.send_str(message)
        except:
            dead_clients.add(ws)

    websocket_clients.difference_update(dead_clients)


def handle_avatar_color(address, *args):
    """Handle /avatar/color OSC messages."""
    print(f"📨 OSC: {address} {args}")

    if len(args) >= 4:
        # Format: name, r, g, b
        name = str(args[0])
        r, g, b = int(args[1]), int(args[2]), int(args[3])
    elif len(args) >= 2 and isinstance(args[1], (list, tuple)):
        # Format: name, [r, g, b]
        name = str(args[0])
        r, g, b = int(args[1][0]), int(args[1][1]), int(args[1][2])
    elif len(args) == 1 and isinstance(args[0], str):
        # Just a name, use default color
        name = args[0]
        r, g, b = 128, 128, 128
    else:
        print(f"⚠️ Unexpected args format: {args}")
        return

    # Clamp values
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))

    avatar_colors[name] = [r, g, b]
    print(f"🎨 {name} -> rgb({r}, {g}, {b})")

    # Broadcast to web clients (need to use asyncio from sync context)
    try:
        loop = asyncio.get_event_loop()
        asyncio.run_coroutine_threadsafe(broadcast_color(name, [r, g, b]), loop)
    except:
        pass


def run_osc_server(port: int):
    """Run the OSC server in a thread."""
    disp = dispatcher.Dispatcher()
    disp.map("/avatar/color", handle_avatar_color)
    disp.set_default_handler(lambda addr, *args: print(f"📨 OSC (unhandled): {addr} {args}"))

    server = osc_server.ThreadingOSCUDPServer(("0.0.0.0", port), disp)
    print(f"🎛️  OSC server listening on port {port}")
    server.serve_forever()


async def main():
    """Run both OSC and web servers."""
    import sys
    osc_port = int(sys.argv[1]) if len(sys.argv) > 1 else 7001
    web_port = int(sys.argv[2]) if len(sys.argv) > 2 else 8082

    # Start OSC server in background thread
    osc_thread = threading.Thread(target=run_osc_server, args=(osc_port,), daemon=True)
    osc_thread.start()

    # Start web server
    app = web.Application()
    app.router.add_get('/', index_handler)
    app.router.add_get('/ws', websocket_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', web_port)
    await site.start()

    print(f"🌐 Web UI at http://localhost:{web_port}")
    print(f"🎛️  OSC server at port {osc_port}")
    print(f"\nExpecting messages like: /avatar/color [\"Sonnet45\", 255, 0, 0]")
    print("Press Ctrl+C to stop\n")

    # Keep running
    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
