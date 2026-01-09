#!/usr/bin/env python3
"""
DMX Controller and OSC Bridge for Melodeus
==========================================

Controls:
- DMX fixtures (par cans, moving heads) via Art-Net
- Behringer X32 mixer via OSC
- Receives speaking events from melodeus OSC
- Web dashboard with manual overrides

Each character/mannequin has:
- A par can fixture that brightens when speaking
- X32 channel for individual speaker output
- Optional moving head positions

Requirements:
    pip install python-osc pyartnet aiohttp

Usage:
    python dmx_osc_bridge.py [--config dmx_config.yaml]
    
Dashboard: http://localhost:8090
"""

import asyncio
import argparse
import yaml
import time
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path

# OSC
from pythonosc import udp_client
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import AsyncIOOSCUDPServer

# Web dashboard
try:
    from aiohttp import web
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    print("⚠️  aiohttp not installed - Web dashboard disabled")
    print("   Install with: pip install aiohttp")

# DMX via Art-Net
try:
    from pyartnet import ArtNetNode
    from pyartnet.impl_artnet.node import UnicastNetworkTarget
    ARTNET_AVAILABLE = True
except ImportError:
    ARTNET_AVAILABLE = False
    print("⚠️  pyartnet not installed - DMX output disabled")
    print("   Install with: pip install pyartnet")


@dataclass
class ParCanFixture:
    """Par can fixture configuration.
    
    Supports 8-channel RGBLAU fixtures:
      Ch1: Red, Ch2: Green, Ch3: Blue, Ch4: Lime, Ch5: Amber, Ch6: UV, Ch7: Shutter, Ch8: Dimmer
    """
    dmx_address: int  # Starting DMX address
    channels: int = 8  # 8-channel RGBLAU+Shutter+Dimmer
    channel_map: Dict[str, int] = field(default_factory=lambda: {
        "red": 0, "green": 1, "blue": 2, "lime": 3, "amber": 4, "uv": 5, "shutter": 6, "dimmer": 7
    })
    color: Tuple[int, ...] = (255, 200, 150, 100, 120, 0)  # R, G, B, Lime, Amber, UV
    shutter_on: int = 63  # DMX value for shutter open (32-63 = On)
    idle_brightness: float = 0.1  # 10% when idle
    active_brightness: float = 1.0  # 100% when speaking
    fade_time: float = 0.3  # Fade time in seconds


@dataclass
class MovingHeadFixture:
    """Moving head fixture configuration."""
    dmx_address: int
    channels: int = 16  # Typical moving head
    channel_map: Dict[str, int] = field(default_factory=lambda: {
        "pan": 0, "pan_fine": 1,
        "tilt": 2, "tilt_fine": 3,
        "speed": 4,
        "dimmer": 5,
        "shutter": 6,
        "color": 7,
        "gobo": 8,
    })
    # Positions are (pan, tilt) in 0-255 range
    home_position: Tuple[int, int] = (128, 128)
    idle_dimmer: int = 0


@dataclass
class CharacterFixtures:
    """Fixtures assigned to a character."""
    par_can: Optional[ParCanFixture] = None
    moving_heads: List[Tuple[MovingHeadFixture, Tuple[int, int]]] = field(default_factory=list)
    # Each moving head has a target position (pan, tilt) for this character


@dataclass 
class X32Channel:
    """X32 mixer channel configuration."""
    channel: int  # 1-32 for channels, or bus number
    channel_type: str = "ch"  # "ch", "bus", "auxin", "fxrtn"
    idle_fader: float = 0.0  # 0.0-1.0
    active_fader: float = 0.75
    mute_when_idle: bool = True
    delay_ms: float = 0.0  # Delay in milliseconds (for main speakers)


# ============================================================================
# WEB DASHBOARD
# ============================================================================

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DMX/OSC Bridge Dashboard</title>
    <style>
        :root {
            --bg-primary: #0a0a12;
            --bg-secondary: #12121f;
            --bg-card: #1a1a2e;
            --accent: #6366f1;
            --accent-glow: rgba(99, 102, 241, 0.3);
            --text-primary: #e0e0e0;
            --text-muted: #888;
            --success: #22c55e;
            --warning: #eab308;
            --danger: #ef4444;
        }
        
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            font-family: 'JetBrains Mono', 'SF Mono', monospace;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 20px;
        }
        
        h1 {
            text-align: center;
            margin-bottom: 20px;
            font-size: 1.5rem;
            color: var(--accent);
            text-transform: uppercase;
            letter-spacing: 3px;
        }
        
        .status-bar {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-bottom: 30px;
            padding: 15px;
            background: var(--bg-secondary);
            border-radius: 8px;
        }
        
        .status-item {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
        }
        
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--danger);
        }
        
        .status-dot.connected { background: var(--success); }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            max-width: 1800px;
            margin: 0 auto;
        }
        
        .mannequin-card {
            background: var(--bg-card);
            border-radius: 12px;
            padding: 20px;
            position: relative;
            transition: all 0.3s ease;
            border: 2px solid transparent;
        }
        
        .mannequin-card.speaking {
            border-color: var(--accent);
            box-shadow: 0 0 30px var(--accent-glow);
        }
        
        .mannequin-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .mannequin-name {
            font-size: 1.2rem;
            font-weight: 700;
            text-transform: uppercase;
        }
        
        .speaking-badge {
            background: var(--accent);
            color: white;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 10px;
            font-weight: 600;
            opacity: 0;
            transition: opacity 0.2s;
        }
        
        .mannequin-card.speaking .speaking-badge {
            opacity: 1;
        }
        
        .mannequin-card.thinking {
            border-color: #eab308;
            box-shadow: 0 0 30px rgba(234, 179, 8, 0.3);
        }
        
        .thinking-badge {
            background: #eab308;
            color: black;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 10px;
            font-weight: 600;
            opacity: 0;
            transition: opacity 0.2s;
        }
        
        .mannequin-card.thinking .thinking-badge {
            opacity: 1;
            animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }
        
        .color-preview {
            height: 40px;
            border-radius: 8px;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            transition: all 0.3s;
        }
        
        .controls {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        
        .control-row {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .control-row label {
            width: 80px;
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
        }
        
        .control-row input[type="range"] {
            flex: 1;
            height: 6px;
            -webkit-appearance: none;
            background: var(--bg-secondary);
            border-radius: 3px;
            cursor: pointer;
        }
        
        .control-row input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            background: var(--accent);
            border-radius: 50%;
            cursor: pointer;
        }
        
        .control-row input[type="color"] {
            width: 40px;
            height: 30px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        
        .value-display {
            width: 50px;
            text-align: right;
            font-size: 12px;
            color: var(--text-muted);
        }
        
        .btn-row {
            display: flex;
            gap: 8px;
            margin-top: 10px;
        }
        
        .btn {
            flex: 1;
            padding: 10px;
            border: none;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            text-transform: uppercase;
            transition: all 0.2s;
        }
        
        .btn-activate {
            background: var(--success);
            color: white;
        }
        
        .btn-activate:hover { background: #16a34a; }
        
        .btn-deactivate {
            background: var(--bg-secondary);
            color: var(--text-muted);
            border: 1px solid #333;
        }
        
        .btn-deactivate:hover { background: #252540; }
        
        .btn-blackout {
            background: var(--danger);
            color: white;
        }
        
        .btn-blackout:hover { background: #dc2626; }
        
        .x32-status {
            margin-top: 10px;
            padding: 10px;
            background: var(--bg-secondary);
            border-radius: 6px;
            font-size: 11px;
        }
        
        .x32-status .label {
            color: var(--text-muted);
        }
        
        .global-controls {
            margin-top: 30px;
            padding: 20px;
            background: var(--bg-secondary);
            border-radius: 12px;
            max-width: 600px;
            margin-left: auto;
            margin-right: auto;
        }
        
        .global-controls h2 {
            font-size: 1rem;
            margin-bottom: 15px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        
        .global-btn-row {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        
        .global-btn-row .btn {
            flex: 1;
            min-width: 120px;
        }
        
        .log-panel {
            margin-top: 30px;
            max-width: 1200px;
            margin-left: auto;
            margin-right: auto;
        }
        
        .log-panel h2 {
            font-size: 1rem;
            margin-bottom: 10px;
            color: var(--text-muted);
        }
        
        .log-content {
            background: var(--bg-card);
            border-radius: 8px;
            padding: 15px;
            height: 150px;
            overflow-y: auto;
            font-size: 11px;
            font-family: monospace;
        }
        
        .log-entry {
            padding: 3px 0;
            border-bottom: 1px solid #222;
        }
        
        .log-entry.event { color: var(--accent); }
        .log-entry.warning { color: var(--warning); }
        .log-entry.error { color: var(--danger); }
    </style>
</head>
<body>
    <h1>🎭 DMX/OSC Bridge</h1>
    
    <div class="status-bar">
        <div class="status-item">
            <div class="status-dot" id="ws-status"></div>
            <span>Dashboard</span>
        </div>
        <div class="status-item">
            <div class="status-dot" id="artnet-status"></div>
            <span>Art-Net</span>
        </div>
        <div class="status-item">
            <div class="status-dot" id="x32-status"></div>
            <span>X32</span>
        </div>
        <div class="status-item">
            <div class="status-dot" id="melodeus-status"></div>
            <span>Melodeus OSC</span>
        </div>
    </div>
    
    <div class="grid" id="mannequin-grid">
        <!-- Cards generated by JS -->
    </div>
    
    <div class="global-controls">
        <h2>Global Controls</h2>
        <div class="global-btn-row">
            <button class="btn btn-blackout" onclick="sendCommand('blackout')">⬛ Blackout</button>
            <button class="btn btn-activate" onclick="sendCommand('all_idle')">🌙 All Idle</button>
            <button class="btn btn-deactivate" onclick="sendCommand('all_full')">☀️ All Full</button>
            <button class="btn btn-deactivate" onclick="sendCommand('x32_refresh')">🔄 Refresh X32</button>
        </div>
    </div>
    
    <div class="log-panel">
        <h2>Event Log</h2>
        <div class="log-content" id="log-content"></div>
    </div>
    
    <script>
        let ws;
        let state = { characters: {}, speaking: [] };
        
        function connect() {
            ws = new WebSocket('ws://' + location.host + '/ws');
            
            ws.onopen = () => {
                document.getElementById('ws-status').classList.add('connected');
                log('Connected to bridge', 'event');
            };
            
            ws.onclose = () => {
                document.getElementById('ws-status').classList.remove('connected');
                log('Disconnected - reconnecting...', 'warning');
                setTimeout(connect, 2000);
            };
            
            ws.onmessage = (e) => {
                const msg = JSON.parse(e.data);
                handleMessage(msg);
            };
        }
        
        function handleMessage(msg) {
            switch(msg.type) {
                case 'state':
                    state = msg.data;
                    renderGrid();
                    updateStatuses(msg.data);
                    break;
                case 'thinking_start':
                    log(`🤔 ${msg.character} started thinking`, 'event');
                    document.getElementById('card-' + msg.character)?.classList.add('thinking');
                    break;
                case 'thinking_stop':
                    log(`💭 ${msg.character} stopped thinking`);
                    document.getElementById('card-' + msg.character)?.classList.remove('thinking');
                    break;
                case 'speaking_start':
                    log(`🎤 ${msg.character} started speaking`, 'event');
                    document.getElementById('card-' + msg.character)?.classList.remove('thinking');
                    document.getElementById('card-' + msg.character)?.classList.add('speaking');
                    break;
                case 'speaking_stop':
                    log(`🔇 ${msg.character} stopped speaking`);
                    document.getElementById('card-' + msg.character)?.classList.remove('speaking');
                    break;
                case 'log':
                    log(msg.message, msg.level);
                    break;
            }
        }
        
        function updateStatuses(data) {
            document.getElementById('artnet-status').classList.toggle('connected', data.artnet_connected);
            document.getElementById('x32-status').classList.toggle('connected', data.x32_connected);
            document.getElementById('melodeus-status').classList.toggle('connected', data.osc_listening);
        }
        
        function renderGrid() {
            const grid = document.getElementById('mannequin-grid');
            grid.innerHTML = '';
            
            for (const [name, char] of Object.entries(state.characters)) {
                const isSpeaking = state.speaking.includes(name);
                const isThinking = (state.thinking || []).includes(name);
                const color = char.color || [255, 200, 150, 200];
                const rgbColor = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
                const brightness = char.current_brightness || 0.1;
                
                const card = document.createElement('div');
                let cardClass = 'mannequin-card';
                if (isSpeaking) cardClass += ' speaking';
                else if (isThinking) cardClass += ' thinking';
                card.className = cardClass;
                card.id = 'card-' + name;
                
                card.innerHTML = `
                    <div class="mannequin-header">
                        <span class="mannequin-name">${name}</span>
                        <span class="thinking-badge">THINKING</span>
                        <span class="speaking-badge">SPEAKING</span>
                    </div>
                    
                    <div class="color-preview" style="background: ${rgbColor}; opacity: ${brightness}">
                        DMX ${char.dmx_address || '?'}
                    </div>
                    
                    <div class="controls">
                        <div class="control-row">
                            <label>Brightness</label>
                            <input type="range" min="0" max="100" value="${brightness * 100}" 
                                   onchange="setBrightness('${name}', this.value/100)"
                                   oninput="this.nextElementSibling.textContent = this.value + '%'">
                            <span class="value-display">${Math.round(brightness * 100)}%</span>
                        </div>
                        
                        <div class="control-row">
                            <label>Color</label>
                            <input type="color" value="${rgbToHex(color[0], color[1], color[2])}"
                                   onchange="setColor('${name}', this.value)">
                            <span class="value-display">RGBW</span>
                        </div>
                        
                        ${char.x32_channel ? `
                        <div class="control-row">
                            <label>X32 Ch${char.x32_channel}</label>
                            <input type="range" min="0" max="100" value="${(char.x32_fader || 0) * 100}"
                                   onchange="setX32Fader('${name}', this.value/100)"
                                   oninput="this.nextElementSibling.textContent = this.value + '%'">
                            <span class="value-display">${Math.round((char.x32_fader || 0) * 100)}%</span>
                        </div>
                        ` : ''}
                    </div>
                    
                    <div class="btn-row">
                        <button class="btn btn-activate" onclick="activate('${name}')">Activate</button>
                        <button class="btn btn-deactivate" onclick="deactivate('${name}')">Idle</button>
                    </div>
                    
                    ${char.x32_channel ? `
                    <div class="x32-status">
                        <span class="label">X32:</span> Ch ${char.x32_channel} | 
                        Fader: ${Math.round((char.x32_fader || 0) * 100)}% |
                        ${char.x32_muted ? '🔇 Muted' : '🔊 On'}
                    </div>
                    ` : ''}
                `;
                
                grid.appendChild(card);
            }
        }
        
        function rgbToHex(r, g, b) {
            return '#' + [r, g, b].map(x => x.toString(16).padStart(2, '0')).join('');
        }
        
        function hexToRgb(hex) {
            const result = /^#?([a-f\\d]{2})([a-f\\d]{2})([a-f\\d]{2})$/i.exec(hex);
            return result ? [
                parseInt(result[1], 16),
                parseInt(result[2], 16),
                parseInt(result[3], 16),
                200
            ] : [255, 255, 255, 200];
        }
        
        function sendCommand(cmd, data = {}) {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ command: cmd, ...data }));
            }
        }
        
        function setBrightness(name, value) {
            sendCommand('set_brightness', { character: name, brightness: value });
        }
        
        function setColor(name, hex) {
            sendCommand('set_color', { character: name, color: hexToRgb(hex) });
        }
        
        function setX32Fader(name, value) {
            sendCommand('set_x32_fader', { character: name, fader: value });
        }
        
        function activate(name) {
            sendCommand('activate', { character: name });
        }
        
        function deactivate(name) {
            sendCommand('deactivate', { character: name });
        }
        
        function log(message, level = 'info') {
            const logEl = document.getElementById('log-content');
            const entry = document.createElement('div');
            entry.className = 'log-entry ' + level;
            const time = new Date().toLocaleTimeString();
            entry.textContent = `[${time}] ${message}`;
            logEl.insertBefore(entry, logEl.firstChild);
            
            // Keep max 100 entries
            while (logEl.children.length > 100) {
                logEl.removeChild(logEl.lastChild);
            }
        }
        
        // Start
        connect();
    </script>
</body>
</html>
"""


class DMXOSCBridge:
    """Main bridge between melodeus OSC events, DMX fixtures, and X32."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        
        # Character -> fixtures mapping
        self.character_fixtures: Dict[str, CharacterFixtures] = {}
        self.character_x32: Dict[str, X32Channel] = {}
        
        # State tracking
        self.speaking_characters: Set[str] = set()
        self.thinking_characters: Set[str] = set()
        self.pulse_tasks: Dict[str, asyncio.Task] = {}  # Character -> pulse task
        self.current_brightness: Dict[str, float] = {}
        self.current_x32_fader: Dict[str, float] = {}
        self.current_x32_muted: Dict[str, bool] = {}
        self.last_update: Dict[str, float] = {}
        
        # Clients
        self.artnet_node: Optional[ArtNetNode] = None
        self.artnet_universe = None
        self.x32_client: Optional[udp_client.SimpleUDPClient] = None
        self.dmx_channels: Dict[str, any] = {}  # Character name -> DMX channel
        
        # OSC server for receiving melodeus events
        self.osc_server = None
        self.osc_listening = False
        
        # Web dashboard
        self.web_app = None
        self.web_runner = None
        self.websockets: Set = set()
        
        self._setup_from_config()
    
    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load configuration from YAML file."""
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            "artnet": {
                "enabled": True,
                "host": "255.255.255.255",  # Broadcast
                "port": 6454,
                "universe": 0,
                "fps": 40,
            },
            "x32": {
                "enabled": True,
                "host": "192.168.1.100",  # X32 IP
                "port": 10023,
            },
            "melodeus_osc": {
                "listen_host": "0.0.0.0",
                "listen_port": 7000,
            },
            "characters": {
                # Example character mappings
                "Opus45": {
                    "par_can": {"dmx_address": 1, "color": [255, 100, 50, 200]},
                    "x32_channel": 1,
                },
                "Opus4": {
                    "par_can": {"dmx_address": 5, "color": [200, 150, 255, 150]},
                    "x32_channel": 2,
                },
                "Sonnet4": {
                    "par_can": {"dmx_address": 9, "color": [100, 255, 150, 200]},
                    "x32_channel": 3,
                },
                "Haiku45": {
                    "par_can": {"dmx_address": 13, "color": [255, 255, 100, 255]},
                    "x32_channel": 4,
                },
            },
            "moving_heads": {
                # Moving heads that point at characters
                "mover_1": {
                    "dmx_address": 100,
                    "character_positions": {
                        "Opus45": [64, 128],
                        "Opus4": [96, 128],
                        "Sonnet4": [160, 128],
                        "Haiku45": [192, 128],
                    }
                },
                "mover_2": {
                    "dmx_address": 116,
                    "character_positions": {
                        "Opus45": [192, 128],
                        "Opus4": [160, 128],
                        "Sonnet4": [96, 128],
                        "Haiku45": [64, 128],
                    }
                },
            },
            "timing": {
                "fade_time": 0.3,
                "mover_speed": 50,  # DMX value for pan/tilt speed
            }
        }
    
    def _setup_from_config(self):
        """Setup fixtures and clients from config."""
        cfg = self.config
        
        # Setup character fixtures
        for char_name, char_cfg in cfg.get("characters", {}).items():
            fixtures = CharacterFixtures()
            
            # Par can
            if "par_can" in char_cfg:
                pc = char_cfg["par_can"]
                color = tuple(pc.get("color", [255, 200, 150, 100, 120, 0]))
                channel_map = pc.get("channel_map", {
                    "red": 0, "green": 1, "blue": 2, "lime": 3, "amber": 4, "uv": 5, "shutter": 6, "dimmer": 7
                })
                fixtures.par_can = ParCanFixture(
                    dmx_address=pc["dmx_address"],
                    channels=pc.get("channels", 8),
                    channel_map=channel_map,
                    color=color,
                    shutter_on=pc.get("shutter_on", 63),
                    idle_brightness=pc.get("idle_brightness", 0.1),
                    active_brightness=pc.get("active_brightness", 1.0),
                    fade_time=cfg.get("timing", {}).get("fade_time", 0.3),
                )
            
            self.character_fixtures[char_name] = fixtures
            
            # X32 channel
            if "x32_channel" in char_cfg:
                self.character_x32[char_name] = X32Channel(
                    channel=char_cfg["x32_channel"],
                    channel_type=char_cfg.get("x32_type", "ch"),
                    idle_fader=char_cfg.get("x32_idle_fader", 0.0),
                    active_fader=char_cfg.get("x32_active_fader", 0.75),
                    mute_when_idle=char_cfg.get("x32_mute_idle", True),
                )
        
        # Setup moving heads
        for mover_name, mover_cfg in cfg.get("moving_heads", {}).items():
            mover = MovingHeadFixture(
                dmx_address=mover_cfg["dmx_address"],
                channels=mover_cfg.get("channels", 16),
            )
            
            # Assign to characters with positions
            for char_name, pos in mover_cfg.get("character_positions", {}).items():
                if char_name in self.character_fixtures:
                    self.character_fixtures[char_name].moving_heads.append(
                        (mover, tuple(pos))
                    )
        
        print(f"📋 Configured {len(self.character_fixtures)} characters")
        for name, fix in self.character_fixtures.items():
            par = f"par@{fix.par_can.dmx_address}" if fix.par_can else "no par"
            movers = len(fix.moving_heads)
            x32 = f"X32 ch{self.character_x32[name].channel}" if name in self.character_x32 else "no X32"
            print(f"   {name}: {par}, {movers} movers, {x32}")
    
    async def start(self):
        """Start the bridge."""
        print("\n🎭 DMX/OSC Bridge Starting...")
        print("=" * 50)
        
        # Start Art-Net
        if ARTNET_AVAILABLE and self.config["artnet"]["enabled"]:
            await self._start_artnet()
        
        # Start X32 client
        if self.config["x32"]["enabled"]:
            self._start_x32_client()
        
        # Start OSC server to receive melodeus events
        await self._start_osc_server()
        
        # Start web dashboard
        if AIOHTTP_AVAILABLE:
            await self._start_web_dashboard()
        
        # Set initial state (all idle)
        await self._set_all_idle()
        
        print("\n✅ Bridge ready!")
        print(f"   Listening for melodeus OSC on port {self.config['melodeus_osc']['listen_port']}")
        if AIOHTTP_AVAILABLE:
            print(f"   Dashboard: http://localhost:{self.config.get('dashboard', {}).get('port', 8090)}")
        
        # Run forever
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            await self.stop()
    
    # =========================================================================
    # WEB DASHBOARD
    # =========================================================================
    
    async def _start_web_dashboard(self):
        """Start the web dashboard server."""
        port = self.config.get("dashboard", {}).get("port", 8090)
        
        self.web_app = web.Application()
        self.web_app.router.add_get('/', self._handle_dashboard)
        self.web_app.router.add_get('/ws', self._handle_websocket)
        
        self.web_runner = web.AppRunner(self.web_app)
        await self.web_runner.setup()
        site = web.TCPSite(self.web_runner, '0.0.0.0', port)
        await site.start()
        print(f"🖥️  Dashboard started on http://0.0.0.0:{port}")
    
    async def _handle_dashboard(self, request):
        """Serve the dashboard HTML."""
        return web.Response(text=DASHBOARD_HTML, content_type='text/html')
    
    async def _handle_websocket(self, request):
        """Handle WebSocket connections for live updates."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.websockets.add(ws)
        
        # Send initial state
        await ws.send_json({
            "type": "state",
            "data": self._get_dashboard_state()
        })
        
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    await self._handle_dashboard_command(data, ws)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    print(f"Dashboard WS error: {ws.exception()}")
        finally:
            self.websockets.discard(ws)
        
        return ws
    
    async def _handle_dashboard_command(self, data: dict, ws):
        """Handle commands from the dashboard."""
        cmd = data.get("command")
        
        if cmd == "activate":
            await self._activate_character(data["character"])
        elif cmd == "deactivate":
            await self._deactivate_character(data["character"])
        elif cmd == "set_brightness":
            await self._set_character_brightness(data["character"], data["brightness"])
        elif cmd == "set_color":
            await self._set_character_color(data["character"], data["color"])
        elif cmd == "set_x32_fader":
            await self._set_x32_fader_manual(data["character"], data["fader"])
        elif cmd == "blackout":
            await self._blackout()
        elif cmd == "all_idle":
            await self._set_all_idle()
        elif cmd == "all_full":
            await self._set_all_full()
        elif cmd == "x32_refresh":
            self._x32_refresh_all()
        
        # Broadcast updated state
        await self._broadcast_state()
    
    def _get_dashboard_state(self) -> dict:
        """Get current state for dashboard."""
        characters = {}
        for name, fixtures in self.character_fixtures.items():
            char_state = {
                "current_brightness": self.current_brightness.get(name, 0.1),
            }
            if fixtures.par_can:
                char_state["dmx_address"] = fixtures.par_can.dmx_address
                char_state["color"] = list(fixtures.par_can.color)
            if name in self.character_x32:
                x32 = self.character_x32[name]
                char_state["x32_channel"] = x32.channel
                char_state["x32_fader"] = self.current_x32_fader.get(name, 0)
                char_state["x32_muted"] = self.current_x32_muted.get(name, True)
            characters[name] = char_state
        
        return {
            "characters": characters,
            "speaking": list(self.speaking_characters),
            "thinking": list(self.thinking_characters),
            "artnet_connected": self.artnet_node is not None,
            "x32_connected": self.x32_client is not None,
            "osc_listening": self.osc_listening,
        }
    
    async def _broadcast_state(self):
        """Broadcast state to all dashboard clients."""
        if not self.websockets:
            return
        
        state = {
            "type": "state",
            "data": self._get_dashboard_state()
        }
        
        for ws in list(self.websockets):
            try:
                await ws.send_json(state)
            except Exception:
                self.websockets.discard(ws)
    
    async def _broadcast_event(self, event_type: str, **kwargs):
        """Broadcast an event to all dashboard clients."""
        msg = {"type": event_type, **kwargs}
        for ws in list(self.websockets):
            try:
                await ws.send_json(msg)
            except Exception:
                self.websockets.discard(ws)
    
    async def _set_character_brightness(self, character: str, brightness: float):
        """Manually set character brightness."""
        if character not in self.character_fixtures:
            return
        
        fixtures = self.character_fixtures[character]
        self.current_brightness[character] = brightness
        
        if fixtures.par_can and character in self.dmx_channels:
            await self._set_par_brightness_for_character(character, brightness)
    
    async def _set_character_color(self, character: str, color: List[int]):
        """Manually set character color."""
        if character not in self.character_fixtures:
            return
        
        fixtures = self.character_fixtures[character]
        if fixtures.par_can:
            fixtures.par_can.color = tuple(color)
            brightness = self.current_brightness.get(character, 0.1)
            if character in self.dmx_channels:
                await self._set_par_brightness_for_character(character, brightness)
    
    async def _set_x32_fader_manual(self, character: str, fader: float):
        """Manually set X32 fader."""
        if character not in self.character_x32:
            return
        
        x32 = self.character_x32[character]
        self.current_x32_fader[character] = fader
        
        if self.x32_client:
            prefix = f"/{x32.channel_type}/{x32.channel:02d}"
            self.x32_client.send_message(f"{prefix}/mix/fader", fader)
    
    async def _blackout(self):
        """Blackout all fixtures."""
        print("⬛ BLACKOUT")
        for char_name in self.character_fixtures:
            self.current_brightness[char_name] = 0
            if char_name in self.dmx_channels:
                await self._set_par_brightness_for_character(char_name, 0)
        await self._broadcast_event("log", message="BLACKOUT", level="warning")
    
    async def _set_all_full(self):
        """Set all fixtures to full brightness."""
        print("☀️ ALL FULL")
        for char_name in self.character_fixtures:
            await self._activate_character(char_name)
    
    async def _start_artnet(self):
        """Initialize Art-Net node."""
        cfg = self.config["artnet"]
        print(f"🔌 Starting Art-Net to {cfg['host']}:{cfg['port']} universe {cfg['universe']}")
        
        # Create network target for unicast to the eDMX node
        # Bind to source IP on the same subnet to ensure correct interface
        source_ip = cfg.get("source_ip")  # Optional: specify source interface
        network = UnicastNetworkTarget.create(cfg["host"], cfg["port"], source_ip=source_ip)
        
        self.artnet_node = ArtNetNode(
            network,
            max_fps=cfg["fps"],
        )
        self.artnet_universe = self.artnet_node.add_universe(cfg["universe"])
        
        # Create a channel for each character's par can
        for char_name, fixtures in self.character_fixtures.items():
            if fixtures.par_can:
                # Create a channel for this par can
                channel = self.artnet_universe.add_channel(
                    start=fixtures.par_can.dmx_address,
                    width=fixtures.par_can.channels
                )
                self.dmx_channels[char_name] = channel
                print(f"   💡 {char_name}: DMX {fixtures.par_can.dmx_address}-{fixtures.par_can.dmx_address + fixtures.par_can.channels - 1}")
        
        # Enter async context and start refresh
        await self.artnet_node.__aenter__()
        self.artnet_node.start_refresh()
        print(f"   ✅ Art-Net started, {len(self.dmx_channels)} fixtures")
    
    def _start_x32_client(self):
        """Initialize X32 OSC client."""
        cfg = self.config["x32"]
        print(f"🎚️  Connecting to X32 at {cfg['host']}:{cfg['port']}")
        
        self.x32_client = udp_client.SimpleUDPClient(cfg["host"], cfg["port"])
        print("   ✅ X32 client ready")
    
    async def _start_osc_server(self):
        """Start OSC server to receive melodeus events."""
        cfg = self.config["melodeus_osc"]
        
        dispatcher = Dispatcher()
        dispatcher.map("/character/speaking/start", self._on_speaking_start)
        dispatcher.map("/character/speaking/stop", self._on_speaking_stop)
        dispatcher.map("/character/thinking/start", self._on_thinking_start)
        dispatcher.map("/character/thinking/stop", self._on_thinking_stop)
        dispatcher.map("/test", self._on_test)
        dispatcher.set_default_handler(self._on_unknown)
        
        self.osc_server = AsyncIOOSCUDPServer(
            (cfg["listen_host"], cfg["listen_port"]),
            dispatcher,
            asyncio.get_event_loop()
        )
        
        transport, protocol = await self.osc_server.create_serve_endpoint()
        self.osc_listening = True
        print(f"📡 OSC server listening on {cfg['listen_host']}:{cfg['listen_port']}")
    
    def _on_test(self, address, *args):
        """Handle test OSC message."""
        print(f"🧪 Test message: {address} {args}")
    
    def _on_unknown(self, address, *args):
        """Handle unknown OSC messages."""
        print(f"❓ Unknown OSC: {address} {args}")
    
    def _on_speaking_start(self, address, character_name):
        """Handle character starting to speak."""
        print(f"🎤 {character_name} started speaking")
        self.speaking_characters.add(character_name)
        
        # Schedule the async update
        asyncio.create_task(self._on_speaking_start_async(character_name))
    
    async def _on_speaking_start_async(self, character_name: str):
        """Async handler for speaking start."""
        # Stop any pulsing from thinking state
        self.thinking_characters.discard(character_name)
        if character_name in self.pulse_tasks:
            self.pulse_tasks[character_name].cancel()
            try:
                await self.pulse_tasks[character_name]
            except asyncio.CancelledError:
                pass
            del self.pulse_tasks[character_name]
        
        # Activate with steady light
        await self._activate_character(character_name)
        await self._broadcast_event("speaking_start", character=character_name)
        await self._broadcast_state()
    
    def _on_speaking_stop(self, address, character_name):
        """Handle character stopping speaking."""
        print(f"🔇 {character_name} stopped speaking")
        self.speaking_characters.discard(character_name)
        
        # Schedule the async update
        asyncio.create_task(self._on_speaking_stop_async(character_name))
    
    async def _on_speaking_stop_async(self, character_name: str):
        """Async handler for speaking stop."""
        await self._deactivate_character(character_name)
        await self._broadcast_event("speaking_stop", character=character_name)
        await self._broadcast_state()
    
    def _on_thinking_start(self, address, character_name):
        """Handle character starting to think (waiting for LLM)."""
        print(f"🤔 {character_name} started thinking")
        self.thinking_characters.add(character_name)
        
        # Schedule the async pulsing
        asyncio.create_task(self._on_thinking_start_async(character_name))
    
    async def _on_thinking_start_async(self, character_name: str):
        """Start pulsing animation for thinking character."""
        # Cancel any existing pulse task for this character
        if character_name in self.pulse_tasks:
            self.pulse_tasks[character_name].cancel()
            try:
                await self.pulse_tasks[character_name]
            except asyncio.CancelledError:
                pass
        
        # Start new pulse task
        task = asyncio.create_task(self._pulse_character(character_name))
        self.pulse_tasks[character_name] = task
        
        # Also raise the X32 bus fader so thinking sound comes from mannequin speaker
        if character_name in self.character_x32 and self.x32_client:
            x32 = self.character_x32[character_name]
            self._x32_set_channel(x32, character_name, active=True)
        
        await self._broadcast_event("thinking_start", character=character_name)
        await self._broadcast_state()
    
    def _on_thinking_stop(self, address, character_name):
        """Handle character stopping thinking."""
        print(f"💭 {character_name} stopped thinking")
        self.thinking_characters.discard(character_name)
        
        # Schedule the async stop
        asyncio.create_task(self._on_thinking_stop_async(character_name))
    
    async def _on_thinking_stop_async(self, character_name: str):
        """Stop pulsing animation for character."""
        # Cancel pulse task
        if character_name in self.pulse_tasks:
            self.pulse_tasks[character_name].cancel()
            try:
                await self.pulse_tasks[character_name]
            except asyncio.CancelledError:
                pass
            del self.pulse_tasks[character_name]
        
        await self._broadcast_event("thinking_stop", character=character_name)
        await self._broadcast_state()
    
    async def _pulse_character(self, character_name: str):
        """Pulse the par can brightness for a thinking character."""
        if character_name not in self.character_fixtures:
            return
        
        fixtures = self.character_fixtures[character_name]
        if not fixtures.par_can or character_name not in self.dmx_channels:
            return
        
        par = fixtures.par_can
        min_brightness = par.idle_brightness
        max_brightness = 0.6  # Pulse up to 60%
        pulse_period = 0.6  # Seconds for one full cycle (faster to match thinking sound)
        
        print(f"   ✨ Starting pulse for {character_name}")
        
        try:
            while character_name in self.thinking_characters:
                # Fade up
                await self._set_par_brightness_for_character(character_name, max_brightness)
                await asyncio.sleep(pulse_period / 2)
                
                if character_name not in self.thinking_characters:
                    break
                
                # Fade down
                await self._set_par_brightness_for_character(character_name, min_brightness)
                await asyncio.sleep(pulse_period / 2)
        except asyncio.CancelledError:
            pass
        finally:
            print(f"   ✨ Stopped pulse for {character_name}")
    
    async def _activate_character(self, character_name: str):
        """Activate fixtures for speaking character."""
        if character_name not in self.character_fixtures:
            print(f"   ⚠️ Unknown character: {character_name}")
            return
        
        fixtures = self.character_fixtures[character_name]
        
        # Brighten par can
        if fixtures.par_can:
            brightness = fixtures.par_can.active_brightness
            self.current_brightness[character_name] = brightness
            if character_name in self.dmx_channels:
                await self._set_par_brightness_for_character(character_name, brightness)
        
        # Move moving heads
        for mover, position in fixtures.moving_heads:
            await self._move_head(mover, position)
        
        # X32: unmute and raise fader
        if character_name in self.character_x32 and self.x32_client:
            x32 = self.character_x32[character_name]
            self._x32_set_channel(x32, character_name, active=True)
    
    async def _deactivate_character(self, character_name: str):
        """Deactivate fixtures when character stops speaking."""
        if character_name not in self.character_fixtures:
            return
        
        fixtures = self.character_fixtures[character_name]
        
        # Dim par can
        if fixtures.par_can:
            brightness = fixtures.par_can.idle_brightness
            self.current_brightness[character_name] = brightness
            if character_name in self.dmx_channels:
                await self._set_par_brightness_for_character(character_name, brightness)
        
        # X32: mute and lower fader
        if character_name in self.character_x32 and self.x32_client:
            x32 = self.character_x32[character_name]
            self._x32_set_channel(x32, character_name, active=False)
    
    async def _set_par_brightness_for_character(self, character_name: str, brightness: float):
        """Set par can brightness for a character."""
        if character_name not in self.dmx_channels:
            return
        
        fixtures = self.character_fixtures.get(character_name)
        if not fixtures or not fixtures.par_can:
            return
        
        par = fixtures.par_can
        channel = self.dmx_channels[character_name]
        
        # Get color values (supports RGBLAU - Red, Green, Blue, Lime, Amber, UV)
        color = par.color
        r = color[0] if len(color) > 0 else 0
        g = color[1] if len(color) > 1 else 0
        b = color[2] if len(color) > 2 else 0
        lime = color[3] if len(color) > 3 else 0
        amber = color[4] if len(color) > 4 else 0
        uv = color[5] if len(color) > 5 else 0
        
        # Build values array in channel order
        values = [0] * par.channels
        
        # Set color channels (scaled by brightness)
        if "red" in par.channel_map:
            values[par.channel_map["red"]] = int(r * brightness)
        if "green" in par.channel_map:
            values[par.channel_map["green"]] = int(g * brightness)
        if "blue" in par.channel_map:
            values[par.channel_map["blue"]] = int(b * brightness)
        if "lime" in par.channel_map:
            values[par.channel_map["lime"]] = int(lime * brightness)
        if "amber" in par.channel_map:
            values[par.channel_map["amber"]] = int(amber * brightness)
        if "uv" in par.channel_map:
            values[par.channel_map["uv"]] = int(uv * brightness)
        if "white" in par.channel_map:
            w = color[3] if len(color) > 3 else 0  # For 4-channel RGBW fixtures
            values[par.channel_map["white"]] = int(w * brightness)
        
        # Shutter: open when brightness > 0, closed when off
        if "shutter" in par.channel_map:
            shutter_on = getattr(par, 'shutter_on', 63)  # Default to "On" value
            values[par.channel_map["shutter"]] = shutter_on if brightness > 0 else 0
        
        # Master dimmer: full when active (color channels already handle brightness)
        if "dimmer" in par.channel_map:
            values[par.channel_map["dimmer"]] = 255 if brightness > 0 else 0
        
        # Set with fade
        fade_ms = int(par.fade_time * 1000)
        channel.add_fade(values, fade_ms)
        
        print(f"   💡 {character_name} @{par.dmx_address}: RGB=({int(r*brightness)},{int(g*brightness)},{int(b*brightness)}) dim={brightness:.0%}")
    
    async def _move_head(self, mover: MovingHeadFixture, position: Tuple[int, int]):
        """Move a moving head to position."""
        # TODO: Moving heads need separate channel setup
        # For now, just log the intent
        pan, tilt = position
        print(f"   🔦 Mover @{mover.dmx_address}: pan={pan} tilt={tilt} (not implemented yet)")
    
    def _x32_set_channel(self, x32: X32Channel, character_name: str, active: bool):
        """Set X32 channel fader and mute state."""
        if not self.x32_client:
            return
        
        # Build OSC address based on channel type
        prefix = f"/{x32.channel_type}/{x32.channel:02d}"
        
        # Set fader
        fader = x32.active_fader if active else x32.idle_fader
        self.x32_client.send_message(f"{prefix}/mix/fader", fader)
        self.current_x32_fader[character_name] = fader
        
        # Set mute (0 = unmuted, 1 = muted on X32)
        muted = not active if x32.mute_when_idle else False
        if x32.mute_when_idle:
            self.x32_client.send_message(f"{prefix}/mix/on", 0 if muted else 1)
        self.current_x32_muted[character_name] = muted
        
        state = "active" if active else "idle"
        print(f"   🎚️  X32 {x32.channel_type}{x32.channel}: {state} (fader={fader:.2f})")
    
    def _x32_refresh_all(self):
        """Refresh all X32 channel states."""
        if not self.x32_client:
            return
        
        print("🔄 Refreshing X32 channels...")
        for char_name, x32 in self.character_x32.items():
            is_speaking = char_name in self.speaking_characters
            self._x32_set_channel(x32, char_name, active=is_speaking)
    
    async def _set_all_idle(self):
        """Set all fixtures to idle state."""
        print("🌙 Setting all fixtures to idle...")
        for char_name in self.character_fixtures:
            await self._deactivate_character(char_name)
    
    async def stop(self):
        """Stop the bridge."""
        print("\n🛑 Stopping bridge...")
        
        # Close websockets
        for ws in list(self.websockets):
            await ws.close()
        
        # Stop web server
        if self.web_runner:
            await self.web_runner.cleanup()
        
        # Blackout all channels
        for char_name, channel in self.dmx_channels.items():
            fixtures = self.character_fixtures.get(char_name)
            if fixtures and fixtures.par_can:
                channel.set_values([0] * fixtures.par_can.channels)
        
        if self.artnet_node:
            self.artnet_node.stop_refresh()
            await self.artnet_node.__aexit__(None, None, None)


def create_example_config():
    """Create an example configuration file."""
    config = {
        "artnet": {
            "enabled": True,
            "host": "255.255.255.255",
            "port": 6454,
            "universe": 0,
            "fps": 40,
        },
        "x32": {
            "enabled": True,
            "host": "192.168.1.100",
            "port": 10023,
        },
        "melodeus_osc": {
            "listen_host": "0.0.0.0",
            "listen_port": 7000,
        },
        "characters": {
            "Opus45": {
                "par_can": {
                    "dmx_address": 1,
                    "channels": 4,
                    "color": [255, 150, 100, 200],
                    "idle_brightness": 0.1,
                    "active_brightness": 1.0,
                },
                "x32_channel": 1,
                "x32_idle_fader": 0.0,
                "x32_active_fader": 0.75,
            },
            "Opus4": {
                "par_can": {
                    "dmx_address": 5,
                    "color": [200, 100, 255, 150],
                },
                "x32_channel": 2,
            },
            "Sonnet4": {
                "par_can": {
                    "dmx_address": 9,
                    "color": [100, 255, 200, 180],
                },
                "x32_channel": 3,
            },
            "Haiku45": {
                "par_can": {
                    "dmx_address": 13,
                    "color": [255, 255, 100, 255],
                },
                "x32_channel": 4,
            },
        },
        "moving_heads": {
            "stage_left": {
                "dmx_address": 100,
                "channels": 16,
                "character_positions": {
                    "Opus45": [50, 128],
                    "Opus4": [85, 128],
                    "Sonnet4": [170, 128],
                    "Haiku45": [205, 128],
                },
            },
            "stage_right": {
                "dmx_address": 116,
                "channels": 16,
                "character_positions": {
                    "Opus45": [205, 128],
                    "Opus4": [170, 128],
                    "Sonnet4": [85, 128],
                    "Haiku45": [50, 128],
                },
            },
        },
        "timing": {
            "fade_time": 0.3,
            "mover_speed": 50,
        },
    }
    
    with open("dmx_config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print("✅ Created dmx_config.yaml")


async def main():
    parser = argparse.ArgumentParser(description="DMX/OSC Bridge for Melodeus")
    parser.add_argument("--config", "-c", help="Config file path", default="dmx_config.yaml")
    parser.add_argument("--create-config", action="store_true", help="Create example config")
    args = parser.parse_args()
    
    if args.create_config:
        create_example_config()
        return
    
    bridge = DMXOSCBridge(args.config if Path(args.config).exists() else None)
    
    try:
        await bridge.start()
    except KeyboardInterrupt:
        await bridge.stop()


if __name__ == "__main__":
    asyncio.run(main())
