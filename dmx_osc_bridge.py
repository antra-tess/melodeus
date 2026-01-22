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
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path

# Persistent state file (stores master volume, per-character volumes, etc.)
STATE_FILE = Path(__file__).parent / ".dmx_bridge_state.json"

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
    narrator_color: Tuple[int, ...] = (255, 255, 255, 200, 0, 0)  # White/neutral for narrator
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
            <button class="btn" style="background: #eab308; color: black;" onclick="sendCommand('stop_all_thinking')">🛑 Stop Thinking</button>
            <button class="btn" style="background: #22c55e; color: white;" onclick="sendCommand('save_colors')">💾 Save Colors</button>
        </div>
        <div style="margin-top: 20px; padding: 15px; background: var(--bg-card); border-radius: 8px; max-width: 400px;">
            <label style="display: flex; align-items: center; gap: 10px; font-size: 14px;">
                <span>🔊 Master Mannequin Volume:</span>
                <span id="master-vol-val" style="min-width: 45px; text-align: right;">75%</span>
            </label>
            <input type="range" id="master-vol-slider" min="0" max="100" value="75" style="width: 100%; margin-top: 8px;"
                   oninput="document.getElementById('master-vol-val').textContent = this.value + '%'"
                   onchange="setMasterVolume(this.value / 100)">
        </div>
    </div>
    
    <!-- House Lights Section -->
    <div class="house-lights-section" id="house-lights-section" style="margin-top: 30px; max-width: 1800px; margin-left: auto; margin-right: auto;">
        <h2 style="font-size: 1rem; margin-bottom: 15px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 2px;">🏠 House Lights</h2>
        <div id="house-lights-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px;">
            <!-- Groups generated by JS -->
        </div>
    </div>
    
    <!-- Movers Section -->
    <div class="movers-section" id="movers-section" style="margin-top: 30px; max-width: 1800px; margin-left: auto; margin-right: auto;">
        <h2 style="font-size: 1rem; margin-bottom: 15px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 2px;">🎯 Moving Heads</h2>
        <div style="margin-bottom: 15px; display: flex; gap: 10px; flex-wrap: wrap;">
            <span style="font-size: 12px; color: var(--text-muted); align-self: center;">Quick Targets:</span>
            <button onclick="moveBigMoversTo('Opus')" style="padding: 8px 15px; background: linear-gradient(135deg, #ff6b35, #f7931e); border: none; border-radius: 6px; color: white; cursor: pointer; font-weight: 600;">🎯 3Opus</button>
            <button onclick="sendCommand('movers_off')" style="padding: 8px 15px; background: var(--bg-secondary); border: none; border-radius: 6px; color: var(--text-muted); cursor: pointer;">All Off</button>
        </div>
        <div id="movers-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 15px;">
            <!-- Movers generated by JS -->
        </div>
    </div>
    
    <div class="log-panel">
        <h2>Event Log</h2>
        <div class="log-content" id="log-content"></div>
    </div>
    
    <script>
        let ws;
        let state = { characters: {}, speaking: [], house_lights: {}, movers: {} };
        let masterStageColor = '#ffffff';
        let masterStageBrightness = 100;
        let groupColors = {};
        
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
                    renderHouseLights();
                    renderMovers();
                    updateStatuses(msg.data);
                    // Update master volume slider
                    if (msg.data.master_volume !== undefined) {
                        const slider = document.getElementById('master-vol-slider');
                        const val = document.getElementById('master-vol-val');
                        if (slider && val) {
                            slider.value = Math.round(msg.data.master_volume * 100);
                            val.textContent = slider.value + '%';
                        }
                    }
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
        
        function renderHouseLights() {
            const grid = document.getElementById('house-lights-grid');
            if (!grid || !state.house_lights) return;
            
            grid.innerHTML = '';
            
            for (const [groupName, fixtures] of Object.entries(state.house_lights)) {
                const groupCard = document.createElement('div');
                groupCard.style.cssText = 'background: var(--bg-card); border-radius: 12px; padding: 15px;';
                
                const displayName = groupName.replace(/_/g, ' ').replace(/\\b\\w/g, c => c.toUpperCase());
                
                let fixtureHtml = '';
                for (const fix of fixtures) {
                    const brightness = Math.round((fix.brightness || 0) * 100);
                    const color = fix.color || [255, 200, 150, 100, 120, 0];
                    const rgb = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
                    const hexColor = rgbToHex(color[0], color[1], color[2]);
                    
                    fixtureHtml += `
                        <div style="display: flex; align-items: center; gap: 8px; margin: 8px 0; padding: 8px; background: var(--bg-secondary); border-radius: 6px; flex-wrap: wrap;">
                            <input type="color" value="${hexColor}" style="width: 24px; height: 24px; border: none; cursor: pointer;"
                                   onchange="setHouseLightColor('${fix.name}', this.value)">
                            <span style="font-size: 11px; min-width: 80px;">${fix.name.replace(/^Back/, 'B').replace(/^Front/, 'F')} <span style="color: var(--text-muted);">@${fix.dmx_address}</span></span>
                            <input type="range" min="0" max="100" value="${brightness}" style="width: 80px; height: 4px;"
                                   onchange="setHouseLight('${fix.name}', this.value/100)">
                            <button onclick="testHouseLight('${fix.name}')" style="padding: 4px 8px; font-size: 10px; background: var(--accent); border: none; border-radius: 4px; color: white; cursor: pointer;">Test</button>
                        </div>
                    `;
                }
                
                const savedGroupColor = groupColors[groupName] || '#ffffff';
                
                groupCard.innerHTML = `
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <span style="font-weight: 600; text-transform: uppercase; font-size: 12px;">${displayName}</span>
                        <div style="display: flex; gap: 5px; align-items: center;">
                            <input type="color" value="${savedGroupColor}" style="width: 24px; height: 24px; border: none; cursor: pointer;" title="Master Color"
                                   onchange="groupColors['${groupName}']=this.value; setHouseGroupColor('${groupName}', this.value)">
                            <button onclick="setHouseGroup('${groupName}', 1.0)" style="padding: 4px 10px; font-size: 10px; background: var(--success); border: none; border-radius: 4px; color: white; cursor: pointer;">On</button>
                            <button onclick="setHouseGroup('${groupName}', 0)" style="padding: 4px 10px; font-size: 10px; background: var(--bg-secondary); border: none; border-radius: 4px; color: var(--text-muted); cursor: pointer;">Off</button>
                        </div>
                    </div>
                    ${fixtureHtml}
                `;
                
                grid.appendChild(groupCard);
            }
            
            // Add master all house lights control - only create if not exists
            let masterCard = document.getElementById('master-stage-card');
            if (!masterCard) {
                masterCard = document.createElement('div');
                masterCard.id = 'master-stage-card';
                masterCard.style.cssText = 'background: var(--bg-card); border-radius: 12px; padding: 15px; border: 2px solid var(--accent);';
                masterCard.innerHTML = `
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <span style="font-weight: 600; text-transform: uppercase; font-size: 12px;">🎨 MASTER ALL STAGE</span>
                    </div>
                    <div style="display: flex; gap: 10px; align-items: center; flex-wrap: wrap;">
                        <input type="color" id="master-stage-color" value="${masterStageColor}" style="width: 40px; height: 40px; border: none; cursor: pointer;"
                               onchange="masterStageColor = this.value">
                        <input type="range" id="master-stage-brightness" min="0" max="100" value="${masterStageBrightness}" style="flex: 1; min-width: 100px;"
                               oninput="masterStageBrightness = parseInt(this.value); document.getElementById('master-stage-val').textContent = this.value + '%'">
                        <span id="master-stage-val" style="min-width: 40px;">${masterStageBrightness}%</span>
                        <button onclick="setAllStageColor()" style="padding: 8px 15px; background: var(--accent); border: none; border-radius: 4px; color: white; cursor: pointer;">Apply</button>
                    </div>
                `;
                grid.appendChild(masterCard);
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
                const isNarrator = (state.narrator || []).includes(name);
                const color = char.color || [255, 200, 150, 200];
                const narratorColor = char.narrator_color || [255, 255, 255, 200, 0, 0];
                const displayColor = isNarrator ? narratorColor : color;
                const rgbColor = `rgb(${displayColor[0]}, ${displayColor[1]}, ${displayColor[2]})`;
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
                        ${isNarrator ? '<span class="speaking-badge" style="background: #888; opacity: 1;">NARRATOR</span>' : ''}
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
                                   onchange="setColor('${name}', this.value)" title="Character voice color">
                            <input type="color" value="${rgbToHex(narratorColor[0], narratorColor[1], narratorColor[2])}"
                                   onchange="setNarratorColor('${name}', this.value)" title="Narrator voice color" style="margin-left: 5px;">
                            <span class="value-display" style="font-size: 9px;">Voice/Narr</span>
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
            // Returns [R, G, B, W, Amber, UV] for ADJ 7P fixtures
            const result = /^#?([a-f\\d]{2})([a-f\\d]{2})([a-f\\d]{2})$/i.exec(hex);
            return result ? [
                parseInt(result[1], 16),
                parseInt(result[2], 16),
                parseInt(result[3], 16),
                0,   // White
                0,   // Amber
                0    // UV
            ] : [255, 255, 255, 0, 0, 0];
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
        
        function setNarratorColor(name, hex) {
            sendCommand('set_narrator_color', { character: name, color: hexToRgb(hex) });
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
        
        // House light controls
        function setHouseLight(name, brightness) {
            sendCommand('set_house_light', { fixture: name, brightness: brightness });
        }
        
        function setHouseGroup(groupName, brightness) {
            sendCommand('set_house_group', { group: groupName, brightness: brightness });
        }
        
        function testHouseLight(name) {
            sendCommand('test_house_light', { fixture: name });
        }
        
        function testHouseGroup(groupName) {
            sendCommand('test_house_group', { group: groupName });
        }
        
        function setHouseLightColor(name, hex) {
            const rgb = hexToRgb(hex);
            sendCommand('set_house_light_color', { fixture: name, color: rgb });
        }
        
        function setHouseGroupColor(groupName, hex) {
            const rgb = hexToRgb(hex);
            sendCommand('set_house_group_color', { group: groupName, color: rgb });
        }
        
        function setAllStageColor() {
            const hex = document.getElementById('master-stage-color').value;
            const brightness = document.getElementById('master-stage-brightness').value / 100;
            const rgb = hexToRgb(hex);
            sendCommand('set_all_stage_color', { color: rgb, brightness: brightness });
        }
        
        // Mover controls
        // Color wheel values for movers
        const moverColors = [
            { name: 'White', value: 0, css: '#ffffff' },
            { name: 'Red', value: 10, css: '#ff0000' },
            { name: 'Yellow', value: 20, css: '#ffff00' },
            { name: 'Green', value: 28, css: '#00ff00' },
            { name: 'Blue', value: 36, css: '#0000ff' },
            { name: 'Cyan', value: 52, css: '#00ffff' },
            { name: 'Magenta', value: 60, css: '#ff00ff' },
        ];
        
        function renderMovers() {
            const grid = document.getElementById('movers-grid');
            if (!grid || !state.movers) return;
            
            grid.innerHTML = '';
            
            for (const [name, mover] of Object.entries(state.movers)) {
                const card = document.createElement('div');
                card.style.cssText = 'background: var(--bg-card); border-radius: 12px; padding: 15px;';
                
                const pan = mover.pan || 128;
                const tilt = mover.tilt || 128;
                const dimmer = Math.round((mover.dimmer || 0) * 100);
                const zoom = mover.zoom || 128;
                const colorVal = mover.color || 0;
                const hasZoom = mover.has_zoom || false;
                
                let zoomHtml = '';
                if (hasZoom) {
                    zoomHtml = `
                        <div style="margin: 8px 0;">
                            <label style="font-size: 10px; color: var(--text-muted);">Zoom: <span id="zoom-val-${name}">${zoom}</span></label>
                            <input type="range" min="0" max="255" value="${zoom}" style="width: 100%;"
                                   oninput="document.getElementById('zoom-val-${name}').textContent=this.value"
                                   onchange="setMover('${name}', {zoom: parseInt(this.value)})">
                        </div>
                    `;
                }
                
                // Color buttons
                let colorBtns = moverColors.map(c => 
                    `<button onclick="setMover('${name}', {color: ${c.value}})" 
                             style="width: 24px; height: 24px; background: ${c.css}; border: 2px solid ${colorVal >= c.value && colorVal < c.value + 8 ? 'white' : '#333'}; border-radius: 4px; cursor: pointer;" 
                             title="${c.name}"></button>`
                ).join('');
                
                card.innerHTML = `
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <span style="font-weight: 600;">${name.replace('FrontSpot', 'Front').replace('BigMover', 'Big')}</span>
                        <span style="font-size: 10px; color: var(--text-muted);">@${mover.dmx_address}</span>
                    </div>
                    <div style="display: flex; gap: 4px; margin-bottom: 10px; flex-wrap: wrap;">
                        ${colorBtns}
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                        <div>
                            <label style="font-size: 10px; color: var(--text-muted);">Pan: <span id="pan-val-${name}">${pan}</span></label>
                            <input type="range" min="0" max="255" value="${pan}" style="width: 100%;"
                                   oninput="document.getElementById('pan-val-${name}').textContent=this.value"
                                   onchange="setMover('${name}', {pan: parseInt(this.value)})">
                        </div>
                        <div>
                            <label style="font-size: 10px; color: var(--text-muted);">Tilt: <span id="tilt-val-${name}">${tilt}</span></label>
                            <input type="range" min="0" max="255" value="${tilt}" style="width: 100%;"
                                   oninput="document.getElementById('tilt-val-${name}').textContent=this.value"
                                   onchange="setMover('${name}', {tilt: parseInt(this.value)})">
                        </div>
                    </div>
                    <div style="margin: 8px 0;">
                        <label style="font-size: 10px; color: var(--text-muted);">Dimmer: <span id="dim-val-${name}">${dimmer}%</span></label>
                        <input type="range" min="0" max="100" value="${dimmer}" style="width: 100%;"
                               oninput="document.getElementById('dim-val-${name}').textContent=this.value+'%'"
                               onchange="setMover('${name}', {dimmer: parseInt(this.value)/100})">
                    </div>
                    ${zoomHtml}
                    <div style="display: flex; gap: 5px; margin-top: 10px;">
                        <button onclick="setMover('${name}', {dimmer: 1})" style="flex:1; padding: 6px; font-size: 11px; background: var(--success); border: none; border-radius: 4px; color: white; cursor: pointer;">On</button>
                        <button onclick="setMover('${name}', {dimmer: 0})" style="flex:1; padding: 6px; font-size: 11px; background: var(--bg-secondary); border: none; border-radius: 4px; color: var(--text-muted); cursor: pointer;">Off</button>
                        <button onclick="setMover('${name}', {pan: 128, tilt: 128})" style="flex:1; padding: 6px; font-size: 11px; background: var(--accent); border: none; border-radius: 4px; color: white; cursor: pointer;">Center</button>
                    </div>
                `;
                
                grid.appendChild(card);
            }
        }
        
        function setMover(name, values) {
            sendCommand('set_mover', { name: name, ...values });
        }
        
        function setMasterVolume(value) {
            sendCommand('set_master_volume', { volume: value });
            log(`Master volume set to ${Math.round(value * 100)}%`, 'event');
        }
        
        function moveBigMoversTo(character) {
            sendCommand('movers_to_character', { character });
            log(`Big movers to ${character}`, 'event');
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
        
        # House lights (not tied to characters)
        self.house_lights: Dict[str, Dict] = {}  # group_name -> {name -> fixture_info}
        self.house_light_channels: Dict[str, any] = {}  # fixture_name -> DMX channel
        self.house_light_state: Dict[str, Dict] = {}  # fixture_name -> current state
        
        # Movers (moving heads)
        self.movers: Dict[str, Dict] = {}  # name -> fixture_info
        self.mover_channels: Dict[str, any] = {}  # name -> DMX channel
        self.mover_state: Dict[str, Dict] = {}  # name -> {pan, tilt, dimmer, zoom, ...}
        self.mover_targets: Dict[str, Dict[str, Dict]] = {}  # character -> {mover_name -> {pan, tilt, zoom}}
        
        # State tracking
        self.speaking_characters: Set[str] = set()
        self.thinking_characters: Set[str] = set()
        self.narrator_active: Set[str] = set()  # Characters currently speaking as narrator
        self.pulse_tasks: Dict[str, asyncio.Task] = {}  # Character -> pulse task
        self.thinking_timeout_tasks: Dict[str, asyncio.Task] = {}  # Character -> timeout task
        self.current_brightness: Dict[str, float] = {}
        self.current_x32_fader: Dict[str, float] = {}
        self.current_x32_muted: Dict[str, bool] = {}
        self.last_update: Dict[str, float] = {}
        self.master_volume: float = 0.75  # Master volume multiplier for all mannequin voices
        self.character_volumes: Dict[str, float] = {}  # Per-character volume adjustments (for uneven TTS loudness)
        
        # Load persistent state (master volume, per-character volumes)
        self._load_persistent_state()
        
        # Body possession tracking - allows characters to inhabit other mannequins
        # Maps character name -> body name (None = use own body)
        self.body_possession: Dict[str, Optional[str]] = {}
        
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
    
    def _load_persistent_state(self):
        """Load persistent state from JSON file (master volume, per-character volumes, colors)."""
        self._saved_character_colors: Dict[str, List[int]] = {}
        self._saved_narrator_colors: Dict[str, List[int]] = {}
        self._saved_house_light_colors: Dict[str, List[int]] = {}
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE) as f:
                    state = json.load(f)
                self.master_volume = state.get("master_volume", 0.75)
                self.character_volumes = state.get("character_volumes", {})
                self._saved_character_colors = state.get("character_colors", {})
                self._saved_narrator_colors = state.get("narrator_colors", {})
                self._saved_house_light_colors = state.get("house_light_colors", {})
                print(f"📂 Loaded persistent state: master={self.master_volume:.0%}")
                if self.character_volumes:
                    print(f"   Character volumes: {self.character_volumes}")
                if self._saved_character_colors:
                    print(f"   Saved colors for: {list(self._saved_character_colors.keys())}")
            except Exception as e:
                print(f"⚠️  Failed to load persistent state: {e}")
    
    def _apply_saved_colors(self):
        """Apply saved colors to fixtures after they're loaded from config."""
        # Apply character colors
        for char_name, color in self._saved_character_colors.items():
            if char_name in self.character_fixtures:
                fixtures = self.character_fixtures[char_name]
                if fixtures.par_can:
                    fixtures.par_can.color = tuple(color)
                    print(f"   🎨 Restored color for {char_name}")
        
        # Apply narrator colors
        for char_name, color in self._saved_narrator_colors.items():
            if char_name in self.character_fixtures:
                fixtures = self.character_fixtures[char_name]
                if fixtures.par_can:
                    fixtures.par_can.narrator_color = tuple(color)
                    print(f"   📖 Restored narrator color for {char_name}")
        
        # Apply house light colors
        for fixture_name, color in self._saved_house_light_colors.items():
            if fixture_name in self.house_light_state:
                self.house_light_state[fixture_name]["color"] = color
                print(f"   🏠 Restored color for {fixture_name}")
    
    def _save_persistent_state(self):
        """Save persistent state to JSON file."""
        # Collect current character colors
        character_colors = {}
        narrator_colors = {}
        for char_name, fixtures in self.character_fixtures.items():
            if fixtures.par_can:
                character_colors[char_name] = list(fixtures.par_can.color)
                narrator_colors[char_name] = list(fixtures.par_can.narrator_color)
        
        # Collect house light colors
        house_light_colors = {}
        for fixture_name, state in self.house_light_state.items():
            if "color" in state:
                house_light_colors[fixture_name] = state["color"]
        
        state = {
            "master_volume": self.master_volume,
            "character_volumes": self.character_volumes,
            "character_colors": character_colors,
            "narrator_colors": narrator_colors,
            "house_light_colors": house_light_colors,
        }
        try:
            with open(STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)
            print(f"💾 Saved persistent state (including colors)")
        except Exception as e:
            print(f"⚠️  Failed to save persistent state: {e}")
    
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
                # Default narrator color: white/neutral if not specified
                narrator_color = tuple(pc.get("narrator_color", [255, 255, 255, 200, 0, 0]))
                channel_map = pc.get("channel_map", {
                    "red": 0, "green": 1, "blue": 2, "lime": 3, "amber": 4, "uv": 5, "shutter": 6, "dimmer": 7
                })
                fixtures.par_can = ParCanFixture(
                    dmx_address=pc["dmx_address"],
                    channels=pc.get("channels", 8),
                    channel_map=channel_map,
                    color=color,
                    narrator_color=narrator_color,
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
        
        # Setup house lights (not tied to characters)
        for group_name, fixtures in cfg.get("house_lights", {}).items():
            self.house_lights[group_name] = {}
            for fix_info in fixtures:
                name = fix_info.get("name", f"{group_name}_{fix_info['dmx_address']}")
                fix_type = fix_info.get("type", "unknown")
                
                # Check if this is a mover (moving head)
                is_mover = "Mover" in fix_type or "375ZX" in fix_type or "Intimidator" in fix_type
                
                if is_mover:
                    # Add to movers dict
                    self.movers[name] = {
                        "dmx_address": fix_info["dmx_address"],
                        "type": fix_type,
                        "channels": fix_info.get("channels", 15),
                        "home_pan": fix_info.get("home_pan", 128),
                        "home_tilt": fix_info.get("home_tilt", 128),
                    }
                    # Initialize mover state
                    self.mover_state[name] = {
                        "pan": fix_info.get("home_pan", 128),
                        "tilt": fix_info.get("home_tilt", 128),
                        "dimmer": 0,
                        "zoom": 128,
                        "color": 0,
                    }
                else:
                    # Regular house light
                    self.house_lights[group_name][name] = {
                        "dmx_address": fix_info["dmx_address"],
                        "type": fix_type,
                        "channels": fix_info.get("channels", 8),
                    }
                    # Initialize state
                    self.house_light_state[name] = {
                        "brightness": 0.0,
                        "color": [0, 0, 0, 0, 0, 0],  # R, G, B, Lime, Amber, UV
                    }
            
            # Remove empty groups
            if not self.house_lights[group_name]:
                del self.house_lights[group_name]
        
        # Load mover targets (preset positions for characters)
        self.mover_targets = cfg.get("mover_targets", {})
        if self.mover_targets:
            print(f"📋 Loaded mover targets for {len(self.mover_targets)} characters")
        
        print(f"📋 Configured {len(self.character_fixtures)} characters")
        for name, fix in self.character_fixtures.items():
            par = f"par@{fix.par_can.dmx_address}" if fix.par_can else "no par"
            movers = len(fix.moving_heads)
            x32 = f"X32 ch{self.character_x32[name].channel}" if name in self.character_x32 else "no X32"
            print(f"   {name}: {par}, {movers} movers, {x32}")
        
        # Apply any saved colors from persistent state
        self._apply_saved_colors()
    
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
        print(f"📥 Dashboard command: {cmd} - {data}")
        
        if cmd == "activate":
            await self._activate_character(data["character"])
        elif cmd == "deactivate":
            await self._deactivate_character(data["character"])
        elif cmd == "set_brightness":
            await self._set_character_brightness(data["character"], data["brightness"])
        elif cmd == "set_color":
            await self._set_character_color(data["character"], data["color"])
        elif cmd == "set_narrator_color":
            await self._set_narrator_color(data["character"], data["color"])
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
        # House light commands
        elif cmd == "set_house_light":
            await self.set_house_light(data["fixture"], data["brightness"])
        elif cmd == "set_house_group":
            await self.set_house_group(data["group"], data["brightness"])
        elif cmd == "test_house_light":
            asyncio.create_task(self.test_house_light(data["fixture"]))
        elif cmd == "test_house_group":
            asyncio.create_task(self.test_house_group(data["group"]))
        elif cmd == "set_house_light_color":
            await self.set_house_light(data["fixture"], 1.0, data["color"])
        elif cmd == "set_house_group_color":
            await self.set_house_group(data["group"], 1.0, data["color"])
        elif cmd == "set_all_stage_color":
            await self.set_all_house_lights(data["brightness"], data["color"])
        # Mover commands
        elif cmd == "set_mover":
            await self.set_mover(data["name"], data.get("pan"), data.get("tilt"), 
                                 data.get("dimmer"), data.get("zoom"), data.get("color"))
        elif cmd == "movers_to_character":
            await self._move_big_movers_to_character(data["character"])
        elif cmd == "movers_off":
            await self._movers_off()
        # Master volume
        elif cmd == "set_master_volume":
            self.master_volume = data["volume"]
            self._save_persistent_state()
            print(f"🔊 Master volume set to {self.master_volume * 100:.0f}%")
        # Stop all thinking
        elif cmd == "stop_all_thinking":
            await self._stop_all_thinking()
        # Save colors
        elif cmd == "save_colors":
            self._save_persistent_state()
            await self._broadcast_event("log", message="Colors saved!", level="event")
        
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
                char_state["narrator_color"] = list(fixtures.par_can.narrator_color)
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
            "narrator": list(self.narrator_active),  # Characters speaking as narrator
            "body_possession": dict(self.body_possession),  # Who is inhabiting which body
            "artnet_connected": self.artnet_node is not None,
            "x32_connected": self.x32_client is not None,
            "osc_listening": self.osc_listening,
            "house_lights": self.get_house_light_info(),
            "movers": self.get_mover_info(),
            "master_volume": self.master_volume,
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
    
    async def _set_narrator_color(self, character: str, color: List[int]):
        """Manually set narrator color for a character."""
        if character not in self.character_fixtures:
            return
        
        fixtures = self.character_fixtures[character]
        if fixtures.par_can:
            fixtures.par_can.narrator_color = tuple(color)
            # If currently in narrator mode, update the display
            if character in self.narrator_active:
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
        
        # Create channels for house lights
        for group_name, fixtures in self.house_lights.items():
            for fix_name, fix_info in fixtures.items():
                try:
                    channel = self.artnet_universe.add_channel(
                        start=fix_info["dmx_address"],
                        width=fix_info["channels"]
                    )
                    self.house_light_channels[fix_name] = channel
                except Exception as e:
                    print(f"   ⚠️  Could not add {fix_name}@{fix_info['dmx_address']}: {e}")
        
        if self.house_light_channels:
            print(f"   🏠 House lights: {len(self.house_light_channels)} fixtures")
        
        # Create channels for movers
        for name, info in self.movers.items():
            try:
                channel = self.artnet_universe.add_channel(
                    start=info["dmx_address"],
                    width=info["channels"]
                )
                self.mover_channels[name] = channel
                print(f"   🎯 {name}: DMX {info['dmx_address']}-{info['dmx_address'] + info['channels'] - 1}")
            except Exception as e:
                print(f"   ⚠️  Could not add mover {name}@{info['dmx_address']}: {e}")
        
        if self.mover_channels:
            print(f"   🎯 Movers: {len(self.mover_channels)} fixtures")
        
        # Enter async context and start refresh
        await self.artnet_node.__aenter__()
        self.artnet_node.start_refresh()
        print(f"   ✅ Art-Net started, {len(self.dmx_channels)} character fixtures, {len(self.house_light_channels)} house fixtures")
    
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
        dispatcher.map("/character/narrator/start", self._on_narrator_start)
        dispatcher.map("/character/narrator/stop", self._on_narrator_stop)
        dispatcher.map("/character/amplitude", self._on_amplitude)
        dispatcher.map("/character/move_body", self._on_move_body)
        dispatcher.map("/avatar/color", self._on_avatar_color)
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
        """Start pulsing animation for thinking character.
        
        If the character is possessing another body, pulse that body's fixture.
        """
        # Cancel any existing pulse task for this character
        if character_name in self.pulse_tasks:
            self.pulse_tasks[character_name].cancel()
            try:
                await self.pulse_tasks[character_name]
            except asyncio.CancelledError:
                pass
        
        # Cancel any existing timeout task
        if character_name in self.thinking_timeout_tasks:
            self.thinking_timeout_tasks[character_name].cancel()
            try:
                await self.thinking_timeout_tasks[character_name]
            except asyncio.CancelledError:
                pass
        
        # Get the effective body (either own or possessed)
        effective_body = self._get_effective_body(character_name)
        
        if effective_body != character_name:
            print(f"   🔄 {character_name} thinking through {effective_body}'s body")
        
        # Start new pulse task for the effective body
        task = asyncio.create_task(self._pulse_character(character_name))
        self.pulse_tasks[character_name] = task
        
        # Start timeout task (60 seconds max thinking time)
        timeout_task = asyncio.create_task(self._thinking_timeout(character_name, 60.0))
        self.thinking_timeout_tasks[character_name] = timeout_task
        
        # Also raise the X32 bus fader so thinking sound comes from mannequin speaker
        # Use the effective body's X32 channel
        if effective_body in self.character_x32 and self.x32_client:
            x32 = self.character_x32[effective_body]
            self._x32_set_channel(x32, effective_body, active=True)
        
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
        
        # Cancel timeout task if exists
        if character_name in self.thinking_timeout_tasks:
            self.thinking_timeout_tasks[character_name].cancel()
            try:
                await self.thinking_timeout_tasks[character_name]
            except asyncio.CancelledError:
                pass
            del self.thinking_timeout_tasks[character_name]
        
        await self._broadcast_event("thinking_stop", character=character_name)
        await self._broadcast_state()
    
    async def _stop_all_thinking(self):
        """Stop all thinking states (cancel all pulsing animations)."""
        print(f"🛑 Stopping all thinking states")
        
        # Get list of currently thinking characters
        thinking_chars = list(self.thinking_characters)
        
        # Clear the set first
        self.thinking_characters.clear()
        
        # Cancel all pulse tasks
        for character_name in thinking_chars:
            if character_name in self.pulse_tasks:
                self.pulse_tasks[character_name].cancel()
                try:
                    await self.pulse_tasks[character_name]
                except asyncio.CancelledError:
                    pass
            
            # Cancel timeout tasks
            if character_name in self.thinking_timeout_tasks:
                self.thinking_timeout_tasks[character_name].cancel()
                try:
                    await self.thinking_timeout_tasks[character_name]
                except asyncio.CancelledError:
                    pass
            
            await self._broadcast_event("thinking_stop", character=character_name)
        
        # Clear task dicts
        self.pulse_tasks.clear()
        self.thinking_timeout_tasks.clear()
        
        print(f"   ✓ Stopped thinking for: {thinking_chars}")
        await self._broadcast_state()
    
    async def _thinking_timeout(self, character_name: str, timeout_seconds: float = 60.0):
        """Auto-stop thinking after timeout."""
        try:
            await asyncio.sleep(timeout_seconds)
            if character_name in self.thinking_characters:
                print(f"⏰ Thinking timeout for {character_name} ({timeout_seconds}s)")
                self.thinking_characters.discard(character_name)
                await self._on_thinking_stop_async(character_name)
        except asyncio.CancelledError:
            pass  # Normal cancellation when thinking stops normally
    
    def _on_narrator_start(self, address, character_name):
        """Handle character starting to speak as narrator (different voice/color)."""
        print(f"📖 {character_name} narrator mode ON")
        self.narrator_active.add(character_name)
        asyncio.create_task(self._apply_narrator_color(character_name))
    
    def _on_narrator_stop(self, address, character_name):
        """Handle character stopping narrator mode (back to normal voice/color)."""
        print(f"📖 {character_name} narrator mode OFF")
        self.narrator_active.discard(character_name)
        asyncio.create_task(self._apply_character_color(character_name))
    
    async def _apply_narrator_color(self, character_name: str):
        """Apply narrator color to character's fixture."""
        effective_body = self._get_effective_body(character_name)
        if effective_body not in self.character_fixtures:
            return
        
        fixtures = self.character_fixtures[effective_body]
        if fixtures.par_can and effective_body in self.dmx_channels:
            # Use narrator color
            color = fixtures.par_can.narrator_color
            brightness = self.current_brightness.get(effective_body, fixtures.par_can.idle_brightness)
            await self._set_par_color_and_brightness(effective_body, color, brightness)
            print(f"   🎨 Applied narrator color to {effective_body}")
        
        await self._broadcast_state()
    
    async def _apply_character_color(self, character_name: str):
        """Apply normal character color to fixture."""
        effective_body = self._get_effective_body(character_name)
        if effective_body not in self.character_fixtures:
            return
        
        fixtures = self.character_fixtures[effective_body]
        if fixtures.par_can and effective_body in self.dmx_channels:
            # Use normal character color
            color = fixtures.par_can.color
            brightness = self.current_brightness.get(effective_body, fixtures.par_can.idle_brightness)
            await self._set_par_color_and_brightness(effective_body, color, brightness)
            print(f"   🎨 Applied normal color to {effective_body}")
        
        await self._broadcast_state()
    
    def _on_amplitude(self, address, character_name, amplitude):
        """Handle real-time audio amplitude for reactive lighting.
        
        Args:
            character_name: The speaking character
            amplitude: Normalized amplitude 0.0-1.0
        """
        # Only modulate if character is currently speaking
        if character_name not in self.speaking_characters:
            return
        
        # Clamp amplitude
        amplitude = max(0.0, min(1.0, float(amplitude)))
        
        # Schedule async update (don't await to avoid blocking OSC)
        asyncio.create_task(self._set_amplitude_brightness(character_name, amplitude))
    
    async def _set_amplitude_brightness(self, character_name: str, amplitude: float):
        """Set brightness based on audio amplitude."""
        effective_body = self._get_effective_body(character_name)
        if effective_body not in self.character_fixtures:
            return
        
        fixtures = self.character_fixtures[effective_body]
        if not fixtures.par_can:
            return
        
        # Map amplitude to brightness range
        # Minimum brightness when speaking (so light doesn't go dark during pauses)
        min_brightness = 0.3
        max_brightness = fixtures.par_can.active_brightness
        
        # Apply curve for more dramatic effect (amplitude^0.7 makes quieter sounds more visible)
        curved_amplitude = amplitude ** 0.7
        brightness = min_brightness + (curved_amplitude * (max_brightness - min_brightness))
        
        # Update brightness (this will use narrator/character color automatically)
        self.current_brightness[effective_body] = brightness
        await self._set_par_brightness_for_character(effective_body, brightness)
    
    async def _set_par_color_and_brightness(self, character: str, color: Tuple[int, ...], brightness: float):
        """Set both color and brightness for a character's par can."""
        if character not in self.character_fixtures:
            return
        
        fixtures = self.character_fixtures[character]
        if not fixtures.par_can or character not in self.dmx_channels:
            return
        
        par = fixtures.par_can
        channel = self.dmx_channels[character]
        
        # Build DMX values with color and brightness
        dmx_values = [0] * par.channels
        
        # Set color channels with brightness scaling
        for color_name, channel_idx in par.channel_map.items():
            if color_name in ["red", "green", "blue", "lime", "amber", "uv"]:
                color_map = {"red": 0, "green": 1, "blue": 2, "lime": 3, "amber": 4, "uv": 5}
                if color_name in color_map and color_map[color_name] < len(color):
                    dmx_values[channel_idx] = int(color[color_map[color_name]] * brightness)
            elif color_name == "shutter":
                dmx_values[channel_idx] = par.shutter_on
            elif color_name == "dimmer":
                dmx_values[channel_idx] = int(255 * brightness)
        
        channel.set_values(dmx_values)
    
    def _on_move_body(self, address, character_name, target_body):
        """Handle character moving to another body/mannequin.
        
        This allows a character (like 3Opus) to inhabit another mannequin's body.
        When the character speaks, their voice will come from the target body's
        speaker and light up that body's fixture instead of their own.
        """
        print(f"🔄 {character_name} moving to body: {target_body}")
        
        # If moving to own body or empty/None, clear possession
        if target_body in [character_name, "", "None", None, "null"]:
            if character_name in self.body_possession:
                del self.body_possession[character_name]
                print(f"   ↩️ {character_name} returned to own body")
        else:
            # Check if target body exists
            if target_body in self.character_fixtures:
                self.body_possession[character_name] = target_body
                print(f"   ✨ {character_name} now inhabits {target_body}'s body")
            else:
                print(f"   ❌ Unknown body: {target_body}")
                print(f"   Available bodies: {list(self.character_fixtures.keys())}")
                return
        
        # Broadcast update
        asyncio.create_task(self._broadcast_state())
    
    def _on_avatar_color(self, address, character_name, *rgb_args):
        """Handle avatar color change request.
        
        Args:
            character_name: The character whose color to change
            rgb_args: RGB values - can be [r, g, b] or individual args
        """
        print(f"🎨 Color change request: {character_name} -> {rgb_args}")
        
        # Parse RGB values
        try:
            if len(rgb_args) == 1 and isinstance(rgb_args[0], str):
                # It's a JSON string like "[255, 0, 0]"
                import json
                rgb = json.loads(rgb_args[0])
            elif len(rgb_args) == 3:
                # Individual r, g, b args
                rgb = list(rgb_args)
            elif len(rgb_args) == 1 and isinstance(rgb_args[0], (list, tuple)):
                rgb = list(rgb_args[0])
            else:
                print(f"   ❌ Invalid RGB format: {rgb_args}")
                return
            
            if len(rgb) != 3:
                print(f"   ❌ Need 3 RGB values, got {len(rgb)}")
                return
            
            r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
            print(f"   ✨ Setting {character_name} color to RGB({r}, {g}, {b})")
            
            asyncio.create_task(self._set_character_color_async(character_name, r, g, b))
            
        except Exception as e:
            print(f"   ❌ Error parsing color: {e}")
    
    async def _set_character_color_async(self, character_name: str, r: int, g: int, b: int):
        """Set a character's par can color."""
        # Find the character's fixtures
        effective_body = self._get_effective_body(character_name)
        
        if effective_body not in self.character_fixtures:
            print(f"   ❌ Character {character_name} (body: {effective_body}) not found")
            return
        
        fixtures = self.character_fixtures[effective_body]
        if not fixtures.par_can:
            print(f"   ❌ No par can configured for {effective_body}")
            return
        
        if effective_body not in self.dmx_channels:
            print(f"   ❌ No DMX channel for {effective_body}")
            return
        
        par = fixtures.par_can
        channel = self.dmx_channels[effective_body]
        
        # Update the stored color (R, G, B, Lime, Amber, UV)
        # Keep lime, amber, UV as they were, just update RGB
        old_color = par.color
        par.color = (r, g, b, old_color[3] if len(old_color) > 3 else 0, 
                     old_color[4] if len(old_color) > 4 else 0,
                     old_color[5] if len(old_color) > 5 else 0)
        
        # Get current brightness (or default to active)
        is_speaking = effective_body in self.speaking_characters
        brightness = par.active_brightness if is_speaking else par.idle_brightness
        
        # Build DMX values
        dmx_values = [
            int(r * brightness),      # R
            int(g * brightness),      # G
            int(b * brightness),      # B
            int(par.color[3] * brightness),  # Lime
            int(par.color[4] * brightness),  # Amber
            int(par.color[5]),               # UV (no brightness scaling)
            255,                              # Strobe (255 = solid on)
            255                               # Dimmer (full)
        ]
        
        # Send to DMX
        channel.set_values(dmx_values)
        print(f"   ✅ Set {effective_body} to RGB({r},{g},{b}) @ {brightness*100:.0f}%")
        
        # Broadcast state update
        await self._broadcast_state()
    
    def _get_effective_body(self, character_name: str) -> str:
        """Get the body a character is currently using (their own or a possessed one)."""
        return self.body_possession.get(character_name, character_name)
    
    async def _pulse_character(self, character_name: str):
        """Pulse the par can brightness for a thinking character.
        
        If the character is possessing another body, pulse that body's fixture.
        """
        # Get the effective body (either own or possessed)
        effective_body = self._get_effective_body(character_name)
        
        if effective_body not in self.character_fixtures:
            return
        
        fixtures = self.character_fixtures[effective_body]
        if not fixtures.par_can or effective_body not in self.dmx_channels:
            return
        
        par = fixtures.par_can
        min_brightness = par.idle_brightness
        max_brightness = 0.6  # Pulse up to 60%
        pulse_period = 0.6  # Seconds for one full cycle (faster to match thinking sound)
        
        print(f"   ✨ Starting pulse for {character_name} (body: {effective_body})")
        
        try:
            while character_name in self.thinking_characters:
                # Fade up
                await self._set_par_brightness_for_character(effective_body, max_brightness)
                await asyncio.sleep(pulse_period / 2)
                
                if character_name not in self.thinking_characters:
                    break
                
                # Fade down
                await self._set_par_brightness_for_character(effective_body, min_brightness)
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
        """Deactivate fixtures when character stops speaking.
        
        If the character was possessing another body, deactivate that body's fixtures.
        """
        # Get the effective body (either own or possessed)
        effective_body = self._get_effective_body(character_name)
        
        if effective_body not in self.character_fixtures:
            return
        
        fixtures = self.character_fixtures[effective_body]
        
        # Dim par can of the effective body
        if fixtures.par_can:
            brightness = fixtures.par_can.idle_brightness
            self.current_brightness[effective_body] = brightness
            if effective_body in self.dmx_channels:
                await self._set_par_brightness_for_character(effective_body, brightness)
        
        # X32: mute and lower fader for the effective body's channel
        if effective_body in self.character_x32 and self.x32_client:
            x32 = self.character_x32[effective_body]
            self._x32_set_channel(x32, effective_body, active=False)
    
    async def _set_par_brightness_for_character(self, character_name: str, brightness: float):
        """Set par can brightness for a character."""
        if character_name not in self.dmx_channels:
            return
        
        fixtures = self.character_fixtures.get(character_name)
        if not fixtures or not fixtures.par_can:
            return
        
        par = fixtures.par_can
        channel = self.dmx_channels[character_name]
        
        # Check if any character using this body is in narrator mode
        is_narrator = character_name in self.narrator_active
        # Also check if any character possessing this body is in narrator mode
        for char, body in self.body_possession.items():
            if body == character_name and char in self.narrator_active:
                is_narrator = True
                break
        
        # Get color values (supports RGBLAU - Red, Green, Blue, Lime, Amber, UV)
        color = par.narrator_color if is_narrator else par.color
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
        
        # Set fader - apply master volume multiplier when active
        base_fader = x32.active_fader if active else x32.idle_fader
        fader = base_fader * self.master_volume if active else base_fader
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
    
    # ========== House Light Controls ==========
    
    async def set_house_light(self, fixture_name: str, brightness: float, 
                               color: Optional[List[int]] = None, fade_ms: int = 50):
        """Set a house light to a specific brightness and color.
        
        Args:
            fixture_name: Name of the fixture
            brightness: 0.0-1.0 brightness level
            color: Optional [R, G, B, W, Amber, UV] values (0-255 each)
            fade_ms: Fade time in milliseconds
        """
        if fixture_name not in self.house_light_channels:
            print(f"⚠️  Unknown house light: {fixture_name}")
            return
        
        channel = self.house_light_channels[fixture_name]
        
        # Get fixture info to determine channel count
        fix_info = None
        for group in self.house_lights.values():
            if fixture_name in group:
                fix_info = group[fixture_name]
                break
        
        num_channels = fix_info["channels"] if fix_info else 8
        fix_type = fix_info.get("type", "") if fix_info else ""
        
        # Handle moving heads differently
        if "Mover" in fix_type or num_channels in [15, 16]:
            dimmer = int(255 * brightness)
            # Get home position from config or use defaults
            home_pan = fix_info.get("home_pan", 128) if fix_info else 128
            home_tilt = fix_info.get("home_tilt", 128) if fix_info else 128
            
            if num_channels == 15:
                # 15-channel small mover (tested):
                # Ch1:Pan, Ch2:PanFine, Ch3:Tilt, Ch4:TiltFine, Ch5:Color, Ch6:Gobo, Ch7:GoboRot
                # Ch8:DIMMER, Ch9:SHUTTER(251-255=open), Ch10:Focus, Ch11:Prism, Ch12-15:Programs
                dmx_values = [
                    home_pan, 0,    # Pan, Pan Fine
                    home_tilt, 0,   # Tilt, Tilt Fine
                    0,              # Color = Open
                    0,              # Gobo = Open
                    0,              # Gobo Rotation = None
                    dimmer,         # DIMMER
                    255,            # SHUTTER = Open
                    128,            # Focus
                    0, 0, 0, 0, 0   # Prism, Programs off
                ]
            else:
                # 16-channel moving head
                dmx_values = [home_pan, 0, home_tilt, 0, 0, dimmer, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        else:
            # Default to warm white if no color specified
            # ADJ 7P: R, G, B, White, Amber, UV
            if color is None:
                color = [200, 150, 100, 255, 180, 0]  # R, G, B, W, Amber, UV - warm white
            
            # Apply brightness to color channels
            scaled_color = [int(c * brightness) for c in color[:6]]
            
            # ADJ 7P channel order: R, G, B, W, Amber, UV, Dimmer, Strobe, [+ 5 more for 13ch]
            dimmer = int(255 * brightness)
            strobe = 255  # 255 = solid LED On (tested empirically)
            
            if num_channels >= 13:
                # 13-channel mode: R, G, B, W, Amber, UV, Dimmer, Strobe, ColorMacro, AutoProg, Speed, Fade, DimMode
                dmx_values = scaled_color + [dimmer, strobe, 0, 0, 0, 0, 0]
            else:
                # 8-channel mode: R, G, B, W, Amber, UV, Dimmer, Strobe
                dmx_values = scaled_color + [dimmer, strobe]
        
        channel.add_fade(dmx_values, fade_ms)
        
        # Update state
        self.house_light_state[fixture_name] = {
            "brightness": brightness,
            "color": color,
        }
    
    async def set_house_group(self, group_name: str, brightness: float,
                               color: Optional[List[int]] = None, fade_ms: int = 50):
        """Set all lights in a group to the same state.
        
        Args:
            group_name: Name of group (e.g., "left_tripod", "stage_pars")
            brightness: 0.0-1.0 brightness level
            color: Optional [R, G, B, Lime, Amber, UV] values
            fade_ms: Fade time in milliseconds
        """
        if group_name not in self.house_lights:
            print(f"⚠️  Unknown house group: {group_name}")
            return
        
        for fix_name in self.house_lights[group_name]:
            await self.set_house_light(fix_name, brightness, color, fade_ms)
    
    async def set_all_house_lights(self, brightness: float,
                                    color: Optional[List[int]] = None, fade_ms: int = 50):
        """Set all house lights to the same state."""
        for fix_name in self.house_light_channels:
            await self.set_house_light(fix_name, brightness, color, fade_ms)
    
    async def test_house_light(self, fixture_name: str, duration: float = 2.0):
        """Flash a house light for testing."""
        await self.set_house_light(fixture_name, 1.0, [255, 255, 255, 255, 255, 0], 0)
        await asyncio.sleep(duration)
        await self.set_house_light(fixture_name, 0.0, None, 0)
    
    async def test_house_group(self, group_name: str, duration: float = 2.0):
        """Flash all lights in a group for testing."""
        await self.set_house_group(group_name, 1.0, [255, 255, 255, 255, 255, 0], 200)
        await asyncio.sleep(duration)
        await self.set_house_group(group_name, 0.0, None, 200)
    
    def get_house_light_info(self) -> Dict:
        """Get info about all house lights for dashboard."""
        result = {}
        for group_name, fixtures in self.house_lights.items():
            result[group_name] = []
            for fix_name, fix_info in fixtures.items():
                state = self.house_light_state.get(fix_name, {"brightness": 0.0, "color": [0]*6})
                result[group_name].append({
                    "name": fix_name,
                    "dmx_address": fix_info["dmx_address"],
                    "type": fix_info["type"],
                    "brightness": state["brightness"],
                    "color": state["color"],
                })
        return result

    # ==================== MOVER CONTROLS ====================
    
    async def set_mover(self, name: str, pan: Optional[int] = None, tilt: Optional[int] = None,
                        dimmer: Optional[float] = None, zoom: Optional[int] = None,
                        color: Optional[int] = None):
        """Set mover position, brightness, and color."""
        if name not in self.mover_channels:
            print(f"⚠️  Unknown mover: {name}")
            return
        
        channel = self.mover_channels[name]
        info = self.movers[name]
        state = self.mover_state.get(name, {
            "pan": info.get("home_pan", 128),
            "tilt": info.get("home_tilt", 128),
            "dimmer": 0,
            "zoom": 128,
            "color": 0
        })
        
        # Update state with provided values
        if pan is not None:
            state["pan"] = pan
        if tilt is not None:
            state["tilt"] = tilt
        if dimmer is not None:
            state["dimmer"] = dimmer
        if zoom is not None:
            state["zoom"] = zoom
        if color is not None:
            state["color"] = color
        
        self.mover_state[name] = state
        
        mover_type = info.get("type", "")
        num_channels = info.get("channels", 15)
        dimmer_val = int(255 * state["dimmer"])
        color_val = state.get("color", 0)
        
        if "Small Mover" in mover_type or "15ch" in mover_type and "375ZX" not in mover_type:
            # Small mover 15ch: Pan, PanF, Tilt, TiltF, Color, Gobo, GoboRot, DIMMER, SHUTTER, Focus, Prism, Prog...
            dmx_values = [
                state["pan"], 0,    # Pan, Pan Fine
                state["tilt"], 0,   # Tilt, Tilt Fine
                color_val,          # Color wheel
                0,                  # Gobo = Open
                0,                  # Gobo Rotation
                dimmer_val,         # DIMMER
                255 if dimmer_val > 0 else 0,  # SHUTTER
                128,                # Focus
                0, 0, 0, 0, 0       # Prism, Programs
            ]
        elif "375ZX" in mover_type or "Intimidator" in mover_type:
            # Intimidator 375ZX 15ch: Pan, PanF, Tilt, TiltF, Speed, Color, Gobo, GoboRot, Prism, Focus, DIMMER, SHUTTER, Func, Macro, Zoom
            dmx_values = [
                state["pan"], 0,    # Pan, Pan Fine
                state["tilt"], 0,   # Tilt, Tilt Fine
                0,                  # Pan/Tilt Speed = fast
                color_val,          # Color wheel
                0,                  # Gobo = Open
                0,                  # Gobo Rotation
                0,                  # Prism = None
                128,                # Focus
                dimmer_val,         # DIMMER
                255 if dimmer_val > 0 else 0,  # SHUTTER = Open
                0,                  # Function
                0,                  # Movement Macros
                state["zoom"],      # Zoom
            ]
        else:
            # Generic 16-channel mover
            dmx_values = [state["pan"], 0, state["tilt"], 0, 0, dimmer_val, 255, color_val, 0, 0, 0, 0, 0, 0, 0, 0]
        
        channel.set_values(dmx_values[:num_channels])
        print(f"   🎯 {name}: Pan={state['pan']}, Tilt={state['tilt']}, Dim={state['dimmer']*100:.0f}%, Color={color_val}")
    
    def get_mover_info(self) -> Dict:
        """Get info about all movers for dashboard."""
        result = {}
        for name, info in self.movers.items():
            state = self.mover_state.get(name, {
                "pan": info.get("home_pan", 128),
                "tilt": info.get("home_tilt", 128),
                "dimmer": 0,
                "zoom": 128,
                "color": 0
            })
            mover_type = info.get("type", "")
            result[name] = {
                "dmx_address": info["dmx_address"],
                "type": mover_type,
                "pan": state["pan"],
                "tilt": state["tilt"],
                "dimmer": state["dimmer"],
                "zoom": state["zoom"],
                "color": state.get("color", 0),
                "has_zoom": "375ZX" in mover_type or "Intimidator" in mover_type,
            }
        return result
    
    async def _move_big_movers_to_character(self, character: str):
        """Move big movers to preset positions for a character."""
        if character not in self.mover_targets:
            print(f"⚠️  No mover targets defined for {character}")
            return
        
        targets = self.mover_targets[character]
        print(f"🎯 Moving big movers to {character}...")
        
        for mover_name, position in targets.items():
            if mover_name in self.mover_channels:
                pan = position.get("pan", 128)
                tilt = position.get("tilt", 128)
                zoom = position.get("zoom", 128)
                await self.set_mover(mover_name, pan=pan, tilt=tilt, dimmer=1.0, zoom=zoom)
                print(f"   🎯 {mover_name} -> Pan={pan}, Tilt={tilt}, Zoom={zoom}")
            else:
                print(f"   ⚠️  Unknown mover: {mover_name}")
    
    async def _movers_off(self):
        """Turn off all movers and center them."""
        print("🎯 Turning off and centering all movers...")
        for name, info in self.movers.items():
            home_pan = info.get("home_pan", 128)
            home_tilt = info.get("home_tilt", 128)
            await self.set_mover(name, pan=home_pan, tilt=home_tilt, dimmer=0, zoom=128)

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
