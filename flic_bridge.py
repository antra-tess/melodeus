#!/usr/bin/env python3
"""
Flic button TCP-to-WebSocket bridge.
Receives TCP messages from Flic Hub and forwards to WebSocket UI server.
"""

import asyncio
import json
import websockets
import logging
import subprocess
from typing import Optional

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("flic-bridge")

class FlicBridge:
    def __init__(self, tcp_host='0.0.0.0', tcp_port=11235, ws_url='ws://localhost:8765'):
        self.tcp_host = tcp_host
        self.tcp_port = tcp_port
        self.ws_url = ws_url
        self.ws_client: Optional[websockets.WebSocketClientProtocol] = None
        self.reconnect_task = None
        
        # Map BD addresses to actions (in order from your test)
        self.button_map = {
            '90:88:a9:50:65:83 ': {'action': 'force_interrupt'},                    # Button 1
            '90:88:a9:50:68:2c': {'action': 'trigger_speaker', 'speaker': 'Opus'},         # Button 2
            '90:88:a9:50:6b:32': {'action': 'trigger_speaker', 'speaker': 'Sonnet36'},       # Button 3
            '90:88:a9:50:67:2c': {'action': 'trigger_speaker', 'speaker': 'Sonnet37'},      # Button 4
            '90:88:a9:50:6b:3d': {'action': 'trigger_speaker', 'speaker': 'Haiku'},        # Button 5
            '90:88:a9:50:63:a0': {'action': 'trigger_speaker', 'speaker': 'Opus4'},         # Button 6
            '90:88:a9:50:65:8c': {'action': 'trigger_speaker', 'speaker': 'Sonnet4'},       # Button 7
            '90:88:a9:50:65:6b': {'action': 'trigger_speaker', 'speaker': 'Sonnet35'},       # Button 8
            '90:88:a9:50:6b:10': {'action': 'trigger_speaker', 'speaker': '3Sonnet'}         # Button 9
        }
        
    def clear_port(self):
        """Kill any existing process using the TCP port."""
        try:
            # Find process using the port
            result = subprocess.run(['lsof', '-ti', f':{self.tcp_port}'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout.strip():
                pid = result.stdout.strip()
                logger.info(f"Found existing process {pid} using port {self.tcp_port}, killing it...")
                
                # Kill the process
                subprocess.run(['kill', '-9', pid], check=True)
                logger.info(f"Successfully killed process {pid}")
            else:
                logger.info(f"Port {self.tcp_port} is free")
                
        except subprocess.CalledProcessError as e:
            logger.warning(f"Error clearing port {self.tcp_port}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error clearing port {self.tcp_port}: {e}")
        
    async def connect_websocket(self):
        """Connect to WebSocket server with auto-reconnect."""
        while True:
            try:
                logger.info(f"Connecting to WebSocket {self.ws_url}")
                self.ws_client = await websockets.connect(self.ws_url)
                
                # Send identification
                await self.ws_client.send(json.dumps({
                    "type": "client_ready",
                    "client_id": "flic-bridge"
                }))
                
                logger.info("WebSocket connected")
                
                # Keep connection alive
                async for message in self.ws_client:
                    # Handle any server messages if needed
                    pass
                    
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self.ws_client = None
                await asyncio.sleep(0.5)  # Reconnect delay
                
    async def handle_tcp_client(self, reader, writer):
        """Handle incoming TCP connection from Flic Hub."""
        addr = writer.get_extra_info('peername')
        logger.info(f"Flic Hub connected from {addr}")
        
        try:
            buffer = ""
            while True:
                data = await reader.read(1024)
                if not data:
                    print("No data received")
                    break
                print(f"Received data: {data}")
                    
                # Accumulate data and process complete JSON lines
                buffer += data.decode('utf-8')
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        await self.process_message(line.strip())
                        
        except Exception as e:
            logger.error(f"TCP error: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
            logger.info(f"Flic Hub disconnected from {addr}")
            
    async def process_message(self, message: str):
        """Process message from Flic Hub."""
        try:
            data = json.loads(message)
            logger.info(f"Received: {data}")
            
            # Handle raw button events
            if 'bdaddr' in data and 'isSingleClick' in data:
                # This is a button event
                bdaddr = data['bdaddr']
                
                # Only process single clicks
                if not data.get('isSingleClick', False):
                    return
                    
                # Look up the action for this button
                if bdaddr in self.button_map:
                    button_action = self.button_map[bdaddr]
                    button_num = list(self.button_map.keys()).index(bdaddr) + 1
                    
                    # Forward to WebSocket
                    if self.ws_client:
                        try:
                            # Build the WebSocket message based on action type
                            if button_action['action'] == 'force_interrupt':
                                ws_message = {'type': 'force_interrupt'}
                                logger.info(f"🛑 Button {button_num}: FORCE INTERRUPT")
                            else:  # trigger_speaker
                                ws_message = {
                                    'type': 'trigger_speaker',
                                    'data': {
                                        'speaker': button_action['speaker']
                                    }
                                }
                                logger.info(f"🎭 Button {button_num}: Trigger {button_action['speaker']}")
                                
                            # Log what we're sending for debugging
                            logger.debug(f"Sending to WebSocket: {ws_message}")
                            await self.ws_client.send(json.dumps(ws_message))
                        except websockets.exceptions.ConnectionClosed:
                            logger.warning("WebSocket connection closed - button press ignored")
                            self.ws_client = None
                    else:
                        logger.warning("WebSocket not connected - button press ignored")
                else:
                    logger.warning(f"Unknown button: {bdaddr}")
                    
            # Handle hub info messages
            elif data.get('type') == 'hubInfo':
                logger.info(f"Flic Hub: {data.get('serial')} "
                          f"(FW: {data.get('firmware')})")
                          
            # Handle button info messages  
            elif data.get('type') == 'buttonInfo':
                button_data = data.get('button', {})
                bdaddr = button_data.get('bdaddr', 'unknown')
                name = button_data.get('name', 'unnamed')
                if bdaddr in self.button_map:
                    action = self.button_map[bdaddr]
                    button_num = list(self.button_map.keys()).index(bdaddr) + 1
                    logger.info(f"Button {button_num} ({name}): {action}")
                    
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON: {message}")
        except Exception as e:
            logger.error(f"Process error: {e}")
            
    async def start(self):
        """Start the bridge server."""
        # Clear any existing process on the port
        self.clear_port()
        
        # Start WebSocket connection task
        self.reconnect_task = asyncio.create_task(self.connect_websocket())
        
        # Start TCP server
        server = await asyncio.start_server(
            self.handle_tcp_client, 
            self.tcp_host, 
            self.tcp_port
        )
        
        addr = server.sockets[0].getsockname()
        logger.info(f"🌉 Flic bridge listening on {addr[0]}:{addr[1]}")
        logger.info(f"   TCP: {self.tcp_host}:{self.tcp_port} <- Flic Hub")
        logger.info(f"   WebSocket: {self.ws_url} -> Voice System")
        
        async with server:
            await server.serve_forever()

async def main():
    """Run the bridge."""
    bridge = FlicBridge(
        tcp_host='0.0.0.0',  # Listen on all interfaces
        tcp_port=11235,      # Flic Hub connects here
        ws_url='ws://localhost:8765'  # Forward to UI server
    )
    
    try:
        await bridge.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")

if __name__ == "__main__":
    asyncio.run(main())