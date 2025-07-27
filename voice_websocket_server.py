#!/usr/bin/env python3
"""
WebSocket Server for Phone-based Voice Participation
Receives audio streams from mobile devices and integrates with the conversation system.
"""

import asyncio
import websockets
import json
import logging
import io
import numpy as np
from pathlib import Path
from typing import Dict, Set, Optional
from datetime import datetime
import tempfile
import subprocess

# Import our existing systems
from config_loader import load_config
from unified_voice_conversation_config import UnifiedVoiceConversation
from async_stt_module import STTResult

class PhoneAudioReceiver:
    """Receives and processes audio from phone connections."""
    
    def __init__(self, conversation_system: UnifiedVoiceConversation):
        self.conversation_system = conversation_system
        self.connected_speakers: Dict[str, Dict] = {}  # speaker_name -> connection_info
        self.active_connections: Set[websockets.WebSocketServerProtocol] = set()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        print("📱 Phone Audio Receiver initialized")
    
    async def register_client(self, websocket: websockets.WebSocketServerProtocol, speaker_name: str):
        """Register a new phone client."""
        self.active_connections.add(websocket)
        self.connected_speakers[speaker_name] = {
            'websocket': websocket,
            'connected_at': datetime.now(),
            'audio_count': 0
        }
        
        print(f"📱 {speaker_name} connected from {websocket.remote_address}")
        
        # Send confirmation
        await websocket.send(json.dumps({
            'type': 'status',
            'message': f'Connected as {speaker_name}',
            'level': 'info'
        }))
        
        # Update conversation system's connected speakers
        self.conversation_system.state.connected_speakers[speaker_name] = {
            'source': 'phone',
            'connected_at': datetime.now()
        }
    
    async def unregister_client(self, websocket: websockets.WebSocketServerProtocol, speaker_name: str = None):
        """Unregister a phone client."""
        self.active_connections.discard(websocket)
        
        if speaker_name and speaker_name in self.connected_speakers:
            del self.connected_speakers[speaker_name]
            print(f"📱 {speaker_name} disconnected")
            
            # Remove from conversation system
            if speaker_name in self.conversation_system.state.connected_speakers:
                del self.conversation_system.state.connected_speakers[speaker_name]
        else:
            # Find and remove by websocket
            to_remove = None
            for name, info in self.connected_speakers.items():
                if info['websocket'] == websocket:
                    to_remove = name
                    break
            
            if to_remove:
                del self.connected_speakers[to_remove]
                print(f"📱 {to_remove} disconnected")
                
                if to_remove in self.conversation_system.state.connected_speakers:
                    del self.conversation_system.state.connected_speakers[to_remove]
    
    async def process_audio_data(self, speaker_name: str, audio_data: bytes, audio_format: str):
        """Process received audio data and send to STT."""
        try:
            print(f"🎤 Processing audio from {speaker_name} ({len(audio_data)} bytes, {audio_format})")
            
            # Update audio count
            if speaker_name in self.connected_speakers:
                self.connected_speakers[speaker_name]['audio_count'] += 1
            
            # Convert WebM audio to format suitable for STT
            audio_array = await self.convert_webm_to_array(audio_data)
            
            if audio_array is not None and len(audio_array) > 0:
                # Create STT result and inject into conversation system
                await self.inject_phone_audio(speaker_name, audio_array)
            else:
                print(f"⚠️ Failed to convert audio from {speaker_name}")
                
        except Exception as e:
            print(f"❌ Error processing audio from {speaker_name}: {e}")
            self.logger.error(f"Audio processing error: {e}")
    
    async def convert_webm_to_array(self, webm_data: bytes) -> Optional[np.ndarray]:
        """Convert WebM audio data to numpy array suitable for STT."""
        try:
            # Create temporary files
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as webm_file:
                webm_file.write(webm_data)
                webm_path = webm_file.name
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as wav_file:
                wav_path = wav_file.name
            
            # Convert WebM to WAV using ffmpeg
            cmd = [
                'ffmpeg', '-y', '-i', webm_path,
                '-ar', '16000',  # 16kHz sample rate
                '-ac', '1',      # Mono
                '-f', 'wav',     # WAV format
                wav_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Read WAV file as numpy array
                import wave
                with wave.open(wav_path, 'rb') as wav:
                    frames = wav.readframes(wav.getnframes())
                    audio_array = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Cleanup temp files
                Path(webm_path).unlink(missing_ok=True)
                Path(wav_path).unlink(missing_ok=True)
                
                return audio_array
            else:
                print(f"❌ FFmpeg conversion failed: {result.stderr}")
                
                # Cleanup temp files
                Path(webm_path).unlink(missing_ok=True)
                Path(wav_path).unlink(missing_ok=True)
                
                return None
                
        except Exception as e:
            print(f"❌ Audio conversion error: {e}")
            return None
    
    async def inject_phone_audio(self, speaker_name: str, audio_array: np.ndarray):
        """Inject phone audio into the conversation system as if it came from STT."""
        try:
            # Use a simple local STT approach or integrate with Deepgram
            # For now, let's create a minimal STT result and inject it
            
            # You could integrate with Deepgram here for transcription
            # For now, let's create a placeholder that will trigger the conversation flow
            transcript = await self.transcribe_audio(audio_array)
            
            if transcript and transcript.strip():
                # Create STT result
                stt_result = STTResult(
                    text=transcript,
                    confidence=0.9,
                    is_final=True,
                    speaker_id=None,
                    speaker_name=speaker_name,
                    timestamp=datetime.now(),
                    raw_data={}
                )
                
                print(f"📝 Phone transcription ({speaker_name}): '{transcript}'")
                
                # Inject into conversation system
                await self.conversation_system._on_utterance_complete(stt_result)
            
        except Exception as e:
            print(f"❌ Error injecting phone audio: {e}")
            self.logger.error(f"Phone audio injection error: {e}")
    
    async def transcribe_audio(self, audio_array: np.ndarray) -> Optional[str]:
        """Transcribe audio using Deepgram STT service."""
        try:
            duration = len(audio_array) / 16000  # Assuming 16kHz sample rate
            if duration < 0.5:  # Skip very short audio
                return None
            
            # Convert numpy array to bytes for Deepgram
            audio_bytes = (audio_array * 32768).astype(np.int16).tobytes()
            
            # Use Deepgram prerecorded API for phone audio
            from deepgram import Deepgram
            
            # Get API key from config
            config = load_config('config.yaml')
            dg_client = Deepgram(config.conversation.deepgram_api_key)
            
            # Transcribe audio
            response = await dg_client.transcription.prerecorded(
                {
                    'buffer': audio_bytes,
                    'mimetype': 'audio/wav'
                },
                {
                    'punctuate': True,
                    'language': 'en',
                    'model': 'nova-2',
                    'smart_format': True
                }
            )
            
            # Extract transcript
            if response and 'results' in response:
                channels = response['results']['channels']
                if channels and len(channels) > 0:
                    alternatives = channels[0]['alternatives']
                    if alternatives and len(alternatives) > 0:
                        transcript = alternatives[0]['transcript'].strip()
                        confidence = alternatives[0].get('confidence', 0.0)
                        
                        if transcript and confidence > 0.6:  # Confidence threshold
                            return transcript
            
            return None
            
        except Exception as e:
            print(f"❌ Deepgram transcription error: {e}")
            self.logger.error(f"Transcription error: {e}")
            
            # Fallback placeholder for debugging
            duration = len(audio_array) / 16000
            return f"[Phone audio {duration:.1f}s - transcription failed]" if duration > 0.5 else None

class VoiceWebSocketServer:
    """WebSocket server for handling phone voice connections."""
    
    def __init__(self, host: str = 'localhost', port: int = 8765):
        self.host = host
        self.port = port
        self.audio_receiver: Optional[PhoneAudioReceiver] = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def initialize_conversation_system(self):
        """Initialize the conversation system."""
        config = load_config('config.yaml')
        self.conversation_system = UnifiedVoiceConversation(config)
        self.audio_receiver = PhoneAudioReceiver(self.conversation_system)
        
        print("🎙️ Conversation system initialized for phone integration")
    
    async def handle_client(self, websocket, path=None):
        """Handle individual client connections."""
        speaker_name = None
        
        try:
            print(f"📱 New connection from {websocket.remote_address}")
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    message_type = data.get('type')
                    
                    if message_type == 'register':
                        speaker_name = data.get('speaker_name', '').strip()
                        if speaker_name:
                            await self.audio_receiver.register_client(websocket, speaker_name)
                        else:
                            await websocket.send(json.dumps({
                                'type': 'status',
                                'message': 'Invalid speaker name',
                                'level': 'error'
                            }))
                    
                    elif message_type == 'audio' and speaker_name:
                        audio_data_list = data.get('audio_data', [])
                        audio_format = data.get('format', 'webm')
                        
                        if audio_data_list:
                            # Convert list back to bytes
                            audio_data = bytes(audio_data_list)
                            await self.audio_receiver.process_audio_data(
                                speaker_name, audio_data, audio_format
                            )
                    
                    else:
                        await websocket.send(json.dumps({
                            'type': 'status',
                            'message': 'Unknown message type or not registered',
                            'level': 'error'
                        }))
                        
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        'type': 'status',
                        'message': 'Invalid JSON message',
                        'level': 'error'
                    }))
                except Exception as e:
                    print(f"❌ Error handling message: {e}")
                    await websocket.send(json.dumps({
                        'type': 'status',
                        'message': 'Server error processing message',
                        'level': 'error'
                    }))
        
        except websockets.exceptions.ConnectionClosed:
            print(f"📱 Connection closed: {websocket.remote_address}")
        except Exception as e:
            print(f"❌ Client handler error: {e}")
        finally:
            await self.audio_receiver.unregister_client(websocket, speaker_name)
    
    async def start_server(self):
        """Start the WebSocket server."""
        await self.initialize_conversation_system()
        
        print(f"🌐 Starting WebSocket server on {self.host}:{self.port}")
        print(f"📱 Phones can connect to: ws://{self.host}:{self.port}/voice")
        
        # Create a wrapper to properly bind the method
        async def websocket_handler(websocket):
            # We don't need the path for our use case
            await self.handle_client(websocket, None)
        
        server = await websockets.serve(
            websocket_handler,
            self.host,
            self.port,
            subprotocols=['voice']
        )
        
        print("✅ WebSocket server started successfully!")
        print("🎭 Ready for phone connections to the Claude Sonnet funeral...")
        
        return server

# QR Code generator for easy phone access
def generate_qr_code(server_url: str, output_path: str = "webapp_qr.png"):
    """Generate QR code for easy phone access."""
    try:
        import qrcode
        
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(server_url)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        img.save(output_path)
        
        print(f"📱 QR code generated: {output_path}")
        print(f"🔗 URL: {server_url}")
        
    except ImportError:
        print("⚠️ QR code generation requires 'qrcode' package:")
        print("   pip install qrcode[pil]")
        print(f"🔗 Manual URL for phones: {server_url}")

async def main():
    """Main function to run the voice WebSocket server."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Voice WebSocket Server for Phone Integration')
    parser.add_argument('--host', default='localhost', help='Server host (default: localhost)')
    parser.add_argument('--port', type=int, default=8765, help='Server port (default: 8765)')
    parser.add_argument('--public-host', help='Public hostname for QR code (e.g., your-ip or domain)')
    
    args = parser.parse_args()
    
    # Create and start server
    server_instance = VoiceWebSocketServer(args.host, args.port)
    server = await server_instance.start_server()
    
    # Generate QR code
    public_host = args.public_host or args.host
    server_url = f"http://{public_host}:{args.port}/webapp/"
    generate_qr_code(server_url)
    
    try:
        # Keep server running
        await server.wait_closed()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down server...")
        server.close()
        await server.wait_closed()
        print("✅ Server stopped")

if __name__ == "__main__":
    asyncio.run(main()) 