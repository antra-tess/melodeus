#!/usr/bin/env python3
"""
Async STT Module for Deepgram Real-time Speech-to-Text
Provides callback-based speech recognition with speaker identification.
"""

import asyncio
import time
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any, List, Tuple
from datetime import datetime
from enum import Enum

from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions

try:
    from titanet_voice_fingerprinting import TitaNetVoiceFingerprinter, create_word_timing_from_deepgram_word
    VOICE_FINGERPRINTING_AVAILABLE = True
    print("✅ TitaNet voice fingerprinting available")
except ImportError:
    VOICE_FINGERPRINTING_AVAILABLE = False
    print("❌ TitaNet voice fingerprinting not available (requires nemo_toolkit)")
    print("💡 Install with: pip install 'nemo_toolkit[asr]'")

# Try to use mel-aec first, fall back to pyaudio if not available
try:
    from mel_aec_adapter import PyAudio as MelAecAudio, paInt16
    print("✅ Using mel-aec for audio I/O with integrated AEC")
    USING_MEL_AEC = True
except ImportError:
    import pyaudio
    from pyaudio import paInt16
    PyAudio = pyaudio.PyAudio
    print("⚠️  Using PyAudio (mel-aec not available)")
    USING_MEL_AEC = False

# Legacy echo cancellation imports (only used if mel-aec is not available)
if not USING_MEL_AEC:
    try:
        from improved_echo_cancellation import SimpleEchoCancellationProcessor
        from debug_echo_cancellation import DebugEchoCancellationProcessor
        from advanced_echo_cancellation import AdaptiveEchoCancellationProcessor
        from improved_aec_processor import ImprovedEchoCancellationProcessor
        ECHO_CANCELLATION_AVAILABLE = True
        print("✅ Legacy echo cancellation available")
    except ImportError:
        ECHO_CANCELLATION_AVAILABLE = False
        print("❌ Legacy echo cancellation not available")
        print("💡 Install with: pip install speexdsp")
else:
    # mel-aec has built-in AEC
    ECHO_CANCELLATION_AVAILABLE = False  # Disable legacy AEC

def find_input_device_by_name(device_name: str) -> Optional[int]:
    """Find audio input device index by name (partial match)."""
    if not device_name:
        return None
        
    if USING_MEL_AEC:
        p = MelAecAudio()
    else:
        p = pyaudio.PyAudio()
    try:
        device_name_lower = device_name.lower()
        
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:  # Only check input devices
                if device_name_lower in info['name'].lower():
                    print(f"🎯 Found input device: '{info['name']}' (index {i}) for name '{device_name}'")
                    return i
        
        print(f"⚠️ No input device found matching '{device_name}'")
        print("Available input devices:")
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"  - {info['name']}")
        return None
        
    finally:
        p.terminate()

class STTEventType(Enum):
    """Types of STT events."""
    UTTERANCE_COMPLETE = "utterance_complete"
    INTERIM_RESULT = "interim_result"
    SPEECH_STARTED = "speech_started"
    SPEECH_ENDED = "speech_ended"
    SPEAKER_CHANGE = "speaker_change"
    CONNECTION_OPENED = "connection_opened"
    CONNECTION_CLOSED = "connection_closed"
    ERROR = "error"

@dataclass
class STTResult:
    """Result from speech-to-text recognition."""
    text: str
    confidence: float
    is_final: bool
    speaker_id: Optional[int] = None
    speaker_name: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    raw_data: Optional[Dict[str, Any]] = None

@dataclass
class STTConfig:
    """Configuration for STT settings."""
    api_key: str
    model: str = "nova-3"
    language: str = "en-US"
    sample_rate: int = 16000
    chunk_size: int = 8000
    channels: int = 1
    smart_format: bool = True
    interim_results: bool = True
    punctuate: bool = True
    diarize: bool = True
    utterance_end_ms: int = 1000
    vad_events: bool = True
    # Audio input device
    input_device_name: Optional[str] = None  # None = default device, or specify device name
    # Speaker identification settings
    enable_speaker_id: bool = False
    speaker_profiles_path: Optional[str] = None
    # Custom vocabulary/keywords for better recognition
    keywords: Optional[List[Tuple[str, float]]] = None  # List of (word, weight) tuples
    # Debug settings
    debug_speaker_data: bool = False  # Enable detailed speaker/timing debug output

class AsyncSTTStreamer:
    """Async STT Streamer with callback support."""
    
    def __init__(self, config: STTConfig, speakers_config=None):
        self.config = config
        self.speakers_config = speakers_config
        self.is_listening = False
        self.connection = None
        self.microphone = None
        self.event_loop = None
        self.audio_task = None
        self.connection_alive = False
        
        # Audio setup
        if USING_MEL_AEC:
            self.p = MelAecAudio()
        else:
            self.p = pyaudio.PyAudio()
        
        # Deepgram client
        self.deepgram = DeepgramClient(config.api_key)
        
        # Callback registry
        self.callbacks: Dict[STTEventType, List[Callable]] = {
            event_type: [] for event_type in STTEventType
        }
        
        # State tracking
        self.current_speaker = None
        self.last_utterance_time = None
        self.session_speakers = {}  # Maps session speaker IDs to names
        self.is_paused = False  # For pause/resume functionality
        
        # TitaNet voice fingerprinting (optional)
        self.voice_fingerprinter = None
        if VOICE_FINGERPRINTING_AVAILABLE and speakers_config and config.enable_speaker_id:
            try:
                # Enable debug audio saving if debug_speaker_data is enabled
                debug_save_audio = getattr(config, 'debug_speaker_data', False)
                self.voice_fingerprinter = TitaNetVoiceFingerprinter(speakers_config, debug_save_audio=debug_save_audio)
                print(f"🤖 TitaNet voice fingerprinting enabled")
                if debug_save_audio:
                    print(f"🐛 Debug audio saving enabled - extracted segments will be saved to debug_audio_segments/")
            except Exception as e:
                print(f"⚠️  TitaNet voice fingerprinting failed to initialize: {e}")
        
        # Speaker identification (optional)
        self.speaker_identifier = None
        if config.enable_speaker_id:
            self._setup_speaker_identification()
        
        # Reconnection settings
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 1.0  # seconds
        self.is_reconnecting = False
        
        # Keepalive settings
        self.keepalive_task = None
        self.keepalive_interval = 10.0  # seconds
        
        # Echo cancellation (optional)
        self.echo_canceller = None
        # Only use legacy echo cancellation if not using mel-aec (which has built-in AEC)
        if not USING_MEL_AEC and ECHO_CANCELLATION_AVAILABLE and hasattr(config, 'enable_echo_cancellation') and config.enable_echo_cancellation:
            try:
                # Get AEC parameters from config or use defaults
                frame_size = getattr(config, 'aec_frame_size', 256)
                filter_length = getattr(config, 'aec_filter_length', 2048)
                delay_ms = getattr(config, 'aec_delay_ms', 100)
                
                # Use improved version for better handling of bursty TTS
                self.echo_canceller = ImprovedEchoCancellationProcessor(
                    frame_size=frame_size,
                    filter_length=filter_length,
                    sample_rate=config.sample_rate,
                    initial_delay_ms=delay_ms,
                    debug_level=1  # Basic debug info
                )
                print(f"🔊 Legacy echo cancellation enabled")
            except Exception as e:
                print(f"⚠️  Echo cancellation failed to initialize: {e}")
                self.echo_canceller = None
        elif USING_MEL_AEC and hasattr(config, 'enable_echo_cancellation') and config.enable_echo_cancellation:
            print(f"🔊 mel-aec built-in echo cancellation is active")
    
    def on(self, event_type: STTEventType, callback: Callable):
        """Register a callback for an event type."""
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        self.callbacks[event_type].append(callback)
    
    def off(self, event_type: STTEventType, callback: Callable):
        """Unregister a callback for an event type."""
        if event_type in self.callbacks and callback in self.callbacks[event_type]:
            self.callbacks[event_type].remove(callback)
    
    async def start_listening(self) -> bool:
        """Start listening for speech."""
        if self.is_listening:
            await self.stop_listening()
        
        # Store the current event loop for scheduling tasks from sync callbacks
        self.event_loop = asyncio.get_event_loop()
        
        try:
            # Setup microphone in executor to avoid blocking
            loop = asyncio.get_event_loop()
            
            # Debug: Check what sample rates are actually supported
            print(f"🔧 [AUDIO DEBUG] Requested sample rate: {self.config.sample_rate}Hz")
            print(f"🔧 [AUDIO DEBUG] Chunk size: {self.config.chunk_size} samples")
            
            # Resolve input device name to index if specified
            input_device_index = None
            if self.config.input_device_name is not None:
                input_device_index = find_input_device_by_name(self.config.input_device_name)
                if input_device_index is not None:
                    print(f"🎤 Using input device: '{self.config.input_device_name}' (index {input_device_index})")
            
            # Create microphone stream
            stream_kwargs = {
                'format': paInt16,
                'channels': self.config.channels,
                'rate': self.config.sample_rate,
                'input': True,
                'frames_per_buffer': self.config.chunk_size
            }
            
            if input_device_index is not None:
                stream_kwargs['input_device_index'] = input_device_index
            
            self.microphone = await loop.run_in_executor(
                None,
                lambda: self.p.open(**stream_kwargs)
            )
            
            # Verify the actual sample rate being used
            actual_rate = self.microphone._rate
            print(f"🔧 [AUDIO DEBUG] Actual microphone rate: {actual_rate}Hz")
            if actual_rate != self.config.sample_rate:
                print(f"⚠️ [AUDIO DEBUG] SAMPLE RATE MISMATCH! Expected {self.config.sample_rate}Hz, got {actual_rate}Hz")
            
            # Setup Deepgram connection
            self.connection = self.deepgram.listen.websocket.v("1")
            
            # Configure options with explicit audio format
            options_dict = {
                "model": self.config.model,
                "language": self.config.language,
                "encoding": "linear16",  # Critical: Explicit audio encoding
                "sample_rate": self.config.sample_rate,  # Explicit sample rate
                "channels": self.config.channels,  # Explicit channel count
                "smart_format": self.config.smart_format,
                "interim_results": self.config.interim_results,
                "punctuate": self.config.punctuate,
                "diarize": self.config.diarize,
                "utterance_end_ms": self.config.utterance_end_ms,
                "vad_events": self.config.vad_events
            }
            
            # Add keywords/keyterms based on model
            if self.config.keywords:
                if self.config.model == "nova-3":
                    # Nova-3 uses keyterm parameter with space-separated terms
                    # Extract just the words (ignore weights for nova-3)
                    keyterms = []
                    for word, weight in self.config.keywords:
                        # Keep the original formatting for proper nouns
                        keyterms.append(word)
                    
                    if keyterms:
                        # Join with spaces for nova-3
                        keyterm_string = " ".join(keyterms)
                        options_dict["keyterm"] = keyterm_string
                        print(f"🔤 Using keyterms (Nova-3): {keyterms[:5]}...")
                        print(f"   Full keyterm string: {keyterm_string[:100]}...")
                else:
                    # Other models use keywords with weights
                    sanitized_keywords = []
                    for word, weight in self.config.keywords:
                        # Replace spaces with underscores for older models
                        sanitized_word = word.replace(" ", "_").replace(",", "")
                        if sanitized_word:
                            sanitized_keywords.append(f"{sanitized_word}:{weight}")
                    
                    if sanitized_keywords:
                        keyword_string = ",".join(sanitized_keywords)
                        options_dict["keywords"] = keyword_string
                        print(f"🔤 Using keywords: {sanitized_keywords[:5]}...")
                        print(f"   Full keyword string: {keyword_string[:100]}...")
            
            options = LiveOptions(**options_dict)
            
            # Set up event handlers
            self.connection.on(LiveTranscriptionEvents.Open, self._on_open)
            self.connection.on(LiveTranscriptionEvents.Transcript, self._on_transcript)
            self.connection.on(LiveTranscriptionEvents.UtteranceEnd, self._on_utterance_end)
            self.connection.on(LiveTranscriptionEvents.SpeechStarted, self._on_speech_started)
            self.connection.on(LiveTranscriptionEvents.Error, self._on_error)
            self.connection.on(LiveTranscriptionEvents.Close, self._on_close)
            
            # Start connection with proper async handling
            connection_success = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.connection.start(options)
            )
            
            if connection_success:
                self.is_listening = True
                # Don't set connection_alive yet - wait for _on_open callback
                
                # Minimal wait for connection to stabilize
                await asyncio.sleep(0.01)
                
                # Verify connection is actually alive before starting audio
                if self.connection_alive:
                    # Start audio streaming task and track it
                    self.audio_task = asyncio.create_task(self._audio_streaming_loop())
                    
                    # Start keepalive task to prevent connection timeout
                    self.keepalive_task = asyncio.create_task(self._keepalive_loop())
                    
                    await self._emit_event(STTEventType.CONNECTION_OPENED, {
                        "message": "STT connection established"
                    })
                    
                    # Reset reconnection attempts on successful connection
                    self.reconnect_attempts = 0
                    
                    return True
                else:
                    await self._emit_event(STTEventType.ERROR, {
                        "error": "Connection opened but not confirmed alive"
                    })
                    return False
            else:
                await self._emit_event(STTEventType.ERROR, {
                    "error": "Failed to start Deepgram connection"
                })
                return False
                
        except Exception as e:
            await self._emit_event(STTEventType.ERROR, {
                "error": f"Failed to start listening: {e}"
            })
            return False
    
    async def pause(self):
        """Pause STT processing (keeps connection alive but stops sending audio)."""
        if not self.is_paused:
            self.is_paused = True
            print("⏸️ STT paused")
            
    async def resume(self):
        """Resume STT processing."""
        if self.is_paused:
            self.is_paused = False
            print("▶️ STT resumed")
    
    async def _handle_reconnection(self):
        """Handle automatic reconnection with exponential backoff."""
        self.is_reconnecting = True
        
        while self.reconnect_attempts < self.max_reconnect_attempts and self.is_listening:
            self.reconnect_attempts += 1
            wait_time = self.reconnect_delay * (2 ** (self.reconnect_attempts - 1))  # Exponential backoff
            
            print(f"🔄 Reconnection attempt {self.reconnect_attempts}/{self.max_reconnect_attempts} in {wait_time}s...")
            await asyncio.sleep(wait_time)
            
            try:
                # Clean up old connection
                if self.connection:
                    try:
                        self.connection.finish()
                    except:
                        pass
                
                # Stop audio task if still running
                if self.audio_task and not self.audio_task.done():
                    self.audio_task.cancel()
                    try:
                        await self.audio_task
                    except asyncio.CancelledError:
                        pass
                
                # Try to reconnect
                print("🔌 Attempting to reconnect to Deepgram...")
                success = await self._reconnect()
                
                if success:
                    print("✅ Successfully reconnected to Deepgram!")
                    self.is_reconnecting = False
                    return
                    
            except Exception as e:
                print(f"❌ Reconnection attempt {self.reconnect_attempts} failed: {e}")
        
        # Max attempts reached
        print(f"❌ Failed to reconnect after {self.max_reconnect_attempts} attempts")
        self.is_reconnecting = False
        self.is_listening = False
        
        await self._emit_event(STTEventType.ERROR, {
            "error": "Failed to reconnect to Deepgram after multiple attempts"
        })
    
    async def _reconnect(self):
        """Attempt to reconnect to Deepgram."""
        # Reset connection state
        self.connection = None
        self.connection_alive = False
        
        # Recreate connection
        self.connection = self.deepgram.listen.websocket.v("1")
        
        # Reapply all the same options
        options_dict = {
            "model": self.config.model,
            "language": self.config.language,
            "encoding": "linear16",
            "sample_rate": self.config.sample_rate,
            "channels": self.config.channels,
            "smart_format": self.config.smart_format,
            "interim_results": self.config.interim_results,
            "punctuate": self.config.punctuate,
            "diarize": self.config.diarize,
            "utterance_end_ms": self.config.utterance_end_ms,
            "vad_events": self.config.vad_events
        }
        
        # Re-add keywords/keyterms
        if self.config.keywords:
            if self.config.model == "nova-3":
                keyterms = [word for word, weight in self.config.keywords]
                if keyterms:
                    options_dict["keyterm"] = " ".join(keyterms)
            else:
                sanitized_keywords = []
                for word, weight in self.config.keywords:
                    sanitized_word = word.replace(" ", "_").replace(",", "")
                    if sanitized_word:
                        sanitized_keywords.append(f"{sanitized_word}:{weight}")
                if sanitized_keywords:
                    options_dict["keywords"] = ",".join(sanitized_keywords)
        
        options = LiveOptions(**options_dict)
        
        # Re-setup event handlers
        self.connection.on(LiveTranscriptionEvents.Open, self._on_open)
        self.connection.on(LiveTranscriptionEvents.Transcript, self._on_transcript)
        self.connection.on(LiveTranscriptionEvents.UtteranceEnd, self._on_utterance_end)
        self.connection.on(LiveTranscriptionEvents.SpeechStarted, self._on_speech_started)
        self.connection.on(LiveTranscriptionEvents.Error, self._on_error)
        self.connection.on(LiveTranscriptionEvents.Close, self._on_close)
        
        # Start connection
        loop = asyncio.get_event_loop()
        connection_success = await loop.run_in_executor(
            None, lambda: self.connection.start(options)
        )
        
        if connection_success:
            # Wait for connection to be confirmed
            await asyncio.sleep(0.1)
            
            if self.connection_alive:
                # Restart audio streaming
                self.audio_task = asyncio.create_task(self._audio_streaming_loop())
                
                # Restart keepalive
                self.keepalive_task = asyncio.create_task(self._keepalive_loop())
                
                return True
        
        return False
    
    async def stop_listening(self):
        """Stop listening for speech."""
        print("🛑 STT stop requested")
        self.is_listening = False
        self.connection_alive = False
        
        # Cancel keepalive task if it's running
        if self.keepalive_task and not self.keepalive_task.done():
            print("🔄 Cancelling keepalive task...")
            self.keepalive_task.cancel()
            try:
                await self.keepalive_task
            except asyncio.CancelledError:
                pass
        
        # Cancel audio streaming task if it's running
        if self.audio_task and not self.audio_task.done():
            print("🔄 Cancelling audio streaming task...")
            self.audio_task.cancel()
            try:
                await self.audio_task
            except asyncio.CancelledError:
                print("✅ Audio streaming task cancelled")
            except Exception as e:
                print(f"Error cancelling audio task: {e}")
            self.audio_task = None
        
        # Close microphone
        if self.microphone:
            try:
                # Use executor to avoid blocking on microphone operations
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    lambda: (
                        self.microphone.stop_stream(),
                        self.microphone.close()
                    )
                )
            except Exception as e:
                print(f"Error closing microphone: {e}")
            self.microphone = None
        
        # Close Deepgram connection
        if self.connection:
            try:
                # Use executor to avoid blocking on connection close
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.connection.finish)
            except Exception as e:
                print(f"Error closing Deepgram connection: {e}")
            self.connection = None
        
        await self._emit_event(STTEventType.CONNECTION_CLOSED, {
            "message": "STT connection closed"
        })
        
        print("✅ STT stopped")
    
    def is_currently_listening(self) -> bool:
        """Check if STT is currently listening."""
        return self.is_listening
    
    def get_session_speakers(self) -> Dict[int, str]:
        """Get the mapping of session speaker IDs to names."""
        return self.session_speakers.copy()
    
    def register_speaker(self, session_speaker_id: int, name: str):
        """Register a name for a session speaker ID."""
        self.session_speakers[session_speaker_id] = name
        print(f"📝 Registered speaker {session_speaker_id} as '{name}'")
    
    async def _keepalive_loop(self):
        """Send keepalive messages to prevent connection timeout."""
        while self.is_listening and self.connection_alive:
            try:
                # Send a keepalive message every interval
                if self.connection:
                    # Deepgram expects a JSON message for keepalive
                    keepalive_message = json.dumps({"type": "KeepAlive"})
                    self.connection.send(keepalive_message)
                await asyncio.sleep(self.keepalive_interval)
            except Exception as e:
                print(f"❌ Keepalive error: {e}")
                break
    
    async def _audio_streaming_loop(self):
        """Main audio streaming loop with proper async handling."""
        print("🎵 Starting audio streaming loop...")
        chunk_count = 0
        last_stats_time = time.time()
        
        # Track stream start time for voice fingerprinting synchronization
        self.stream_start_time = time.time()
        if self.voice_fingerprinter:
            print(f"🕐 [STREAM] Stream started at {self.stream_start_time:.3f}")
        
        # Minimal wait for connection to be fully established
        await asyncio.sleep(0.001)
        
        try:
            while self.is_listening and self.microphone and self.connection and self.connection_alive:
                try:
                    # Check if we should still be running
                    if not self.connection_alive:
                        print("🔌 Connection no longer alive, stopping audio stream")
                        break
                    
                    # Skip sending audio if paused
                    if self.is_paused:
                        await asyncio.sleep(0.1)  # Small delay to prevent busy loop
                        continue
                        
                    # Read audio data in executor to avoid blocking
                    try:
                        loop = asyncio.get_event_loop()
                        data = await loop.run_in_executor(
                            None, 
                            lambda: self.microphone.read(
                                self.config.chunk_size, 
                                exception_on_overflow=False
                            )
                        )
                    except Exception as read_error:
                        print(f"❌ Failed to read audio data: {read_error}")
                        break
                    
                    
                    # Apply echo cancellation if enabled
                    processed_data = data
                    if self.echo_canceller:
                        try:
                            # Debug: log timing every 100 chunks
                            if chunk_count % 100 == 0:
                                print(f"🎤 MIC: Processing {len(data)} bytes at {time.time():.3f}")
                            processed_data = self.echo_canceller.process(data)
                        except Exception as aec_error:
                            # Don't break streaming if AEC fails
                            if chunk_count % 1000 == 0:  # Only log occasionally
                                print(f"⚠️  Echo cancellation error: {aec_error}")
                            processed_data = data  # Fall back to original
                    
                    # Convert audio data to numpy array for voice fingerprinting
                    if self.voice_fingerprinter:
                        try:
                            # Convert bytes to int16 numpy array, then to float32
                            audio_np = np.frombuffer(processed_data, dtype=np.int16).astype(np.float32) / 32768.0
                            # Feed to voice fingerprinter with stream-relative timestamp
                            stream_relative_time = time.time() - self.stream_start_time
                            self.voice_fingerprinter.add_audio_chunk(audio_np, stream_relative_time)
                        except Exception as fp_error:
                            # Don't break streaming if fingerprinting fails
                            if chunk_count % 1000 == 0:  # Only log occasionally
                                print(f"⚠️  Voice fingerprinting error: {fp_error}")
                    
                    # Try to send data to Deepgram
                    try:
                        self.connection.send(processed_data)
                        chunk_count += 1
                        
                        # Less frequent debug output
                        if chunk_count % 100 == 0:
                            print(f"📊 Sent {chunk_count} audio chunks")
                            
                        # Print echo cancellation stats every 10 seconds
                        # if self.echo_canceller and hasattr(self.echo_canceller, 'print_stats'):
                        #     current_time = time.time()
                        #     if current_time - last_stats_time > 10:
                        #         #self.echo_canceller.print_stats()
                        #         last_stats_time = current_time
                            
                    except Exception as send_error:
                        print(f"❌ Failed to send audio data to Deepgram: {send_error}")
                        print(f"📊 Sent {chunk_count} chunks before connection failed")
                        # Connection is likely closed, mark as not alive
                        self.connection_alive = False
                        await self._emit_event(STTEventType.ERROR, {
                            "error": f"Connection send failed: {send_error}"
                        })
                        break
                    
                    # Brief async sleep to yield control
                    await asyncio.sleep(0.001)  # 1ms delay
                    
                except Exception as e:
                    print(f"❌ Audio streaming error: {e}")
                    await self._emit_event(STTEventType.ERROR, {
                        "error": f"Audio streaming error: {e}"
                    })
                    break
                    
        except asyncio.CancelledError:
            print("🔄 Audio streaming loop cancelled")
            raise
        except Exception as e:
            print(f"❌ Audio streaming loop fatal error: {e}")
            await self._emit_event(STTEventType.ERROR, {
                "error": f"Audio streaming loop error: {e}"
            })
        finally:
            print(f"🏁 Audio streaming loop ended. Sent {chunk_count} chunks total.")
    
    def _on_open(self, *args, **kwargs):
        """Handle Deepgram connection open."""
        print("🔗 Deepgram STT connection opened")
        self.connection_alive = True
    
    def _on_transcript(self, *args, **kwargs):
        """Handle incoming transcripts."""
        result = kwargs.get('result')
        if not result or not hasattr(result, 'channel'):
            return
        
        try:
            channel = result.channel
            alternative = channel.alternatives[0]
            transcript = alternative.transcript.strip()
            confidence = alternative.confidence
            is_final = result.is_final
            
            # # Debug: Print raw response structure for final results
            # if is_final and hasattr(self.config, 'debug_speaker_data') and self.config.debug_speaker_data:
            #     print(f"🔬 [RAW DEBUG] Full response structure:")
            #     print(f"  result.is_final: {is_final}")
            #     print(f"  channel has alternatives: {hasattr(channel, 'alternatives')}")
            #     if hasattr(channel, 'alternatives') and channel.alternatives and len(channel.alternatives) > 0 and len(channel.alternatives[0].words) > 0:
            #         alt = channel.alternatives[0]
            #         print(f"  alternative.transcript: '{alt.transcript}'")
            #         print(f"  alternative.confidence: {alt.confidence}")
            #         print(f"  alternative has words: {hasattr(alt, 'words')}")
            #         if hasattr(alt, 'words'):
            #             print(f"  words count: {len(alt.words) if alt.words else 0}")
            
            if not transcript:
                return
            
            # Extract speaker information from words (for live streaming)
            speaker_id = None
            speaker_name = None
            
            # Get speaker information from words array (this is where Deepgram puts it for live streaming)
            if hasattr(alternative, 'words') and alternative.words:
                # Debug: Print detailed word-level information (show when diarization enabled or debug flag set)
                # if is_final and (self.config.debug_speaker_data or self.config.diarize):
                #     print(f"🔍 [SPEAKER DEBUG] Found {len(alternative.words)} words with timing:")
                #     for i, word in enumerate(alternative.words):
                #         word_text = getattr(word, 'word', 'NO_WORD')
                #         word_speaker = getattr(word, 'speaker', 'NO_SPEAKER')
                #         word_start = getattr(word, 'start', 'NO_START')
                #         word_end = getattr(word, 'end', 'NO_END')
                #         word_confidence = getattr(word, 'confidence', 'NO_CONF')
                #         print(f"  [{i:2d}] '{word_text}' | Speaker: {word_speaker} | Time: {word_start:.3f}-{word_end:.3f}s | Conf: {word_confidence:.3f}")
                
                # Find the most common speaker in this utterance
                speaker_counts = {}
                for word in alternative.words:
                    if hasattr(word, 'speaker'):
                        speaker = word.speaker
                        speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
                
                if speaker_counts:
                    # Use the speaker that spoke the most words in this utterance
                    speaker_id = max(speaker_counts, key=speaker_counts.get)
                    speaker_name = self.session_speakers.get(speaker_id)
                    # if is_final and (self.config.debug_speaker_data or self.config.diarize):
                    #     print(f"🎯 [SPEAKER DEBUG] Speaker analysis: {speaker_counts} → Primary speaker: {speaker_id}")
                    
                    # Process words for voice fingerprinting
                    if is_final and self.voice_fingerprinter:
                        try:
                            #print(f"🔊 [VOICE FINGERPRINT] Processing {len(alternative.words)} words for fingerprinting")
                            # Convert Deepgram words to our format
                            word_timings = []
                            utterance_start_time = time.time() - (alternative.words[-1].end if alternative.words else 0.0)
                            
                            for word in alternative.words:
                                word_timing = create_word_timing_from_deepgram_word(word)
                                word_timing.utterance_start = utterance_start_time
                                word_timings.append(word_timing)
                            
                            # Process synchronously (called from sync callback context)
                            if word_timings:
                                #print(f"🔊 [VOICE FINGERPRINT] Processing transcript words")
                                self.voice_fingerprinter.process_transcript_words(word_timings, utterance_start_time)
                        except Exception as fp_error:
                            print(f"⚠️  Voice fingerprinting word processing error: {fp_error}")
                    
                    # Check if we have a speaker name from voice fingerprinting
                    if self.voice_fingerprinter and speaker_id is not None:
                        fingerprint_name = self.voice_fingerprinter.get_speaker_name(speaker_id)
                        if fingerprint_name:
                            speaker_name = fingerprint_name
                            self.session_speakers[speaker_id] = fingerprint_name
            
            # Fallback: check metadata (for pre-recorded files)
            if speaker_id is None and hasattr(channel, 'metadata') and channel.metadata:
                speaker_id = getattr(channel.metadata, 'speaker', None)
                if speaker_id is not None:
                    speaker_name = self.session_speakers.get(speaker_id)
            
            # Create STT result
            stt_result = STTResult(
                text=transcript,
                confidence=confidence,
                is_final=is_final,
                speaker_id=speaker_id,
                speaker_name=speaker_name,
                timestamp=datetime.now(),
                raw_data=kwargs
            )
            
            # Apply speaker identification if enabled
            if self.config.enable_speaker_id and self.speaker_identifier:
                stt_result = self._apply_speaker_identification(stt_result)
            
            # Emit appropriate event
            if is_final:
                self._schedule_event(
                    STTEventType.UTTERANCE_COMPLETE, 
                    stt_result
                )
                
                # Check for speaker change
                if speaker_id is not None and speaker_id != self.current_speaker:
                    self.current_speaker = speaker_id
                    self._schedule_event(
                        STTEventType.SPEAKER_CHANGE,
                        {
                            "speaker_id": speaker_id,
                            "speaker_name": speaker_name,
                            "previous_speaker": self.current_speaker
                        }
                    )
                
                self.last_utterance_time = datetime.now()
                
            else:
                self._schedule_event(
                    STTEventType.INTERIM_RESULT,
                    stt_result
                )
                
        except Exception as e:
            self._schedule_event(STTEventType.ERROR, {
                "error": f"Transcript processing error: {e}"
            })
    
    def _on_utterance_end(self, *args, **kwargs):
        """Handle utterance end events."""
        self._schedule_event(STTEventType.SPEECH_ENDED, {
            "timestamp": datetime.now()
        })
    
    def _on_speech_started(self, *args, **kwargs):
        """Handle speech start events."""
        self._schedule_event(STTEventType.SPEECH_STARTED, {
            "timestamp": datetime.now()
        })
    
    def _on_error(self, *args, **kwargs):
        """Handle STT errors."""
        error = kwargs.get('error', 'Unknown error')
        self._schedule_event(STTEventType.ERROR, {
            "error": f"Deepgram error: {error}"
        })
    
    def _on_close(self, *args, **kwargs):
        """Handle connection close."""
        print("🔌 Deepgram connection closed by server")
        self.connection_alive = False
        
        # Cancel keepalive task if running
        if self.keepalive_task and not self.keepalive_task.done():
            self.keepalive_task.cancel()
        
        # Trigger reconnection if not already reconnecting and not shutting down
        if self.is_listening and not self.is_reconnecting:
            print("🔄 Attempting automatic reconnection...")
            asyncio.create_task(self._handle_reconnection())
        else:
            self.is_listening = False
            self._schedule_event(STTEventType.CONNECTION_CLOSED, {
                "message": "Deepgram connection closed by server"
            })
    
    def add_reference_audio(self, audio_data: bytes):
        """Add reference audio for echo cancellation (TTS output)."""
        if self.echo_canceller:
            self.echo_canceller.add_reference_audio(audio_data)
    
    def _setup_speaker_identification(self):
        """Setup speaker identification if enabled."""
        if self.config.speaker_profiles_path:
            try:
                # Import speaker identification module if available
                from improved_speaker_identification import SpeakerIdentifier
                self.speaker_identifier = SpeakerIdentifier(
                    profiles_file=self.config.speaker_profiles_path
                )
                print("✅ Speaker identification enabled")
            except ImportError:
                print("⚠️ Speaker identification module not found")
                self.speaker_identifier = None
    
    def _apply_speaker_identification(self, stt_result: STTResult) -> STTResult:
        """Apply speaker identification to the result."""
        if self.speaker_identifier and stt_result.raw_data:
            try:
                # Extract audio features and identify speaker
                # This would need to be implemented based on your speaker ID system
                identified_name = self.speaker_identifier.identify_speaker(
                    # Pass appropriate audio data
                )
                if identified_name:
                    stt_result.speaker_name = identified_name
            except Exception as e:
                print(f"Speaker identification error: {e}")
        
        return stt_result
    
    async def _emit_event(self, event_type: STTEventType, data: Any):
        """Emit an event to all registered callbacks."""
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    print(f"Callback error for {event_type}: {e}")
    
    def _schedule_event(self, event_type: STTEventType, data: Any):
        """Schedule an event emission from a synchronous context."""
        try:
            # Try to get the current event loop
            current_loop = None
            try:
                current_loop = asyncio.get_running_loop()
            except RuntimeError:
                current_loop = self.event_loop
            
            if current_loop and current_loop.is_running():
                if current_loop == self.event_loop:
                    # We're in the same loop, schedule normally
                    future = asyncio.run_coroutine_threadsafe(
                        self._emit_event(event_type, data), 
                        current_loop
                    )
                    # Don't wait for result to avoid blocking
                else:
                    # Different loop, use stored event loop
                    if self.event_loop and self.event_loop.is_running():
                        asyncio.run_coroutine_threadsafe(
                            self._emit_event(event_type, data), 
                            self.event_loop
                        )
                    else:
                        # Fallback to sync emission
                        self._emit_event_sync(event_type, data)
            else:
                # No running loop, emit synchronously
                self._emit_event_sync(event_type, data)
                
        except Exception as e:
            print(f"Error scheduling event {event_type}: {e}")
            # Final fallback
            self._emit_event_sync(event_type, data)
    
    def _emit_event_sync(self, event_type: STTEventType, data: Any):
        """Emit an event synchronously (fallback for when no event loop is available)."""
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                try:
                    if not asyncio.iscoroutinefunction(callback):
                        callback(data)
                    # Skip async callbacks in sync context
                except Exception as e:
                    print(f"Sync callback error for {event_type}: {e}")
    
    async def cleanup(self):
        """Clean up all resources."""
        await self.stop_listening()
        
        # Clear event loop reference
        self.event_loop = None
        
        # Clean up PyAudio in executor to avoid segfault
        if self.p:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._cleanup_pyaudio)
            except Exception as e:
                print(f"Error terminating PyAudio: {e}")
            finally:
                self.p = None
    
    def _cleanup_pyaudio(self):
        """Clean up PyAudio in a separate thread context."""
        try:
            if self.p:
                self.p.terminate()
        except Exception as e:
            print(f"PyAudio cleanup error: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        # Don't call terminate in destructor - too risky for segfaults
        pass

# Convenience functions for quick usage
async def start_listening(api_key: str, on_utterance: Callable[[STTResult], None]) -> AsyncSTTStreamer:
    """
    Convenience function to start listening with default settings.
    
    Args:
        api_key: Deepgram API key
        on_utterance: Callback for completed utterances
        
    Returns:
        AsyncSTTStreamer: The STT streamer instance
    """
    config = STTConfig(api_key=api_key)
    stt = AsyncSTTStreamer(config)
    stt.on(STTEventType.UTTERANCE_COMPLETE, on_utterance)
    
    success = await stt.start_listening()
    if success:
        return stt
    else:
        await stt.cleanup()
        raise Exception("Failed to start STT")

# Example usage
async def main():
    """Example usage of the STT module."""
    api_key = input("Enter your Deepgram API key: ").strip()
    
    config = STTConfig(        api_key=api_key,
        interim_results=True,
        diarize=True
    )
    stt = AsyncSTTStreamer(config)
    
    # Set up callbacks
    async def on_utterance_complete(result: STTResult):
        speaker_info = f" (Speaker {result.speaker_id})" if result.speaker_id is not None else ""
        print(f"🎯 Final{speaker_info}: {result.text} (confidence: {result.confidence:.2f})")
    
    def on_interim_result(result: STTResult):
        print(f"💭 Interim: {result.text}")
    
    async def on_speech_started(data):
        print("🎤 Speech started")
    
    async def on_speech_ended(data):
        print("🔇 Speech ended")
    
    async def on_error(data):
        print(f"❌ Error: {data['error']}")
    
    # Register callbacks
    stt.on(STTEventType.UTTERANCE_COMPLETE, on_utterance_complete)
    stt.on(STTEventType.INTERIM_RESULT, on_interim_result)
    stt.on(STTEventType.SPEECH_STARTED, on_speech_started)
    stt.on(STTEventType.SPEECH_ENDED, on_speech_ended)
    stt.on(STTEventType.ERROR, on_error)
    
    try:
        print("🎤 Starting STT test...")
        success = await stt.start_listening()
        
        if success:
            print("✅ STT started successfully!")
            print("💡 Speak into your microphone. Press Ctrl+C to stop.")
            
            # Keep running until interrupted
            while stt.is_currently_listening():
                await asyncio.sleep(1)
        else:
            print("❌ Failed to start STT")
            
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    finally:
        await stt.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 