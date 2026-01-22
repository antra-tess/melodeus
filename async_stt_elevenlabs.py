#!/usr/bin/env python3
"""
Async STT Module for ElevenLabs WebSocket Streaming (Scribe)
Alternative to Deepgram with same callback interface.
"""

import asyncio
import queue
import traceback
import base64
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
from datetime import datetime

import websockets
import numpy as np

from mel_aec_audio import ensure_stream_started, prepare_capture_chunk

# Import shared types from the main STT module
from async_stt_module import STTEventType, STTResult, STTConfig

# Try to import TitaNet for speaker identification
try:
    from titanet_voice_fingerprinting import (
        TitaNetVoiceFingerprinter,
        WordTiming
    )
    TITANET_AVAILABLE = True
except ImportError:
    TITANET_AVAILABLE = False

# Try to import noise gate
try:
    from audio_gate import NoiseGate, GateConfig, create_gate_from_config
    GATE_AVAILABLE = True
except ImportError:
    GATE_AVAILABLE = False


class AsyncSTTElevenLabs:
    """Async STT Streamer using ElevenLabs Scribe API with same interface as Deepgram module."""
    
    # ElevenLabs WebSocket endpoint
    WS_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
    
    def __init__(self, config: STTConfig, speakers_config=None):
        self.config = config
        self.speakers_config = speakers_config
        
        # Audio queue and state
        self.audio_queue = queue.Queue(maxsize=1280)
        self.listening = False
        self.num_audio_frames_received = 0
        
        # Set up audio input callback
        stream = ensure_stream_started()
        stream.set_input_callback(self._handle_audio_capture)
        
        # Tasks
        self.audio_task = None
        self.websocket_task = None
        
        # Turn tracking (similar to Deepgram module)
        self.current_partial_text = ""
        self.last_committed_text = ""
        self.current_turn_id = None
        self.last_sent_message_uuid = None
        self.turn_history: List[Tuple[float, str]] = []  # (timestamp, text) for autosend logic
        self.current_turn_autosent_transcript = None
        
        # Audio window tracking for speaker identification
        self.audio_window_start = 0
        self.audio_window_end = 0
        
        # Callbacks
        self.callbacks: Dict[STTEventType, List[Callable]] = {
            event_type: [] for event_type in STTEventType
        }
        
        # Audio level metering callback (for UI visualization)
        self.input_level_callback: Optional[Callable[[float], None]] = None
        self._level_update_counter = 0  # Throttle level updates
        
        # TitaNet voice fingerprinting (optional)
        self.voice_fingerprinter = None
        if config.enable_speaker_id and TITANET_AVAILABLE:
            try:
                debug_save_audio = getattr(config, 'debug_speaker_data', False)
                save_user_audio = getattr(config, 'save_user_audio', False)
                self.voice_fingerprinter = TitaNetVoiceFingerprinter(
                    speakers_config,
                    debug_save_audio=debug_save_audio,
                    save_user_audio=save_user_audio
                )
                print(f"🤖 TitaNet voice fingerprinting enabled (ElevenLabs STT)")
            except Exception as e:
                print(f"⚠️  TitaNet voice fingerprinting failed to initialize: {e}")
        
        # Noise gate (optional) - configured via gate_config in STTConfig
        self.noise_gate = None
        gate_config = getattr(config, 'gate_config', None)
        if gate_config and GATE_AVAILABLE:
            try:
                self.noise_gate = create_gate_from_config(gate_config, config.sample_rate)
                print(f"🚪 Noise gate enabled: threshold={gate_config.get('threshold_db', -40)}dB, "
                      f"attack={gate_config.get('attack_ms', 1)}ms, "
                      f"hold={gate_config.get('hold_ms', 100)}ms, "
                      f"release={gate_config.get('release_ms', 50)}ms")
            except Exception as e:
                print(f"⚠️  Noise gate failed to initialize: {e}")
    
    def on(self, event_type: STTEventType, callback: Callable):
        """Register a callback for an event type."""
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        self.callbacks[event_type].append(callback)
    
    def off(self, event_type: STTEventType, callback: Callable):
        """Unregister a callback for an event type."""
        if event_type in self.callbacks and callback in self.callbacks[event_type]:
            self.callbacks[event_type].remove(callback)
    
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
                    print(f"⚠️  Callback error for {event_type}: {e}")
                    traceback.print_exc()
    
    def _build_ws_url(self) -> str:
        """Build WebSocket URL with query parameters."""
        params = {
            "model_id": "scribe_v2_realtime",  # ElevenLabs Scribe v2 Realtime (~150ms latency)
            "audio_format": f"pcm_{self.config.sample_rate}",
            "commit_strategy": "vad",  # Use VAD for automatic turn detection
            "include_timestamps": "true",  # Get word-level timestamps
            "include_language_detection": "true",  # Auto-detect language (can switch mid-conversation)
            "vad_silence_threshold_secs": str(self.config.utterance_end_ms / 1000),
            "vad_threshold": "0.4",
            "min_speech_duration_ms": "100",
            "min_silence_duration_ms": "100",
            "enable_logging": "true",
        }
        
        if self.config.language and self.config.language != "en-US":
            # Convert en-US to en format if needed
            lang = self.config.language.split("-")[0] if "-" in self.config.language else self.config.language
            params["language_code"] = lang
        
        # Build query string
        query_parts = [f"{k}={v}" for k, v in params.items()]
        
        # Add keyterms for custom vocabulary (up to 100 terms, max 50 chars each)
        # ElevenLabs uses context-aware keyterm prompting for better recognition
        if self.config.keywords:
            keyterms = []
            for kw in self.config.keywords:
                # Handle both dict format {"word": "...", "weight": ...} and simple strings
                if isinstance(kw, dict):
                    word = kw.get("word", "")
                elif isinstance(kw, tuple):
                    word = kw[0]
                else:
                    word = str(kw)
                
                if word and len(word) <= 50:
                    keyterms.append(word)
            
            # Add each keyterm as a separate parameter (up to 100)
            for term in keyterms[:100]:
                query_parts.append(f"keyterms={term}")
            
            if keyterms:
                print(f"🔤 ElevenLabs keyterms: {keyterms[:10]}{'...' if len(keyterms) > 10 else ''}")
        
        return f"{self.WS_URL}?{'&'.join(query_parts)}"
    
    def _handle_audio_capture(self, audio_data) -> None:
        """Handle incoming audio from the microphone."""
        try:
            if self.listening:
                self.audio_queue.put_nowait(audio_data)
        except queue.Full:
            pass  # Drop frames if queue is full
        except Exception as e:
            print(f"⚠️  Failed to queue audio: {e}")
    
    def clear_audio_queue(self):
        """Clear any pending audio in the queue."""
        while True:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
    
    def _has_meaningful_change(self, a: Optional[str], b: Optional[str]) -> bool:
        """Check if two transcripts have meaningful differences (ignoring punctuation)."""
        if a is None or b is None:
            return True
        
        # Strip punctuation and whitespace for comparison
        import re
        def clean(s):
            return re.sub(r'[^\w\s]', '', s).lower().strip()
        
        return clean(a) != clean(b)
    
    async def _handle_message(self, message: str):
        """Handle incoming WebSocket message from ElevenLabs."""
        try:
            data = json.loads(message)
            msg_type = data.get("message_type", "")
            
            if msg_type == "session_started":
                print(f"🔗 ElevenLabs STT session started: {data.get('session_id', 'unknown')}")
                await self._emit_event(STTEventType.CONNECTION_OPENED, {
                    "session_id": data.get("session_id"),
                    "config": data.get("config", {})
                })
            
            elif msg_type == "partial_transcript":
                text = data.get("text", "")
                if text:
                    self.current_partial_text = text
                    self.turn_history.append((time.time(), text))
                    
                    # Emit interim result
                    stt_result = STTResult(
                        text=text,
                        confidence=0.9,  # ElevenLabs doesn't provide confidence for partials
                        is_final=False,
                        is_edit=False,
                        message_id=None,
                        speaker_id=None,
                        speaker_name=None,
                        timestamp=datetime.now()
                    )
                    await self._emit_event(STTEventType.INTERIM_RESULT, stt_result)
                    
                    # Auto-send logic (similar to Deepgram module)
                    await self._check_autosend(text)
            
            elif msg_type == "committed_transcript":
                text = data.get("text", "")
                if text and self._has_meaningful_change(text, self.last_committed_text):
                    await self._emit_final_result(text, data)
                    self.last_committed_text = text
                    self._reset_turn_state()
            
            elif msg_type == "committed_transcript_with_timestamps":
                text = data.get("text", "")
                words = data.get("words", [])
                language = data.get("language_code")
                
                if language:
                    # Log detected language (useful for multilingual conversations)
                    print(f"🌐 Detected language: {language}")
                
                if text and self._has_meaningful_change(text, self.last_committed_text):
                    await self._emit_final_result(text, data, words=words)
                    self.last_committed_text = text
                    self._reset_turn_state()
            
            elif msg_type in ("error", "auth_error", "quota_exceeded", "throttled", 
                             "rate_limited", "transcriber_error"):
                error_msg = data.get("error", "Unknown error")
                print(f"❌ ElevenLabs STT error ({msg_type}): {error_msg}")
                await self._emit_event(STTEventType.ERROR, {
                    "error": error_msg,
                    "type": msg_type
                })
            
        except json.JSONDecodeError as e:
            print(f"⚠️  Failed to parse ElevenLabs message: {e}")
        except Exception as e:
            print(f"⚠️  Error handling ElevenLabs message: {e}")
            traceback.print_exc()
    
    async def _check_autosend(self, text: str):
        """Check if we should auto-send the transcript (similar to Deepgram module)."""
        if not text or not self._has_meaningful_change(text, self.current_turn_autosent_transcript):
            return
        
        ms_until_autosend = 750
        
        # Find oldest time with same transcript
        oldest_time_still_same = time.time()
        for old_time, old_text in reversed(self.turn_history):
            if old_text != text:
                break
            oldest_time_still_same = old_time
        
        # If text has been stable for long enough, auto-send
        if (time.time() - oldest_time_still_same) * 1000 > ms_until_autosend:
            is_edit = self.current_turn_autosent_transcript is not None
            self.current_turn_autosent_transcript = text
            
            await self._emit_final_result(
                text, 
                {"autosend": True}, 
                is_edit=is_edit,
                is_autosend=True
            )
    
    async def _emit_final_result(self, text: str, raw_data: Dict, 
                                  words: Optional[List] = None,
                                  is_edit: bool = False,
                                  is_autosend: bool = False):
        """Emit a final/committed transcript result."""
        # Determine speaker using TitaNet if available
        user_tag = "User"
        if self.voice_fingerprinter and words:
            # Extract speaker from word-level data if available
            speaker_ids = set()
            for word in words:
                if word.get("speaker_id"):
                    speaker_ids.add(word["speaker_id"])
            
            # If ElevenLabs provided speaker IDs, use them
            if speaker_ids:
                user_tag = f"Speaker {list(speaker_ids)[0]}"
            else:
                # Fall back to TitaNet
                # Convert frame counts to seconds for WordTiming
                start_seconds = self.audio_window_start / self.config.sample_rate
                end_seconds = self.audio_window_end / self.config.sample_rate
                word_timing = WordTiming(
                    word=text,
                    speaker_id="speaker_0",
                    start_time=start_seconds,
                    end_time=end_seconds,
                    confidence=1.0
                )
                user_tag = self.voice_fingerprinter.process_transcript_words(
                    word_timings=[word_timing],
                    sample_rate=self.config.sample_rate
                )
        elif self.voice_fingerprinter:
            # Convert frame counts to seconds for WordTiming
            start_seconds = self.audio_window_start / self.config.sample_rate
            end_seconds = self.audio_window_end / self.config.sample_rate
            word_timing = WordTiming(
                word=text,
                speaker_id="speaker_0",
                start_time=start_seconds,
                end_time=end_seconds,
                confidence=1.0
            )
            user_tag = self.voice_fingerprinter.process_transcript_words(
                word_timings=[word_timing],
                sample_rate=self.config.sample_rate
            )
        
        # Generate or reuse message ID
        if is_edit and self.last_sent_message_uuid:
            message_uuid = self.last_sent_message_uuid
        else:
            message_uuid = str(uuid.uuid4())
            self.last_sent_message_uuid = message_uuid
        
        stt_result = STTResult(
            text=text,
            confidence=1.0,
            is_final=True,
            is_edit=is_edit,
            message_id=message_uuid,
            speaker_id=None,
            speaker_name=user_tag,
            timestamp=datetime.now(),
            raw_data=raw_data,
            audio_window_start=self.audio_window_start,
            audio_window_end=self.audio_window_end
        )
        
        await self._emit_event(STTEventType.UTTERANCE_COMPLETE, stt_result)
    
    def _reset_turn_state(self):
        """Reset state for a new turn."""
        self.current_partial_text = ""
        self.turn_history = []
        self.current_turn_autosent_transcript = None
        self.last_sent_message_uuid = None
        # Update audio window for next turn
        self.audio_window_start = self.audio_window_end
    
    async def _send_audio_loop(self, websocket):
        """Send audio chunks to ElevenLabs via WebSocket."""
        try:
            self.listening = False
            self.clear_audio_queue()
            self.listening = True
            self._reset_turn_state()
            self.audio_window_start = 0
            self.audio_window_end = 0
            
            while True:
                # Check queue size
                waiting_amount = self.audio_queue.qsize()
                if waiting_amount > 500:
                    print(f"⚠️  {waiting_amount} audio chunks in queue, falling behind")
                
                try:
                    audio_data = self.audio_queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.05)
                    continue
                
                # Prepare PCM audio
                pcm_bytes = prepare_capture_chunk(audio_data, self.config.sample_rate)
                
                # Convert to float for processing
                audio_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Apply noise gate if enabled
                if self.noise_gate is not None:
                    audio_np = self.noise_gate.process(audio_np)
                
                # Calculate input level for UI metering (throttled to ~30 updates/sec)
                self._level_update_counter += 1
                if self.input_level_callback and self._level_update_counter >= 10:
                    self._level_update_counter = 0
                    rms = np.sqrt(np.mean(audio_np ** 2))
                    level = min(1.0, rms / 0.15)  # Normalize to 0-1
                    try:
                        self.input_level_callback(level)
                    except Exception:
                        pass  # Don't let callback errors affect STT
                
                # Feed to TitaNet if enabled
                # Note: audio_np is already resampled to config.sample_rate by prepare_capture_chunk
                if self.voice_fingerprinter is not None:
                    self.voice_fingerprinter.add_audio_chunk(
                        audio_np,
                        self.num_audio_frames_received,
                        self.config.sample_rate,  # Audio is resampled to this rate
                    )
                self.num_audio_frames_received += len(audio_np)
                
                # Update audio window
                self.audio_window_end = self.num_audio_frames_received
                
                # Convert back to PCM bytes (after gate processing)
                gated_pcm = (audio_np * 32768.0).astype(np.int16).tobytes()
                
                # Encode audio as base64 for ElevenLabs
                audio_b64 = base64.b64encode(gated_pcm).decode('utf-8')
                
                # Send to ElevenLabs
                message = {
                    "message_type": "input_audio_chunk",
                    "audio_base_64": audio_b64,
                    "commit": False,  # Let VAD handle commits
                    "sample_rate": self.config.sample_rate
                }
                
                await websocket.send(json.dumps(message))
                
        except asyncio.CancelledError:
            print("🔊 Audio send task cancelled")
            raise
        except websockets.exceptions.ConnectionClosed:
            print("🔊 Audio task ended - WebSocket closed")
        except Exception as e:
            print(f"❌ Error in audio send loop: {e}")
            traceback.print_exc()
            raise
        finally:
            self.listening = False
    
    async def _receive_loop(self, websocket):
        """Receive and process messages from ElevenLabs."""
        try:
            async for message in websocket:
                await self._handle_message(message)
        except asyncio.CancelledError:
            raise
        except websockets.exceptions.ConnectionClosed:
            print("🔗 ElevenLabs WebSocket connection closed")
        except Exception as e:
            print(f"❌ Error in receive loop: {e}")
            traceback.print_exc()
            raise
    
    async def _websocket_task(self):
        """Main WebSocket task - connects and manages send/receive loops."""
        while True:
            try:
                ws_url = self._build_ws_url()
                headers = {
                    "xi-api-key": self.config.api_key
                }
                
                print(f"🔗 Connecting to ElevenLabs STT...")
                
                async with websockets.connect(
                    ws_url,
                    additional_headers=headers,
                    ping_interval=20,
                    ping_timeout=10
                ) as websocket:
                    print("✅ ElevenLabs STT connected")
                    
                    # Reset frame counter and TitaNet buffer
                    self.num_audio_frames_received = 0
                    if self.voice_fingerprinter is not None:
                        self.voice_fingerprinter.clear_audio_buffer()
                    
                    # Run send and receive concurrently
                    send_task = asyncio.create_task(self._send_audio_loop(websocket))
                    receive_task = asyncio.create_task(self._receive_loop(websocket))
                    
                    try:
                        # Wait for either task to complete (usually due to error or cancellation)
                        done, pending = await asyncio.wait(
                            [send_task, receive_task],
                            return_when=asyncio.FIRST_COMPLETED
                        )
                        
                        # Cancel remaining tasks
                        for task in pending:
                            task.cancel()
                            try:
                                await task
                            except asyncio.CancelledError:
                                pass
                        
                        # Check for exceptions
                        for task in done:
                            if task.exception():
                                raise task.exception()
                                
                    except asyncio.CancelledError:
                        send_task.cancel()
                        receive_task.cancel()
                        try:
                            await send_task
                        except asyncio.CancelledError:
                            pass
                        try:
                            await receive_task
                        except asyncio.CancelledError:
                            pass
                        raise
                    
            except asyncio.CancelledError:
                print("🔗 ElevenLabs STT task cancelled")
                raise
            except websockets.exceptions.ConnectionClosedError as e:
                print(f"⚠️  ElevenLabs connection closed: {e}. Reconnecting...")
                await asyncio.sleep(1)
            except Exception as e:
                print(f"❌ ElevenLabs STT error: {e}")
                traceback.print_exc()
                await asyncio.sleep(2)  # Wait before reconnecting
    
    async def start_listening(self) -> bool:
        """Start the STT stream."""
        try:
            # Cancel existing task if running
            if self.websocket_task:
                self.websocket_task.cancel()
                try:
                    await self.websocket_task
                except asyncio.CancelledError:
                    pass
            
            # Start new WebSocket task
            self.websocket_task = asyncio.create_task(self._websocket_task())
            return True
            
        except Exception as e:
            print(f"❌ Failed to start ElevenLabs STT: {e}")
            return False
    
    async def stop_listening(self) -> bool:
        """Stop the STT stream."""
        try:
            if self.websocket_task:
                self.websocket_task.cancel()
                try:
                    await self.websocket_task
                except asyncio.CancelledError:
                    pass
                self.websocket_task = None
            
            self.listening = False
            return True
            
        except Exception as e:
            print(f"❌ Error stopping ElevenLabs STT: {e}")
            return False
    
    async def pause(self):
        """Pause STT (same as stop_listening)."""
        await self.stop_listening()
    
    async def resume(self):
        """Resume STT (same as start_listening)."""
        await self.start_listening()
    
    async def cleanup(self):
        """Clean up resources."""
        await self.stop_listening()
        self.clear_audio_queue()


# Factory function to create the appropriate STT provider
def create_stt_provider(config: STTConfig, speakers_config=None, provider: str = "deepgram"):
    """
    Create an STT provider instance.
    
    Args:
        config: STT configuration
        speakers_config: Speaker profiles for identification
        provider: "deepgram" or "elevenlabs"
    
    Returns:
        STT streamer instance with unified interface
    """
    provider = provider.lower()
    
    if provider == "elevenlabs":
        print("🎤 Using ElevenLabs Scribe v2 Realtime for STT (~150ms latency)")
        return AsyncSTTElevenLabs(config, speakers_config)
    else:
        # Default to Deepgram
        from async_stt_module import AsyncSTTStreamer
        print("🎤 Using Deepgram for STT")
        return AsyncSTTStreamer(config, speakers_config)
