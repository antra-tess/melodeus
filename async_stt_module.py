
import asyncio
import queue
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable
from enum import Enum
from dataclasses import field
import websockets
import time
import uuid

import numpy as np
import json
from datetime import datetime

from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType
from deepgram.extensions.types.sockets import (
    ListenV2ConnectedEvent,
    ListenV2FatalErrorEvent,
    ListenV2TurnInfoEvent,
)

from mel_aec_audio import ensure_stream_started, prepare_capture_chunk
from titanet_voice_fingerprinting import (
    TitaNetVoiceFingerprinter,
    WordTiming
)

from async_tts_module import _collapse

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
    is_edit: bool
    message_id: str
    speaker_id: Optional[int] = None
    speaker_name: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    raw_data: Optional[Dict[str, Any]] = None

@dataclass
class STTConfig:
    """Configuration for STT settings."""
    api_key: str
    model: str = "flux-general-en"
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
    enable_speaker_id: bool = True
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
        # Deepgram client
        self.deepgram = AsyncDeepgramClient(api_key=config.api_key)
        self.audio_queue = queue.Queue(maxsize=1280)
        self.listening = False
        self.num_audio_frames_recieved = 0
        stream = ensure_stream_started()
        stream.set_input_callback(self._handle_audio_capture)
        self.audio_task = None
        self.websocket_task = None
        self._turn_transcripts: Dict[int, str] = {}
        self.current_turn_history = []
        self.current_turn_autosent_transcript = None
        self.last_sent_message_uuid = None


        self.callbacks: Dict[STTEventType, List[Callable]] = {
            event_type: [] for event_type in STTEventType
        }

        # TitaNet voice fingerprinting (optional)
        self.voice_fingerprinter = None
        if config.enable_speaker_id:
            try:
                # Enable debug audio saving if debug_speaker_data is enabled
                debug_save_audio = getattr(config, 'debug_speaker_data', False)
                self.voice_fingerprinter = TitaNetVoiceFingerprinter(speakers_config, debug_save_audio=debug_save_audio)
                print(f"🤖 TitaNet voice fingerprinting enabled")
                if debug_save_audio:
                    print(f"🐛 Debug audio saving enabled - extracted segments will be saved to debug_audio_segments/")
            except Exception as e:
                print(f"⚠️  TitaNet voice fingerprinting failed to initialize: {e}")
        

    def on(self, event_type: STTEventType, callback: Callable):
        """Register a callback for an event type."""
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        self.callbacks[event_type].append(callback)
    
    def off(self, event_type: STTEventType, callback: Callable):
        """Unregister a callback for an event type."""
        if event_type in self.callbacks and callback in self.callbacks[event_type]:
            self.callbacks[event_type].remove(callback)

    def _build_listen_params(self) -> Dict[str, Any]:
        """Prepare query parameters for Deepgram Listen v2 connection."""
        model = (self.config.model or "").strip()
        if not model:
            model = "flux-general-en"

        # Gracefully fall back if a legacy v1 model is configured
        legacy_models = {
            "nova-2": "flux-general-en",
            "nova-3": "flux-general-en",
            "nova-2-general": "flux-general-en",
            "nova-3-general": "flux-general-en",
        }
        if model in legacy_models:
            fallback = legacy_models[model]
            print(f"⚠️  Listen v2 does not support model '{model}'. Falling back to '{fallback}'.")
            model = fallback

        params: Dict[str, Any] = {
            "model": model,
            "encoding": "linear16",
            "sample_rate": str(self.config.sample_rate),
        }

        # Only nova/flux models accept keyterm; v2 ignores legacy keywords.
        if self.config.keywords:
            keyterms = [word for word, _ in self.config.keywords if word]
            if keyterms:
                params["keyterm"] = " ".join(keyterms)
        params["keyterm"] = "Haiku Sonnet Claude Opus"
        return params
    

    def _on_open(self, *args, **kwargs):
        """Handle Deepgram connection open."""
        print("🔗 Deepgram STT connection opened")
    
    async def _handle_socket_message(self, message: Any):
        """Dispatch Deepgram websocket messages to the appropriate handlers."""
        try:
            if isinstance(message, ListenV2TurnInfoEvent):
                await self._on_turn_info(message)
            elif isinstance(message, ListenV2FatalErrorEvent):
                print(f"❌ Deepgram fatal error ({message.code}): {message.description}")
            elif isinstance(message, ListenV2ConnectedEvent):
                # Already handled by EventType.OPEN; nothing additional needed.
                pass
        except Exception as dispatch_error:
            print(f"⚠️  Failed to process Deepgram message:")
            print(traceback.print_exc())

    def _handle_socket_error(self, error: Exception):
        """Handle errors emitted by the Deepgram websocket client."""
        error_message = str(error)
        print(f"⚠️ Deepgram websocket error")
        print(error_message)

    async def _on_speech_started(self):
        await self._emit_event(STTEventType.SPEECH_STARTED, {
            "timestamp": datetime.now()
        })

    def has_meaningful_change(self, a, b):
        if a is None or b is None: return True
        ignore=set("*\n\r\t #@\\/.,?!-+[]()&%$:")
        # only send edited if it actually changed content (we don't care about punctuation diffs)
        a_cleaned, _ = _collapse(a, ignore)
        b_cleaned, _ = _collapse(b, ignore)
        return a_cleaned.lower() != b_cleaned.lower()

    async def _on_turn_info(self, message: ListenV2TurnInfoEvent):
        event_type = (message.event or "").lower()
        turn_idx = message.turn_index
        transcript = message.transcript or ""
        if event_type == "startofturn":
            await self._on_speech_started()

        send_message = False
        edit_message = False
        # code to end turn earlier than deepgram decides for decreased latency
        if turn_idx != self.prev_turn_idx:
            self.current_turn_history = []    
        if len(transcript) > 0:
            self.current_turn_history.append((time.time(), transcript))
        if turn_idx == self.prev_turn_idx and len(transcript) > 0 and self.has_meaningful_change(transcript, self.current_turn_autosent_transcript):
            ms_until_autosend = 750
            # if we took longer than that and still the same, autosend
            oldest_time_still_same = time.time()
            for old_time, old_transcript in self.current_turn_history[::-1]:
                if old_transcript != transcript:
                    break
                else:
                    oldest_time_still_same = old_time
            if (time.time() - oldest_time_still_same)*1000 > ms_until_autosend:
                #print("Autosend")
                if self.current_turn_autosent_transcript is not None:
                    edit_message = True
                else:
                    send_message = True
                self.current_turn_autosent_transcript = transcript
         
        # If we are on a new turn, send old turn
        if turn_idx != self.prev_turn_idx and self.prev_turn_idx != None and self.prev_transcript:
            if self.current_turn_autosent_transcript is not None:
                if self.has_meaningful_change(self.current_turn_autosent_transcript, self.prev_transcript):
                    edit_message = True
            else:
                send_message = True
            
        if send_message or edit_message:
            if self.voice_fingerprinter:
                word_timing = WordTiming(
                    word=self.prev_transcript,
                    speaker_id=f"speaker {turn_idx}", # v2 doesn't have diarization yet
                    start_time=self.prev_audio_window_start,
                    end_time=self.prev_audio_window_end,
                    confidence=1)
                user_tag = self.voice_fingerprinter.process_transcript_words(word_timings=[word_timing], sample_rate=self.config.sample_rate)
            else:
                user_tag = "User"
            message_uuid = str(uuid.uuid4()) if send_message else self.last_sent_message_uuid
            self.last_sent_message_uuid = message_uuid
            #print(("send" if send_message else "edit"), "with data")
            #print(self.prev_transcript)
            stt_result = STTResult(
                text = self.prev_transcript,
                confidence = 1.0,
                is_final = True,
                is_edit = edit_message,
                speaker_id = None,
                speaker_name = user_tag,
                timestamp = datetime.now(),
                message_id = message_uuid,
            )
            await self._emit_event(
                STTEventType.UTTERANCE_COMPLETE,
                stt_result
            )
        if turn_idx != self.prev_turn_idx:
            self.last_sent_message_uuid = None
            self.current_turn_autosent_transcript = None
            

        self.prev_turn_idx = turn_idx
        self.prev_transcript = transcript
        self.prev_audio_window_start = message.audio_window_start
        self.prev_audio_window_end = message.audio_window_end

        if transcript and not send_message and not edit_message and self.last_sent_message_uuid is None:
            self.most_recent_turn_text = transcript
            
            stt_result = STTResult(
                text = transcript,
                confidence = 1.0,
                is_final = False,
                speaker_id = None,
                speaker_name =  None,
                timestamp = datetime.now(),
                message_id = None,
                is_edit = False
            ) 

            await self._emit_event(
                STTEventType.INTERIM_RESULT,
                stt_result
            )
            #print(f"Turn {turn_idx} {transcript}")
        else:
            #print(f"Turn {turn_idx} Dummy turn")
            pass
        #if event_type in {"turn_end", "speech_ended", "speech_end"}:
        #    self._on_utterance_end(message)

    async def _emit_event(self, event_type: STTEventType, data: Any):
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    print(f"Sync callback error for {event_type}")
                    print(traceback.print_exc())

    def _on_utterance_end(self, message: ListenV2TurnInfoEvent):
        pass

    def _on_close(self, *args, **kwargs):
        """Handle Deepgram connection open."""
        print("🔗 Deepgram STT connection closed")
    

    def _handle_audio_capture(self, audio_data) -> None:
        try:
            if self.listening:
                self.audio_queue.put_nowait(audio_data)
        except Exception as prep_error:
            print(f"⚠️  Failed to prepare capture chunk")
            print(traceback.print_exc())
    
    async def create_audio_task(self, connection):
        try:
            self.listening = False
            self.clear_audio_queue()
            self.listening = True
            self.prev_turn_idx = None
            self.prev_transcript = ""
            self.prev_audio_window_start = 0
            self.prev_audio_window_end = 0
            self.current_turn_history = []
            self.current_turn_autosent_transcript = None
            self.last_sent_message_uuid = None
            while True:
                waitingAmount = self.audio_queue.qsize()
                if waitingAmount > 500:
                    print(f"Warning: {waitingAmount} audio in queue, we are falling behind")
                try:
                    audio_data = self.audio_queue.get_nowait()
                except queue.Empty:
                    # poll
                    await asyncio.sleep(0.05)
                    continue
                
                pcm_bytes = prepare_capture_chunk(audio_data, self.config.sample_rate)
                if self.voice_fingerprinter is not None:
                    audio_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    self.voice_fingerprinter.add_audio_chunk(
                        audio_np,
                        self.num_audio_frames_recieved,
                        self.config.sample_rate,
                    )
                    # increment frame count
                    self.num_audio_frames_recieved += len(audio_np)
                
                await connection.send_media(pcm_bytes)
        except asyncio.CancelledError:
            print("Audio task canceled")
            raise
        except websockets.exceptions.ConnectionClosedOK:
            print("Audio task canceled due to websocket being closed")
            pass
        except:
            print(f"Error in audio task")
            print(traceback.print_exc())
            raise
        finally:
            self.listening = False
            self.audio_task = None

    async def cancel_task(self, task):
        if not task is None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass # intentional
            except:
                print(f"Error in task cancel")
                print(traceback.print_exc())

    async def create_websocket_task(self):
        try:
            while True:
                params = self._build_listen_params()
                async with asyncio.TaskGroup() as tg: # this audio awaits for it to cancel/finish when we try to exit context
                    self.num_audio_frames_recieved = 0
                    async with self.deepgram.listen.v2.connect(
                        model="flux-general-en",
                        encoding="linear16",
                        sample_rate="16000",
                        eot_timeout_ms="500",
                        keyterm=params['keyterm']) as connection:
                        connection.on(EventType.OPEN, self._on_open)
                        connection.on(EventType.MESSAGE, self._handle_socket_message)
                        connection.on(EventType.ERROR, self._handle_socket_error)
                        connection.on(EventType.CLOSE, self._on_close)
                        # create audio task (leaving the with auto-cancels/awaits)
                        audio_task = tg.create_task(self.create_audio_task(connection))
                        await connection.start_listening()
                        # important to cancel and wait for it to return before closing connection by exiting contextc
                        audio_task.cancel()
                        await audio_task
        except asyncio.CancelledError:
            print("Websocket task canceled")
            raise
        except:
            print(traceback.print_exc())
            print("Error in websocket task")
            raise
        finally:
            self.websocket_task = None

    def clear_audio_queue(self):
        while True:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    async def pause(self):
        await self.stop_listening()
    
    async def resume(self):
        await self.start_listening()
    
    async def start_listening(self):
        await self.cancel_task(self.websocket_task) # cancel listening task if already exists
        self.websocket_task = asyncio.create_task(self.create_websocket_task())
        return True

    async def stop_listening(self):
        await self.cancel_task(self.websocket_task)
        return True
    
    async def cleanup(self):
        await self.stop_listening()
