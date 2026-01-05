#!/usr/bin/env python3
"""
Real-time Speech-to-Text with Live Diarization using Deepgram API
"""

import asyncio
import json
import os
import threading
from datetime import datetime
from typing import Dict, Optional

from deepgram import (
    DeepgramClient,
    LiveTranscriptionEvents,
    LiveOptions,
)
from colorama import Fore, Style, init
from dotenv import load_dotenv

try:
    from mel_aec_audio import ensure_stream_started, prepare_capture_chunk, stop_stream
except ImportError as mel_aec_error:  # pragma: no cover - environment specific
    ensure_stream_started = None
    prepare_capture_chunk = None
    stop_stream = None
    MEL_AEC_IMPORT_ERROR = mel_aec_error
else:  # pragma: no cover - trivial assigning
    MEL_AEC_IMPORT_ERROR = None

# Initialize colorama for colored console output
init(autoreset=True)

# Load environment variables
load_dotenv()

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHUNK_SAMPLES = 1600

def _raise_mel_aec_import_error() -> None:
    """Helper to raise a clear error when mel-aec is unavailable."""
    if MEL_AEC_IMPORT_ERROR is not None:
        raise RuntimeError(
            "mel_aec_audio is required for realtime_stt_diarization but could not be imported. "
            "Ensure the mel-aec project is built and on PYTHONPATH."
        ) from MEL_AEC_IMPORT_ERROR

class RealTimeSTTDiarization:
    def __init__(self, api_key: str, config: Optional[object] = None):
        """
        Initialize the real-time STT with diarization system.
        
        Args:
            api_key (str): Deepgram API key
        """
        _raise_mel_aec_import_error()

        self.api_key = api_key
        self.config = config
        self.deepgram = DeepgramClient(api_key)
        self.connection = None
        self.is_running = False
        self.audio_stop_event: Optional[threading.Event] = None
        self.audio_stream = None
        self.audio_sample_rate = DEFAULT_SAMPLE_RATE
        self.audio_chunk_samples = DEFAULT_CHUNK_SAMPLES
        self.model = "nova-3"
        self.language = "en-US"
        self.smart_format = True
        self.interim_results = True
        self.utterance_end_ms = 1000
        self.vad_events = True
        self.punctuate = True
        self.diarize = True
        self.multichannel = False
        self.alternatives = 1
        self.speakers: Dict[int, str] = {}  # Track speakers and their colors
        self.speaker_colors = [
            Fore.CYAN,
            Fore.MAGENTA,
            Fore.YELLOW,
            Fore.GREEN,
            Fore.BLUE,
            Fore.RED,
        ]

        self._audio_callback_registered = False

        if config is not None:
            self._configure_from_config(config)

    def _configure_from_config(self, config: object) -> None:
        """Apply configuration overrides from VoiceAI configuration."""
        stt_cfg = getattr(config, "stt", None)
        if stt_cfg:
            self.model = getattr(stt_cfg, "model", self.model)
            self.language = getattr(stt_cfg, "language", self.language)
            self.smart_format = getattr(stt_cfg, "smart_format", self.smart_format)
            self.interim_results = getattr(stt_cfg, "interim_results", self.interim_results)
            self.diarize = getattr(stt_cfg, "diarize", self.diarize)
            self.punctuate = getattr(stt_cfg, "punctuate", self.punctuate)
            self.utterance_end_ms = getattr(stt_cfg, "utterance_end_ms", self.utterance_end_ms)
            self.vad_events = getattr(stt_cfg, "vad_events", self.vad_events)
            sample_rate = getattr(stt_cfg, "sample_rate", self.audio_sample_rate)
            chunk_size = getattr(stt_cfg, "chunk_size", self.audio_chunk_samples)
            try:
                self.audio_sample_rate = max(8000, int(sample_rate))
            except (TypeError, ValueError):
                self.audio_sample_rate = DEFAULT_SAMPLE_RATE
            try:
                self.audio_chunk_samples = max(320, int(chunk_size))
            except (TypeError, ValueError):
                self.audio_chunk_samples = DEFAULT_CHUNK_SAMPLES
        
    def get_speaker_color(self, speaker_id: int) -> str:
        """Get a consistent color for a speaker."""
        if speaker_id not in self.speakers:
            color_index = len(self.speakers) % len(self.speaker_colors)
            self.speakers[speaker_id] = self.speaker_colors[color_index]
        return self.speakers[speaker_id]
    
    def setup_connection(self):
        """Set up the WebSocket connection to Deepgram."""
        try:
            # Configure live transcription options
            options = LiveOptions(
                model=self.model,
                language=self.language,
                smart_format=self.smart_format,
                interim_results=self.interim_results,
                utterance_end_ms=self.utterance_end_ms,
                vad_events=self.vad_events,
                punctuate=self.punctuate,
                diarize=self.diarize,
                multichannel=self.multichannel,
                alternatives=self.alternatives,
                encoding="linear16",
                sample_rate=self.audio_sample_rate,
                channels=1,
            )
            
            # Create the connection
            self.connection = self.deepgram.listen.websocket.v("1")
            
            # Register event handlers
            self.connection.on(LiveTranscriptionEvents.Open, self.on_open)
            self.connection.on(LiveTranscriptionEvents.Transcript, self.on_transcript)
            self.connection.on(LiveTranscriptionEvents.Metadata, self.on_metadata)
            self.connection.on(LiveTranscriptionEvents.Error, self.on_error)
            self.connection.on(LiveTranscriptionEvents.Close, self.on_close)
            
            # Start the connection
            if not self.connection.start(options):
                print(f"{Fore.RED}Failed to start Deepgram connection")
                return False
                
            return True
            
        except Exception as e:
            print(f"{Fore.RED}Error setting up connection: {str(e)}")
            return False

    def start_audio_stream(self) -> bool:
        """Start streaming audio from mel-aec to Deepgram using the capture callback."""
        if self._audio_callback_registered:
            return True

        if ensure_stream_started is None or prepare_capture_chunk is None:
            print(f"{Fore.RED}mel_aec_audio is not available—cannot start audio stream.")
            return False

        try:
            stream = ensure_stream_started()
        except Exception as exc:
            print(f"{Fore.RED}Error starting mel-aec audio stream: {exc}")
            return False

        self.audio_stream = stream
        self.audio_stop_event = threading.Event()
        self.audio_stop_event.clear()

        stream.set_input_callback(self._handle_audio_capture)
        self._audio_callback_registered = True
        return True

    def _handle_audio_capture(self, audio_data) -> None:
        """Callback invoked by mel-aec when new capture audio is available."""
        if self.audio_stop_event and self.audio_stop_event.is_set():
            return

        if not self.is_running or self.connection is None:
            return

        pcm_bytes = prepare_capture_chunk(audio_data, self.audio_sample_rate)
        if not pcm_bytes:
            return

        try:
            self.connection.send(pcm_bytes)
        except Exception as exc:
            if self.audio_stop_event and not self.audio_stop_event.is_set():
                print(f"{Fore.RED}Error sending audio chunk to Deepgram: {exc}")
                self.audio_stop_event.set()

    def stop_audio_stream(self) -> None:
        """Stop streaming audio from the mel-aec capture callback."""
        if self.audio_stop_event:
            self.audio_stop_event.set()
        self._audio_callback_registered = False
        self.audio_stream = None
        self.audio_stop_event = None
    
    def on_open(self, *args, **kwargs):
        """Handle connection open event."""
        print(f"{Fore.GREEN}✓ Connected to Deepgram")
        print(f"{Fore.GREEN}✓ Live diarization enabled")
        print(f"{Fore.WHITE}{'='*60}")
        print(f"{Fore.WHITE}Start speaking... (Ctrl+C to stop)")
        print(f"{Fore.WHITE}{'='*60}")
    
    def on_transcript(self, *args, **kwargs):
        """Handle transcript events with diarization."""
        try:
            result = kwargs.get("result")
            if result is None:
                return

            if isinstance(result, dict):
                transcript = result.get("transcript", "")
                is_final = result.get("is_final", False)
                channel = result.get("channel")
            else:
                transcript = getattr(result, "transcript", "")
                is_final = getattr(result, "is_final", False)
                channel = getattr(result, "channel", None)

            if isinstance(channel, dict):
                alternatives = channel.get("alternatives", [])
            else:
                alternatives = getattr(channel, "alternatives", []) if channel is not None else []

            if not transcript and alternatives:
                first_alt = alternatives[0]
                if isinstance(first_alt, dict):
                    transcript = first_alt.get("transcript", "")
                else:
                    transcript = getattr(first_alt, "transcript", "") or transcript

            if not transcript:
                return

            timestamp = datetime.now().strftime("%H:%M:%S")
            status = "FINAL" if is_final else "INTERIM"

            def _iter_words(alt):
                if isinstance(alt, dict):
                    return alt.get("words", []) or []
                return getattr(alt, "words", []) or []

            words = _iter_words(alternatives[0]) if alternatives else []
            if words:
                speaker_segments = {}
                for word in words:
                    if isinstance(word, dict):
                        speaker = word.get("speaker", 0)
                        word_text = word.get("word", "")
                    else:
                        speaker = getattr(word, "speaker", 0)
                        word_text = getattr(word, "word", "")

                    if speaker not in speaker_segments:
                        speaker_segments[speaker] = []
                    speaker_segments[speaker].append(word_text)

                for speaker, pieces in speaker_segments.items():
                    text = " ".join(pieces).strip()
                    if not text:
                        continue
                    speaker_color = self.get_speaker_color(speaker)
                    print(
                        f"{Fore.WHITE}[{timestamp}] {speaker_color}Speaker {speaker}: "
                        f"{Style.BRIGHT}{text}{Style.RESET_ALL} {Fore.LIGHTBLACK_EX}({status})"
                    )
            else:
                print(
                    f"{Fore.WHITE}[{timestamp}] {Fore.WHITE}Unknown Speaker: "
                    f"{Style.BRIGHT}{transcript}{Style.RESET_ALL} {Fore.LIGHTBLACK_EX}({status})"
                )

        except Exception as exc:
            print(f"{Fore.RED}Error processing transcript: {exc}")
    
    def on_metadata(self, *args, **kwargs):
        """Handle metadata events."""
        try:
            metadata = kwargs.get('metadata')
            if metadata:
                # Handle metadata properly - it might be an object, not a dict
                request_id = getattr(metadata, 'request_id', None)
                if request_id:
                    print(f"{Fore.LIGHTBLACK_EX}[DEBUG] Request ID: {request_id}")
        except Exception as e:
            print(f"{Fore.RED}Error processing metadata: {str(e)}")
    
    def on_error(self, *args, **kwargs):
        """Handle error events."""
        error = kwargs.get('error')
        print(f"{Fore.RED}❌ Deepgram Error: {error}")
    
    def on_close(self, *args, **kwargs):
        """Handle connection close event."""
        print(f"{Fore.YELLOW}Connection closed")
    
    async def start_streaming(self):
        """Start the real-time streaming process."""
        try:
            print(f"{Fore.CYAN}🎤 Initializing Real-time STT with Live Diarization...")
            
            # Setup connection
            if not self.setup_connection():
                return

            # Give the websocket a brief moment to fully open before streaming audio
            await asyncio.sleep(0.2)

            self.is_running = True

            if not self.start_audio_stream():
                print(f"{Fore.RED}Failed to start audio streaming")
                self.is_running = False
                return

            # Keep the connection alive
            while self.is_running:
                await asyncio.sleep(0.1)

        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Stopping...")
        except Exception as e:
            print(f"{Fore.RED}Error during streaming: {str(e)}")
        finally:
            self.stop_streaming()
    
    def stop_streaming(self):
        """Stop the streaming process."""
        self.is_running = False

        self.stop_audio_stream()

        if self.connection:
            try:
                self.connection.finish()
            except Exception as exc:
                print(f"{Fore.YELLOW}Warning closing Deepgram connection: {exc}")
            finally:
                self.connection = None

        if stop_stream is not None:
            try:
                stop_stream()
            except Exception as exc:
                print(f"{Fore.YELLOW}Warning stopping mel-aec stream: {exc}")
        
        print(f"{Fore.GREEN}✓ Streaming stopped")


def main():
    """Main function to run the real-time STT with diarization."""

    config = None
    try:
        from config_loader import load_config

        config = load_config("config.yaml")
        print(f"{Fore.CYAN}🎛️ Loaded config.yaml for mel-aec stream settings")
    except FileNotFoundError:
        print(f"{Fore.YELLOW}No config.yaml found – using default mel-aec settings")
    except Exception as exc:
        print(f"{Fore.YELLOW}⚠️  Unable to load config.yaml: {exc}")
        print(f"{Fore.YELLOW}    Proceeding with default mel-aec settings")
    
    # Get API key from environment or prompt user
    api_key = os.getenv('DEEPGRAM_API_KEY')
    if not api_key: api_key = config.conversation.deepgram_api_key
    
    if not api_key:
        print(f"{Fore.RED}Please set your DEEPGRAM_API_KEY environment variable")
        print(f"{Fore.WHITE}You can get an API key from: https://console.deepgram.com/")
        print(f"{Fore.WHITE}Then run: export DEEPGRAM_API_KEY='your_api_key_here'")
        return
    
    # Create and run the STT system
    stt_system = RealTimeSTTDiarization(api_key, config=config)
    
    try:
        asyncio.run(stt_system.start_streaming())
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Exiting...")
    except Exception as e:
        print(f"{Fore.RED}Fatal error: {str(e)}")


if __name__ == "__main__":
    main() 
