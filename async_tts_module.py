#!/usr/bin/env python3
"""
Async TTS Module for ElevenLabs WebSocket Streaming
Provides interruptible text-to-speech with real-time audio playback.
Now uses ElevenLabs alignment data to track spoken content.
"""

import asyncio
import websockets
import json
import base64
import time
import re
import difflib
import bisect
from collections import defaultdict
import threading
import traceback
import queue
from typing import Optional, AsyncGenerator, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass, field
import numpy as np
import scipy.io.wavfile as wavfile
import os

try:
    from mel_aec_audio import (
        ensure_stream_started,
        shared_sample_rate,
        write_playback_pcm,
        interrupt_playback,
        int16_bytes_to_float,
        _resample
    )
except ImportError:  # pragma: no cover - fallback when running from repo root
    import os
    import sys

    CURRENT_DIR = os.path.dirname(__file__)
    if CURRENT_DIR and CURRENT_DIR not in sys.path:
        sys.path.insert(0, CURRENT_DIR)
    from mel_aec_audio import (
        ensure_stream_started,
        shared_sample_rate,
        write_playback_pcm,
        interrupt_playback,
    )

@dataclass
class SpokenContent:
    """Represents content that was actually spoken by TTS."""
    text: str
    start_time: float
    end_time: float
    confidence: float = 1.0


@dataclass
class AlignmentChar:
    """Represents a single character alignment with its timing."""
    char: str
    start_time: float  # Seconds since audio playback start
    end_time: float    # Seconds since audio playback start
    global_index: int


@dataclass
class ToolCall:
    """Represents a tool call embedded in the text."""
    tag_name: str  # e.g., "function", "search", etc.
    content: str  # The full XML content including tags
    start_position: int  # Character position in generated_text where tool should execute
    end_position: int  # Character position where tool content ends
    executed: bool = False

@dataclass
class ToolResult:
    """Result from tool execution."""
    should_interrupt: bool  # Whether to interrupt current speech
    content: Optional[str] = None  # Optional content to insert into conversation/speak
    

@dataclass
class TTSConfig:
    """Configuration for TTS settings."""
    api_key: str
    voice_id: str = "T2KZm9rWPG5TgXTyjt7E"  # Catalyst voice
    model_id: str = "eleven_multilingual_v2"
    output_format: str = "pcm_22050"
    sample_rate: int = 22050
    speed: float = 1.0
    stability: float = 0.5
    similarity_boost: float = 0.8
    chunk_size: int = 1024
    buffer_size: int = 2048
    # Multi-voice support
    emotive_voice_id: Optional[str] = None  # Voice for text in asterisks (*emotive text*)
    emotive_speed: float = 1.0
    emotive_stability: float = 0.5
    emotive_similarity_boost: float = 0.8
    # Audio output device
    output_device_name: Optional[str] = None  # None = default device, or specify device name
    # Audio archiving
    audio_archive_dir: Optional[str] = None  # Directory to save audio files for replay

@dataclass
class AlignmentData:
    start_time_played: float
    # {'chars': [
    # 'W', 'h', 'a', 't', ' ', 'd', 'o', ' ', 'y', 'o', 'u', ' ',
    #  'k', 'n', 'o', 'w', ' ', 'a', 'b', 'o', 'u', 't', ' ', 'w', 'h', 'a', 't',
    # ' ', 'I', "'", 'm', ' ', 'd', 'o', 'i', 'n', 'g', '?', ' '],
    # 'charStartTimesMs':
    # [0, 81, 128, 151, 174, 197, 221, 244, 279, 302, 325, 360, 418, 464, 499, 534, 569, 627, 673, 720, 766, 801, 836, 871, 894, 929, 964, 987, 1022, 1057, 1091, 1138, 1196, 1242, 1358, 1428, 1463, 1533, 1614], 'charDurationsMs': [81, 47, 23, 23, 23, 24, 23, 35, 23, 23, 35, 58, 46, 35, 35, 35, 58, 46, 47, 46, 35, 35, 35, 23, 35, 35, 23, 35, 35, 34, 47, 58, 46, 116, 70, 35, 70, 81, 244]}
    chars: list
    chars_start_times_ms: list
from rapidfuzz.distance import Levenshtein

# with a of like "*wow hi there* I like bees" and b of "wow hi there i like" this will give us index of end of like inside a
def _collapse(s: str, ignore: set[str]):
    kept = []
    idx_map = []
    for idx, ch in enumerate(s):
        if ch in ignore:
            continue
        kept.append(ch)
        idx_map.append(idx)
    return "".join(kept), idx_map

def _noisy_prefix_end(filtered_a: str, filtered_b: str) -> int | None:
    ops = iter(Levenshtein.editops(filtered_b, filtered_a))
    op = next(ops, None)
    i = j = 0
    last_match = -1

    while i < len(filtered_b):
        if op and op.src_pos == i and op.dst_pos == j:
            if op.tag in {"delete", "replace"}:
                return None
            j += 1               # insertion in filtered_a
            op = next(ops, None)
        else:
            last_match = j
            i += 1
            j += 1

    return last_match + 1

def trimmed_end(a: str, b: str, ignore="*\n\r\t #@\\/.,?!-+[]()&%$:"):
    ignore_set = set(ignore)
    filtered_a, idx_map_a = _collapse(a, ignore_set)
    filtered_b, _ = _collapse(b, ignore_set)

    end = _noisy_prefix_end(filtered_a, filtered_b)
    if end is None:
        return None
    if end == 0:
        return 0
    return idx_map_a[end - 1] + 1


class AsyncTTSStreamer:
    """Async TTS Streamer with interruption capabilities and spoken content tracking."""
    
    def __init__(self, config: TTSConfig):
        self.config = config
        self.speak_task = None
        self.generated_text = ""
        self._archive_audio_buffer = []  # Accumulates audio for archiving
        self._current_message_id = None
        
        # Audio level metering callback (for UI visualization)
        self.output_level_callback: Optional[Callable[[float], None]] = None

    def _save_audio_archive(self, audio_data: np.ndarray, message_id: str):
        """Save audio to archive directory as WAV file."""
        if not self.config.audio_archive_dir or not message_id:
            return

        try:
            # Create ai subdirectory
            ai_dir = os.path.join(self.config.audio_archive_dir, "ai")
            os.makedirs(ai_dir, exist_ok=True)

            # Save as WAV file at 48kHz (playback sample rate)
            filepath = os.path.join(ai_dir, f"{message_id}.wav")

            # Convert float32 [-1, 1] to int16
            audio_int16 = (audio_data * 32767).astype(np.int16)
            wavfile.write(filepath, shared_sample_rate(), audio_int16)
            print(f"Saved AI audio to {filepath}")
        except Exception as e:
            print(f"Error saving audio archive: {e}")

    def _get_voice_settings(self, is_emotive: bool) -> Dict[str, float]:
        """Get voice settings for regular or emotive speech."""
        if is_emotive:
            return {
                "speed": self.config.emotive_speed,
                "stability": self.config.emotive_stability,
                "similarity_boost": self.config.emotive_similarity_boost
            }
        else:
            return {
                "speed": self.config.speed,
                "stability": self.config.stability,
                "similarity_boost": self.config.similarity_boost
            }

    def _get_voice_key(self, voice_id: str, voice_settings: Dict[str, float]) -> str:
        """Generate a unique key for voice and settings combination."""
        settings_str = "_".join(f"{k}:{v}" for k, v in sorted(voice_settings.items()))
        return f"{voice_id}_{settings_str}"

    async def _osc_emit_helper(self, osc_client, osc_client_address, progress_callback=None):
        try:
            prev_offset = 0
            prev_progress_offset = 0
            appended_text = ""
            while True:
                offset_in_original_text = self.get_current_index_in_text()
                
                # Send progress callback with full text up to current position (preserves punctuation)
                if progress_callback and prev_progress_offset < offset_in_original_text:
                    spoken_text = self.generated_text[:offset_in_original_text]
                    try:
                        await progress_callback(spoken_text, offset_in_original_text)
                    except Exception as e:
                        print(f"Progress callback error: {e}")
                    prev_progress_offset = offset_in_original_text
                
                # OSC word emission (existing behavior)
                if prev_offset < offset_in_original_text:
                    match = re.search(r"\w+\b", self.generated_text[prev_offset:])
                    if match:
                        new_text = self.generated_text[prev_offset:prev_offset+match.end()]
                        prev_offset = prev_offset+match.end()
                    else:
                        prev_offset = len(self.generated_text)
                        new_text = self.generated_text[prev_offset:]
                    for word in re.sub(r"\s+", " ", new_text).split(" "):
                        if len(word.strip()) > 0:
                            print(f"sent message {word}")
                            osc_client.send_message(f'/message',f'{word}')
                await asyncio.sleep(0.02)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(traceback.print_exc())
            print("Error in osc emit")


    async def _speak_text_helper(self, text_generator: AsyncGenerator[str, None], first_audio_callback, osc_client, osc_client_address, progress_callback=None, character_name: Optional[str] = None) -> bool:
        """
        Speak the given text (task that can be canceled)
        
        Args:
            text_generator: yields text to convert to speech
            progress_callback: async callback(spoken_text, char_index) called as TTS progresses
            character_name: optional character name for amplitude OSC messages
            
        Returns:
            bool: True if completed successfully, False if interrupted
        """
        current_voice = None
        websocket = None
        self.alignments = []
        self.generated_text = ""
        self._archive_audio_buffer = []  # Reset audio buffer for archiving
        osc_emit_task = None
        start_time_played = None
        try:
            osc_emit_task = asyncio.create_task(self._osc_emit_helper(osc_client, osc_client_address, progress_callback))
            audio_stream = ensure_stream_started()
            audios = []
            async def flush_websocket(websocket):
                nonlocal first_audio_callback, start_time_played
                if websocket is not None:
                    await websocket.send(json.dumps({
                        "text": "", # final message
                    }))

                    audio_datas = []
                    segment_alignments = []

                    # Get the audio and add it to audio queue
                    # we buffer all audio before playing because that prevents jitters
                    # (this is okay because we do this one sentence at a time)
                    async for message in websocket:
                        data = json.loads(message)
                        audio_base64 = data.get("audio")
                        if audio_base64 and len(audio_base64) > 0:
                            audio_data = base64.b64decode(audio_base64)
                            # call callback (this will stop the "thinking" sound)
                            if not first_audio_callback is None:
                                await first_audio_callback()
                                first_audio_callback = None
                            if start_time_played is None:
                                start_time_played = time.time()

                            float_audio = int16_bytes_to_float(audio_data)
                            resampled = _resample(float_audio, self.config.sample_rate, shared_sample_rate())
                            if len(resampled) > 0:
                                audio_datas.append(resampled)
                                segment_alignments.append((len(resampled), data["alignment"]))
                        elif data.get("isFinal"):
                            break # done
                    
                    # now send the audio, all in one piece
                    concat_data = np.concatenate(audio_datas)

                    # Accumulate for archiving
                    self._archive_audio_buffer.append(concat_data)

                    # see when it'll actually be played
                    current_time = time.time()
                    buffered_duration = audio_stream.get_buffered_duration()
                    audio_stream.write(concat_data)
                    
                    # Send amplitude OSC messages for reactive lighting
                    if osc_client and character_name:
                        asyncio.create_task(self._send_amplitude_osc(
                            osc_client, character_name, concat_data, 
                            current_time + buffered_duration, shared_sample_rate()
                        ))
                    play_start_time = current_time + buffered_duration
                    for buffer_len, alignment in segment_alignments:
                        if alignment is not None: # sometimes we get no alignments but still audio data
                            alignment_data = AlignmentData(
                                start_time_played=play_start_time,
                                chars=alignment['chars'],
                                chars_start_times_ms=alignment['charStartTimesMs']
                            )
                            self.alignments.append(alignment_data)
                        play_start_time += buffer_len/float(shared_sample_rate())


                    await websocket.close()
                    
            async for sentence in stream_sentences(text_generator):
                # add spaces back between the sentences
                self.generated_text = (self.generated_text + " " + sentence).strip()

                # Filter out <function_calls>...</function_calls> content for TTS
                # Keep it in generated_text (history) but don't speak it
                speakable = re.sub(r'<function_calls>[\s\S]*?</function_calls>', '', sentence)
                # Also filter partial/unclosed function_calls blocks
                speakable = re.sub(r'<function_calls>[\s\S]*$', '', speakable)
                speakable = speakable.strip()

                # Skip if nothing left to speak
                if not speakable:
                    continue

                # split out *emotive* into seperate parts
                eleven_messages = []
                for text_part, is_emotive in self.extract_emotive_text(speakable):

                    if text_part.strip():
                        voice_id = self.config.emotive_voice_id if is_emotive else self.config.voice_id
                        voice_settings = self._get_voice_settings(is_emotive)

                        eleven_messages.append((voice_id, {
                            "text": text_part.strip(),
                            "voice_settings": voice_settings,
                        }))

                # send text to websockets and receive and play audio
                for voice_id, message in eleven_messages:
                    # Connect to WebSocket (if needed)
                    if current_voice != voice_id:
                        # Disconnect old websocket
                        if not websocket is None:
                            await flush_websocket(websocket)
                            websocket = None
                        current_voice = voice_id
            
                        uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input"
                        params = f"?model_id={self.config.model_id}&output_format={self.config.output_format}"

                        websocket = await websockets.connect(uri + params)

                        # Send initial configuration
                        initial_message = {
                            "text": " ",
                            "voice_settings": message["voice_settings"],
                            "xi_api_key": self.config.api_key
                        }
                        await websocket.send(json.dumps(initial_message))
                    await websocket.send(json.dumps({
                        "text": message['text'].strip() + " ", # elevenlabs always requires ends with single space
                    }))

                # stream the speech, this allows us to start outputting speech before it's done outputting
                await flush_websocket(websocket)
                websocket = None
                current_voice = None

            # wait for buffered audio to drain (polling)
            while audio_stream.get_buffered_duration() > 0:
                await asyncio.sleep(0.05)

            # If no audio was ever generated (e.g., all content was filtered),
            # still call the callback to stop the thinking sound
            if first_audio_callback is not None:
                await first_audio_callback()
                first_audio_callback = None

            # Save accumulated audio to archive (only if completed successfully)
            if self._archive_audio_buffer and self._current_message_id:
                full_audio = np.concatenate(self._archive_audio_buffer)
                self._save_audio_archive(full_audio, self._current_message_id)

            return True  # Completed successfully

        except asyncio.CancelledError:
            print(f"Cancelled tts, interrupting playback")
            # not enough audio played and interrupted, make empty
            if start_time_played is None or time.time() - start_time_played < 2.0:
                self.generated_text = ""
            # no audio played yet, empty text
            elif len(self.alignments) == 0:
                self.generated_text = ""
            else:
                # AI Alignment TM
                # (computes where in the text it was interrupted and trims context to that)
                offset_in_original_text = self.get_current_index_in_text()
                self.generated_text = self.generated_text[:offset_in_original_text]
                #print(f"Fuzzy matched to position {offset_in_original_text}")
                #print(self.generated_text)

            # interrupt audio, this clears the buffers
            interrupt_playback()
            raise
        except Exception as e:
            print(f"TTS error")
            print(traceback.print_exc())
            return False  # Failed due to error
        finally:
            if osc_emit_task:
                try:
                    osc_emit_task.cancel()
                    await osc_emit_task
                except asyncio.CancelledError:
                    pass # intentional
            # close websocket
            if websocket:
                try:
                    await websocket.close()
                except Exception as e:
                    print(f"TTS websocket close error")
                    print(traceback.print_exc())
                

    def get_current_index_in_text(self):
        if len(self.alignments) == 0:
            return 0
        # AI Alignment TM
        # (computes where in the text it was interrupted and trims context to that)
        current_time = time.time()
        end_alignment_i = 0 # if none have been played, don't include any
        end_char_i_in_alignment_i = 0 # if no characters have been played, don't include any
        for alignmentI, alignment in list(enumerate(self.alignments))[::-1]:
            # find the latest thing that is already started playing
            if alignment.start_time_played < current_time:
                end_alignment_i = alignmentI
                millis_since_start_time = (current_time-alignment.start_time_played)*1000
                # find the latest char that has already played
                end_char_i_in_alignment_i = 0 # default to first (if none in array)
                for charI, (char, char_ms) in list(enumerate(zip(alignment.chars, alignment.chars_start_times_ms)))[::-1]:
                    if char_ms < millis_since_start_time:
                        end_char_i_in_alignment_i = charI+1
                        break
                break
            
        played_chars = [alignment.chars for alignment in self.alignments[:end_alignment_i]] + [self.alignments[end_alignment_i].chars[:end_char_i_in_alignment_i]]
        played_text = " ".join(["".join(chars) for chars in played_chars])
        # do some fuzzy matching to handle the loss of things like *
        #print(self.generated_text)
        return trimmed_end(self.generated_text, played_text)

    async def _send_amplitude_osc(self, osc_client, character_name: str, audio_data: np.ndarray, play_start_time: float, sample_rate: int):
        """Send amplitude OSC messages timed to audio playback.
        
        Breaks audio into ~30ms chunks and sends amplitude values at the right time.
        """
        chunk_duration = 0.03  # 30ms chunks for ~33 updates/sec
        chunk_samples = int(sample_rate * chunk_duration)
        
        try:
            for i in range(0, len(audio_data), chunk_samples):
                chunk = audio_data[i:i + chunk_samples]
                
                # Calculate RMS amplitude
                rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
                # Normalize to 0-1 range (adjust 0.3 based on typical TTS loudness)
                amplitude = min(1.0, rms / 0.3)
                
                # Wait until it's time to send this chunk's amplitude
                chunk_play_time = play_start_time + (i / sample_rate)
                wait_time = chunk_play_time - time.time()
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                
                # Send OSC message
                osc_client.send_message("/character/amplitude", [character_name, float(amplitude)])
                
                # Also send to UI level callback if registered
                if self.output_level_callback:
                    try:
                        self.output_level_callback(amplitude)
                    except Exception:
                        pass
                
        except asyncio.CancelledError:
            pass  # Task cancelled, stop sending
        except Exception as e:
            print(f"⚠️ Amplitude OSC error: {e}")
    
    async def speak_text(self, text_generator: AsyncGenerator[str, None], first_audio_callback, osc_client, osc_client_address, message_id: Optional[str] = None, progress_callback=None, character_name: Optional[str] = None):
        # interrupt (if already running)
        await self.interrupt()

        # Store message_id for audio archiving
        self._current_message_id = message_id

        # start the task (this way it's cancellable and we don't need to spam checks)
        self.speak_task = asyncio.create_task(self._speak_text_helper(text_generator, first_audio_callback, osc_client, osc_client_address, progress_callback, character_name))

        # wait for it to finish
        result = await self.speak_task

        self.speak_task = None
        return result
    
    def is_currently_playing(self):
        return self.speak_task is not None
    
    async def interrupt(self):
        if self.speak_task is not None: # if existing one, stop it
            try:
                self.speak_task.cancel()
                await self.speak_task # wait for it to cancel
            except asyncio.CancelledError:
                pass # intentional
            except Exception as e:
                print(f"TTS await error")
                print(traceback.print_exc()) 
            finally:
                self.speak_task = None
        return self.generated_text
        
    async def cleanup(self):
        await self.interrupt()

    def extract_emotive_text(self, text: str) -> List[Tuple[str, bool]]:
        """
        Parse text to separate regular text from emotive text (in asterisks).

        Args:
            text: Input text that may contain *emotive* parts

        Returns:
            List of (text_chunk, is_emotive) tuples

        Note:
            Only matches single asterisks (*text*), not double (**bold**) which is
            markdown formatting. The regex uses negative lookbehind/lookahead to
            ensure we don't match asterisks that are part of ** pairs.
        """
        if not self.config.emotive_voice_id:
            # No emotive voice configured, return all as regular text
            return [(text, False)]

        parts = []
        current_pos = 0

        # Find all *text* patterns, but NOT **text** (markdown bold)
        # (?<!\*) = not preceded by another asterisk
        # (?!\*) = not followed by another asterisk
        # This ensures we match *single* but not **double**
        for match in re.finditer(r'(?<!\*)\*([^*]+)\*(?!\*)', text):
            # Add regular text before the emotive part
            if match.start() > current_pos:
                regular_text = text[current_pos:match.start()]
                if regular_text.strip():
                    parts.append((regular_text, False))

            # Add emotive text (content inside asterisks)
            emotive_text = match.group(1)
            if emotive_text.strip():
                parts.append((emotive_text, True))

            current_pos = match.end()

        # Add remaining regular text
        if current_pos < len(text):
            remaining_text = text[current_pos:]
            if remaining_text.strip():
                parts.append((remaining_text, False))

        return parts if parts else [(text, False)]


_SENTENCE_END = re.compile(r'([.?!][\'")\]]*)(?=\s)')

async def stream_sentences(
    text_stream: AsyncGenerator[str, None]
) -> AsyncGenerator[str, None]:
    buffer = ""

    async for chunk in text_stream:
        buffer += chunk

        while True:
            match = _SENTENCE_END.search(buffer)
            if not match:
                break

            end = match.end(1)
            sentence = buffer[:end].strip()
            buffer = buffer[end:].lstrip()

            if sentence:
                yield sentence

    tail = buffer.strip()
    if tail:
        yield tail
