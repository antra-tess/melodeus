#!/usr/bin/env python3
"""
Async TTS Module for ElevenLabs WebSocket Streaming
Provides interruptible text-to-speech with real-time audio playback.
Now includes Whisper-based tracking of actual spoken content.
"""

import asyncio
import websockets
import json
import base64
import threading
import queue
import time
import numpy as np
import re
import difflib
from typing import Optional, AsyncGenerator, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from scipy import signal

# Try to use mel-aec first, fall back to pyaudio if not available
try:
    from mel_aec_adapter import PyAudio as MelAecAudio, paInt16
    print("✅ Using mel-aec for TTS audio output with interruption tracking")
    USING_MEL_AEC = True
except ImportError:
    import pyaudio
    from pyaudio import paInt16
    PyAudio = pyaudio.PyAudio
    print("⚠️  Using PyAudio for TTS (mel-aec not available)")
    USING_MEL_AEC = False

# Import Whisper TTS Tracker
try:
    from whisper_tts_tracker import WhisperTTSTracker, SpokenContent
    WHISPER_AVAILABLE = True
except ImportError:
    print("⚠️ Whisper TTS Tracker not available - spoken content tracking disabled")
    WHISPER_AVAILABLE = False
    SpokenContent = None

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
class TTSSession:
    """Represents a single TTS speaking session with its own isolated state."""
    session_id: str
    generated_text: str = ""  # Text that was sent to TTS for this session
    current_spoken_content: List[SpokenContent] = field(default_factory=list)  # What Whisper captured for this session
    was_interrupted: bool = False  # Whether this session was interrupted
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    whisper_tracker: Optional['WhisperTTSTracker'] = None  # Session-specific tracker
    tool_calls: List[ToolCall] = field(default_factory=list)  # Tools to execute during speech
    spoken_text_for_tts: str = ""  # Text actually sent to TTS (excludes tool content)

class AsyncTTSStreamer:
    """Async TTS Streamer with interruption capabilities and spoken content tracking."""
    
    def __init__(self, config: TTSConfig):
        self.config = config
        self.audio_queue = queue.Queue()
        self.is_playing = False
        self.is_streaming = False
        self.playback_thread = None
        self.websocket = None
        self.audio_task = None
        
        # Audio setup
        if USING_MEL_AEC:
            self.p = MelAecAudio()
            print("🎯 mel-aec: Precise playback position tracking enabled")
        else:
            self.p = pyaudio.PyAudio()
        self.stream = None
        
        # Control flags
        self._stop_requested = False
        self._interrupted = False
        
        # Callback for first audio chunk
        self.first_audio_callback = None
        self._first_audio_received = False
        
        # Multi-voice completion tracking
        self._websockets_completed = 0
        self._total_websockets = 0
        self._audio_completion_event = asyncio.Event()
        self._final_audio_received = False
        self._queue_monitoring_task = None
        self._all_text_sent = False  # Track when all text has been sent to websockets
        self._actively_playing_chunk = False  # Track if audio thread is playing a chunk
        
        # Whisper tracking configuration
        self.track_spoken_content = WHISPER_AVAILABLE
        
        # Current session tracking
        self.current_session: Optional[TTSSession] = None
        self._session_counter = 0  # For generating unique session IDs
        self._chunks_played = 0  # Track how many audio chunks have been played
        
        # Store last session's generated text for recovery
        self.last_session_generated_text = ""
        
        # Tool execution callback
        self.on_tool_execution = None  # Callback: async def(tool_call: ToolCall) -> ToolResult
        
        # Voice connection management for prosodic continuity
        self._current_voice_connection = None  # Current voice connection
        self._current_voice_key = None  # Key of current voice
        self._current_voice_task = None  # Task handling current connection
        
        
        # Whisper will be initialized per session
    
    def _fuzzy_find_position(self, whisper_text: str, tts_text: str) -> int:
        """
        Find the approximate position in TTS text that corresponds to the end of Whisper text.
        Uses very fuzzy matching to handle Whisper transcription errors.
        
        Returns:
            Character position in tts_text that best matches the end of whisper_text
        """
        if not whisper_text or not tts_text:
            return 0
            
        # Normalize texts for comparison
        def normalize(text):
            # Convert to lowercase
            text = text.lower()
            # Keep ONLY letters and spaces - remove ALL punctuation, numbers, asterisks, etc.
            text = re.sub(r'[^a-z\s]', ' ', text)
            # Collapse multiple spaces
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        # Normalize and split into words for more robust matching
        norm_whisper = normalize(whisper_text)
        norm_tts = normalize(tts_text)
        
        whisper_words = norm_whisper.split()
        tts_words = norm_tts.split()
        
        if not whisper_words:
            return 0
            
        # Try to find the best match using a sliding window approach
        best_match_score = 0
        best_match_end = 0
        
        # Look for the last few words of Whisper in TTS (more reliable than full text)
        window_size = min(5, len(whisper_words))  # Use last 5 words or less
        search_words = whisper_words[-window_size:]
        
        # Slide through TTS text looking for best match
        for i in range(len(tts_words) - window_size + 1):
            window = tts_words[i:i + window_size]
            
            # Calculate similarity score
            matcher = difflib.SequenceMatcher(None, window, search_words)
            score = matcher.ratio()
            
            # If this is a better match, update
            if score > best_match_score:
                best_match_score = score
                best_match_end = i + window_size
        
        # If we found a good match (>60% similar), use it
        if best_match_score > 0.6:
            # Convert word position to character position
            word_count = 0
            for i, char in enumerate(tts_text):
                if char.isspace() and i > 0 and not tts_text[i-1].isspace():
                    word_count += 1
                    if word_count >= best_match_end:
                        return i
            
            # If we counted all words, return end of text
            return len(tts_text)
        
        # Fallback: Try character-level fuzzy matching on normalized text
        if norm_whisper in norm_tts:
            # Find the position in normalized text
            norm_pos = norm_tts.find(norm_whisper) + len(norm_whisper)
            # Estimate position in original text
            ratio = norm_pos / len(norm_tts)
            result = int(len(tts_text) * ratio)
            return result
        
        # Final fallback: percentage-based estimation with safety margin
        # Whisper often captures less than TTS sends, so add 10% margin
        whisper_ratio = len(whisper_text) / max(len(tts_text), 1)
        estimated_pos = int(len(tts_text) * min(whisper_ratio * 1.1, 1.0))
        
        return estimated_pos
    
    def _create_session(self) -> TTSSession:
        """Create a new TTS session with unique ID and optional Whisper tracker."""
        self._session_counter += 1
        session_id = f"tts_session_{int(time.time())}_{self._session_counter}"
        session = TTSSession(session_id=session_id)
        
        # Initialize Whisper tracker for this session if available
        if WHISPER_AVAILABLE and self.track_spoken_content:
            try:
                session.whisper_tracker = WhisperTTSTracker(sample_rate=16000)
                print(f"✅ Whisper TTS tracking enabled for session {session_id}")
            except Exception as e:
                print(f"⚠️ Failed to initialize Whisper tracker for session: {e}")
                session.whisper_tracker = None
        
        return session
        
    async def speak_text(self, text: str) -> bool:
        """
        Speak the given text with interruption support.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            bool: True if completed successfully, False if interrupted
        """
        if self.is_streaming:
            await self.stop()
            
        try:
            self._stop_requested = False
            self._interrupted = False
            self._first_audio_received = False
            self._chunks_played = 0  # Reset chunks played counter
            
            # Create a new session for this speaking operation
            self.current_session = self._create_session()
            self.current_session.generated_text = text
            # Store for recovery in case of error
            self.last_session_generated_text = text
            
            if self.current_session.whisper_tracker and self.track_spoken_content:
                self.current_session.whisper_tracker.start_tracking()
            
            await self._start_streaming(text)
            
            # If we reached here without stop being requested, it completed successfully
            completed_successfully = not self._stop_requested
            self._interrupted = self._stop_requested  # Set interrupted flag based on actual completion
            
            return completed_successfully
            
        except Exception as e:
            print(f"TTS error: {e}")
            return False
        finally:
            await self._cleanup_stream()
    
    def _parse_emotive_text(self, text: str) -> List[Tuple[str, bool]]:
        """
        Parse text to separate regular text from emotive text (in asterisks).
        
        Args:
            text: Input text that may contain *emotive* parts
            
        Returns:
            List of (text_chunk, is_emotive) tuples
        """
        if not self.config.emotive_voice_id:
            # No emotive voice configured, return all as regular text
            return [(text, False)]
        
        parts = []
        current_pos = 0
        
        # Find all *text* patterns
        for match in re.finditer(r'\*([^*]+)\*', text):
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


    async def speak_stream_multi_voice(self, text_generator: AsyncGenerator[str, None]) -> bool:
        """
        Speak streaming text with multi-voice support for emotive expressions.
        Detects *emotive text* and uses different voice for those parts.
        If no emotive voice is configured, uses main voice for everything.
        
        Args:
            text_generator: Async generator yielding text chunks
            
        Returns:
            bool: True if completed successfully, False if interrupted
        """
            
        if self.is_streaming:
            await self.stop()
            
        # Initialize progress_monitor_task early to avoid UnboundLocalError
        progress_monitor_task = None
            
        try:
            self._stop_requested = False
            self._interrupted = False
            self._first_audio_received = False
            
            # Cancel any existing session and create a new one
            if self.current_session and not self.current_session.was_interrupted:
                print(f"⚠️ Cancelling previous session: {self.current_session.session_id}")
                self.current_session.was_interrupted = True
                
            # Close any existing voice connection from previous session
            if self._current_voice_connection:
                await self._close_current_voice_connection()
            
            # Create a new session for this speaking operation
            self.current_session = self._create_session()
            print(f"🆕 Created TTS session: {self.current_session.session_id}")
            self._websockets_completed = 0
            self._total_websockets = 0
            self._audio_completion_event.clear()
            self._final_audio_received = False
            self._queue_monitoring_task = None
            self._all_text_sent = False
            self._speech_monitor_task = None
            
            if self.track_spoken_content and self.current_session.whisper_tracker:
                self.current_session.whisper_tracker.start_tracking()
            
            # Don't start monitoring here - wait until we know if we have tool calls
            
            # Ensure audio playback is ready
            # But first, ensure any previous audio is completely stopped
            if self.playback_thread and self.playback_thread.is_alive():
                print("⏳ Stopping previous audio playback...")
                # Force stop the previous playback
                self.is_playing = False
                self._clear_audio_queue()  # Clear any remaining audio
                
                # Close the stream to force the thread to exit
                if self.stream:
                    try:
                        self.stream.stop_stream()
                        self.stream.close()
                        self.stream = None
                    except Exception as e:
                        print(f"Error closing previous stream: {e}")
                
                # Wait for thread to actually exit
                max_wait = 1.0
                start_time = time.time()
                while (self.playback_thread.is_alive() and 
                       (time.time() - start_time) < max_wait):
                    await asyncio.sleep(0.05)
                
                if self.playback_thread.is_alive():
                    print("⚠️ Previous audio thread didn't stop, but proceeding anyway")
                else:
                    print("✅ Previous audio thread stopped")
                    
                self.playback_thread = None
                    
            # Now start fresh audio playback
            if not self.is_playing:
                self._start_audio_playback()
            self.is_streaming = True
            
            text_buffer = ""
            xml_buffer = ""  # Buffer for incomplete XML tags
            in_xml_tag = False
            current_xml_start = -1
            
            async for chunk in text_generator:
                if self._stop_requested:
                    self._interrupted = True
                    break
                    
                # Add to generated text regardless
                if self.current_session:
                    self.current_session.generated_text += chunk
                    # Also update recovery text
                    self.last_session_generated_text = self.current_session.generated_text
                
                # Process each character to detect XML tags
                for i, char in enumerate(chunk):
                    if char == '<' and not in_xml_tag:
                        # Potential start of XML tag
                        in_xml_tag = True
                        current_xml_start = len(self.current_session.generated_text) - len(chunk) + i
                        xml_buffer = char
                        # Debug: Log when we start detecting XML
                        print(f"🔍 DEBUG: Started XML tag detection at position {current_xml_start}, text_buffer so far: {repr(text_buffer[-20:])}")
                    elif in_xml_tag:
                        xml_buffer += char
                        if char == '>':
                            # Check if this is a closing tag or complete tag
                            if xml_buffer.endswith('/>') or self._is_closing_tag(xml_buffer):
                                # Complete XML tag found
                                tag_end = len(self.current_session.generated_text) - len(chunk) + i + 1
                                # The position where the tool should execute is where the tag started
                                self._process_xml_tag(xml_buffer, len(self.current_session.spoken_text_for_tts), tag_end)
                                print(f"🔍 DEBUG: Complete XML tag: {repr(xml_buffer)}, text_buffer after: {repr(text_buffer[-20:])}")
                                in_xml_tag = False
                                xml_buffer = ""
                                current_xml_start = -1
                            else:
                                # Found '>' but tag not complete - might be inside tag content
                                # For example: <tag attr="value>something">
                                print(f"🔍 DEBUG: Found '>' but tag not complete: {repr(xml_buffer)}")
                    else:
                        # Regular text - add to buffer only if not in XML
                        text_buffer += char
                        if self.current_session:
                            self.current_session.spoken_text_for_tts += char
                
                # Process complete sentences/phrases for voice switching
                if not in_xml_tag:  # Only process when not inside XML
                    sentences = self._split_into_sentences(text_buffer)
                    
                    # Keep the last incomplete sentence in buffer
                    if sentences and not text_buffer.rstrip().endswith(('.', '!', '?')):
                        complete_sentences = sentences[:-1]
                        text_buffer = sentences[-1]
                    else:
                        complete_sentences = sentences
                        text_buffer = ""
                    
                    # Process each complete sentence with appropriate voice
                    for sentence in complete_sentences:
                        if sentence.strip():
                            # Debug: Check if sentence has both asterisks and angle brackets
                            if '*' in sentence and ('<' in sentence or '>' in sentence):
                                print(f"⚠️ DEBUG: Sentence with asterisks and XML: {repr(sentence)}")
                            
                            await self._speak_sentence_with_voice_switching(sentence)
                            
                            if self._stop_requested:
                                self._interrupted = True
                                break
                else:
                    # We're still in an XML tag at the end of this chunk
                    print(f"🔍 DEBUG: Chunk ended while in XML tag. xml_buffer: {repr(xml_buffer)}, text_buffer: {repr(text_buffer[-30:])}")
            
            # Process any remaining text in buffer
            if text_buffer.strip() and not self._stop_requested:
                await self._speak_sentence_with_voice_switching(text_buffer)
            
            # Check for incomplete XML tag at end of message
            if xml_buffer and in_xml_tag and not self._stop_requested:
                print(f"⚠️ Incomplete tool call at end of message: {xml_buffer}")
                print(f"⚠️ DEBUG: Final text_buffer: {repr(text_buffer)}")
                print(f"⚠️ DEBUG: Was in_xml_tag: {in_xml_tag}")
                # Check if it's a complete self-closing tag or has matching closing tag
                if self._is_closing_tag(xml_buffer):
                    tag_end = len(self.current_session.generated_text)
                    self._process_xml_tag(xml_buffer, len(self.current_session.spoken_text_for_tts), tag_end)
                    print(f"✅ Processed tool call at end of message")
            
            # Mark that all text has been sent
            self._all_text_sent = True
            print(f"📝 All text sent to TTS ({self._total_websockets} websockets created)")
            
            # Close current voice connection and send completion signal
            if self._current_voice_connection:
                await self._close_current_voice_connection()
            
            # Start monitoring speech progress for tool execution if we have tool calls
            if self.current_session.tool_calls and self.track_spoken_content:
                print(f"🔍 Starting speech progress monitoring for {len(self.current_session.tool_calls)} tool calls")
                self._speech_monitor_task = asyncio.create_task(self._monitor_speech_progress())
            
            # Wait for all audio to finish playing (simple polling approach)
            if not self._stop_requested:
                await self._wait_for_audio_completion()
            
            # Stop Whisper tracking
            if self.track_spoken_content and self.current_session and self.current_session.whisper_tracker:
                self.current_session.current_spoken_content = self.current_session.whisper_tracker.stop_tracking()
            
            # Check for any unexecuted tools after audio completes
            if self.current_session.tool_calls and not self._stop_requested:
                await self._execute_remaining_tools()
            
            return not self._interrupted
            
        except Exception as e:
            print(f"Multi-voice TTS error: {e}")
            self._interrupted = True
            # Stop Whisper tracking on error
            if self.track_spoken_content and self.current_session and self.current_session.whisper_tracker:
                try:
                    self.current_session.whisper_tracker.stop_tracking()
                except Exception:
                    pass
            return False
        finally:
            # Cancel progress monitor if running
            if hasattr(self, '_speech_monitor_task') and self._speech_monitor_task and not self._speech_monitor_task.done():
                self._speech_monitor_task.cancel()
                try:
                    await self._speech_monitor_task
                except asyncio.CancelledError:
                    pass
            if progress_monitor_task and not progress_monitor_task.done():
                progress_monitor_task.cancel()
                try:
                    await progress_monitor_task
                except asyncio.CancelledError:
                    pass
            
            self.is_streaming = False
            await self._cleanup_multi_voice()
            
        # Return success/failure based on interruption
        return not self._interrupted

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for voice processing."""
        # Simple sentence splitting - can be enhanced
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _is_closing_tag(self, xml_buffer: str) -> bool:
        """Check if the buffer contains a complete tag (either self-closing or with closing tag)."""
        # Check for self-closing tag
        if xml_buffer.endswith('/>'):
            return True
        # Check if we have a closing tag pattern
        if re.match(r'</\w+>', xml_buffer):
            return True
        # Check if we have opening and closing tags
        tag_match = re.match(r'<(\w+)[^>]*>', xml_buffer)
        if tag_match:
            tag_name = tag_match.group(1)
            return f'</{tag_name}>' in xml_buffer
        return False
    
    def _process_xml_tag(self, xml_content: str, start_pos: int, end_pos: int):
        """Process a complete XML tag and add it to tool calls."""
        if not self.current_session:
            return
            
        # Extract tag name
        tag_match = re.match(r'<(\w+)[^>]*>', xml_content)
        if tag_match:
            tag_name = tag_match.group(1)
            
            # Create tool call
            tool_call = ToolCall(
                tag_name=tag_name,
                content=xml_content,
                start_position=start_pos,
                end_position=end_pos
            )
            
            self.current_session.tool_calls.append(tool_call)
            print(f"🔧 Found tool call: {tag_name} at TTS position {start_pos}")
            print(f"   Current spoken_text_for_tts length: {len(self.current_session.spoken_text_for_tts)}")
            print(f"   Current generated_text length: {len(self.current_session.generated_text)}")
    
    async def _monitor_speech_progress(self):
        """Monitor speech progress using Whisper and execute tools when appropriate."""
        if not self.current_session or not self.current_session.whisper_tracker:
            return
            
        while self.is_streaming and not self._stop_requested:
            await asyncio.sleep(0.5)  # Check every 500ms
            
            # Get current spoken text and TTS text
            spoken_text = self.current_session.whisper_tracker.get_spoken_text()
            tts_text = self.current_session.spoken_text_for_tts
            
            # Use fuzzy matching to find actual position
            fuzzy_position = self._fuzzy_find_position(spoken_text, tts_text)
            
            # Check which tools should be executed based on fuzzy position
            for tool_call in self.current_session.tool_calls:
                if not tool_call.executed:
                    if fuzzy_position >= tool_call.start_position:
                        # Execute the tool
                        progress = (fuzzy_position / tool_call.start_position) * 100
                        print(f"📊 Tool '{tool_call.tag_name}' reached! Progress: {progress:.1f}%")
                        await self._execute_tool(tool_call)
                        tool_call.executed = True
    
    async def _execute_tool(self, tool_call: ToolCall):
        """Execute a tool call and handle the result."""
        print(f"🚀 Executing tool: {tool_call.tag_name} at spoken position {tool_call.start_position}")
        
        # Call the registered callback if available
        if self.on_tool_execution:
            try:
                result = await self.on_tool_execution(tool_call)
                
                if result.should_interrupt:
                    print(f"🛑 Tool requested interruption")
                    # Stop current TTS playback
                    self._stop_requested = True
                    self.current_session.was_interrupted = True
                    
                    # If there's content to speak, queue it up
                    if result.content:
                        print(f"📢 Tool returned content to speak: {result.content[:50]}...")
                        # Store the content for the conversation handler to process
                        # This will be handled by the conversation manager
                        
            except Exception as e:
                print(f"❌ Tool execution error: {e}")
    
    async def _execute_remaining_tools(self):
        """Execute any remaining unexecuted tools after audio completes."""
        unexecuted_tools = [tc for tc in self.current_session.tool_calls if not tc.executed]
        
        if unexecuted_tools:
            print(f"🔧 Checking {len(unexecuted_tools)} remaining tool(s) after audio completion")
            
            # Check if the message was interrupted
            was_interrupted = self._interrupted or self.current_session.was_interrupted
            
            if not was_interrupted:
                # Message completed naturally - execute ALL remaining tools
                print(f"✅ Message completed naturally - executing all remaining tools")
                for tool_call in unexecuted_tools:
                    print(f"🚀 Executing tool '{tool_call.tag_name}' at position {tool_call.start_position}")
                    await self._execute_tool(tool_call)
                    tool_call.executed = True
            else:
                # Message was interrupted - use fuzzy matching to determine which tools to execute
                print(f"⚠️ Message was interrupted - checking tool progress")
                
                # Get final spoken text
                final_spoken_text = ""
                if self.current_session.current_spoken_content:
                    final_spoken_text = " ".join(content.text for content in self.current_session.current_spoken_content)
                elif self.current_session.whisper_tracker:
                    final_spoken_text = self.current_session.whisper_tracker.get_spoken_text()
                
                tts_text = self.current_session.spoken_text_for_tts
                final_fuzzy_position = self._fuzzy_find_position(final_spoken_text, tts_text)
                
                # Execute remaining tools based on progress
                for tool_call in unexecuted_tools:
                    if not tool_call.executed:
                        progress = (final_fuzzy_position / tool_call.start_position) * 100 if tool_call.start_position > 0 else 0
                        
                        # Execute tools that were close to being reached
                        # Use a more forgiving threshold (80%) for tools near the end
                        threshold = 80 if tool_call.start_position > len(tts_text) * 0.8 else 85
                        
                        if progress >= threshold:
                            print(f"   ✅ Executing interrupted tool at {progress:.1f}% progress")
                            await self._execute_tool(tool_call)
                            tool_call.executed = True
                        else:
                            print(f"   ❌ Skipping tool at {progress:.1f}% progress (threshold: {threshold}%)")

    async def _speak_sentence_with_voice_switching(self, sentence: str):
        """Speak a sentence with appropriate voice switching for emotive parts."""
        parts = self._parse_emotive_text(sentence)
        
        # Group consecutive parts by voice to maintain prosodic continuity
        voice_groups = []
        current_group = {"voice_id": None, "voice_settings": None, "parts": []}
        
        for text_part, is_emotive in parts:
            if text_part.strip():
                voice_id = self.config.emotive_voice_id if is_emotive else self.config.voice_id
                voice_settings = self._get_voice_settings(is_emotive)
                
                # Check if this part uses the same voice as current group
                if (current_group["voice_id"] == voice_id and 
                    current_group["voice_settings"] == voice_settings):
                    # Add to current group
                    current_group["parts"].append(text_part)
                else:
                    # Start new group, but first save the current one
                    if current_group["parts"]:
                        voice_groups.append(current_group)
                    
                    current_group = {
                        "voice_id": voice_id,
                        "voice_settings": voice_settings,
                        "parts": [text_part]
                    }
        
        # Add the final group
        if current_group["parts"]:
            voice_groups.append(current_group)
        
        # Process each voice group
        for group in voice_groups:
            if self._stop_requested:
                break
                
            # Get voice key for this group
            voice_key = self._get_voice_key(group["voice_id"], group["voice_settings"])
            
            # Check if we need to switch voices (close current and create new)
            if self._current_voice_key and self._current_voice_key != voice_key:
                # Voice change detected - close current connection
                print(f"🔄 Voice change detected: {self._current_voice_key} → {voice_key}")
                await self._close_current_voice_connection()
            
            # Track websockets created
            if self._current_voice_key != voice_key:
                self._total_websockets += 1
            
            # Combine all parts for this voice into one text
            combined_text = "".join(group["parts"])
            
            # Speak using current or new connection
            await self._speak_text_part(combined_text, group["voice_id"], group["voice_settings"])

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
    
    async def _get_or_create_voice_connection(self, voice_id: str, voice_settings: Dict[str, float]):
        """Get current voice connection or create a new one."""
        voice_key = self._get_voice_key(voice_id, voice_settings)
        
        # Check if this is the current voice
        if self._current_voice_key == voice_key and self._current_voice_connection:
            return self._current_voice_connection
        
        # If we have a different voice active, it should have been closed already
        # Create new connection
        uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input"
        params = f"?model_id={self.config.model_id}&output_format={self.config.output_format}"
        
        websocket = await websockets.connect(uri + params)
        
        # Send initial configuration
        initial_message = {
            "text": " ",
            "voice_settings": voice_settings,
            "xi_api_key": self.config.api_key
        }
        await websocket.send(json.dumps(initial_message))
        
        # Store as current connection
        self._current_voice_connection = websocket
        self._current_voice_key = voice_key
        
        # Start task to handle audio responses
        self._current_voice_task = asyncio.create_task(self._handle_websocket_audio(websocket))
        
        print(f"🔗 Created voice connection: {voice_key}")
        return websocket
    
    async def _close_current_voice_connection(self):
        """Close the current voice connection after sending completion signal."""
        if not self._current_voice_connection:
            return
            
        try:
            # Send completion signal to trigger audio generation
            await self._current_voice_connection.send(json.dumps({"text": ""}))
            print(f"🔚 Sent completion signal for voice: {self._current_voice_key}")
            
            # IMPORTANT: Wait for audio task to complete BEFORE closing the connection
            # This ensures all audio is received and played
            if self._current_voice_task:
                try:
                    await self._current_voice_task
                except Exception as e:
                    print(f"Error in audio task: {e}")
            
            # Now safe to close the connection
            await self._current_voice_connection.close()
            # Track completion
            self._websockets_completed += 1
            print(f"🔗 Closed voice connection: {self._current_voice_key} ({self._websockets_completed}/{self._total_websockets})")
        except Exception as e:
            print(f"Error closing voice connection {self._current_voice_key}: {e}")
        
        # Clear current connection
        self._current_voice_connection = None
        self._current_voice_key = None
        self._current_voice_task = None

    async def _speak_text_part(self, text: str, voice_id: str, voice_settings: Dict[str, float]):
        """Speak a single text part with specified voice and settings."""
        try:
            # Check if we should stop before starting
            if self._stop_requested:
                return
                
            # Get or create persistent connection for this voice
            websocket = await self._get_or_create_voice_connection(voice_id, voice_settings)
            
            # Check interruption before sending text
            if self._stop_requested:
                return
                
            # Send the text to existing connection
            text_message = {
                "text": text,
                "try_trigger_generation": True
            }
            await websocket.send(json.dumps(text_message))
            print(f"📤 Sent to persistent TTS connection: '{text[:50]}{'...' if len(text) > 50 else ''}'")
                
        except Exception as e:
            if not self._stop_requested:  # Don't log errors if we were interrupted
                print(f"Error speaking text part '{text[:50]}...': {e}")

    async def _handle_websocket_audio(self, websocket):
        """Handle audio responses from a websocket connection."""
        try:
            async for message in websocket:
                if self._stop_requested:
                    break
                    
                data = json.loads(message)
                
                if data.get("audio") and not self._stop_requested:
                    # Decode and queue audio only if not stopped
                    audio_data = base64.b64decode(data["audio"])
                    if not self._stop_requested:  # Double-check before queuing
                        self.audio_queue.put(audio_data)
                        
                        # Call first audio callback if not already called
                        if not self._first_audio_received and self.first_audio_callback:
                            self._first_audio_received = True
                            asyncio.create_task(self.first_audio_callback())
                    
                    # Update last audio time for completion tracking
                    self._last_audio_time = time.time()
                    
                    # NOTE: For multi-voice, Whisper tracking is handled by the audio playback thread
                    # to avoid double-tracking from multiple websocket connections
                
                elif data.get("isFinal"):
                    # This websocket connection has finished
                    self._websockets_completed += 1
                    print(f"🔊 Websocket completed ({self._websockets_completed}/{self._total_websockets})")
                    
                    # Just track completion, no monitoring task needed
                    
                    break
                    
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            print(f"WebSocket audio handling error: {e}")


    async def _wait_for_audio_completion(self):
        """Wait for audio playback to complete in multi-voice mode."""
        try:
            print(f"🔊 Waiting for audio completion...")
            
            # First, wait for all websockets to finish generating chunks OR stop request
            while (self._websockets_completed < self._total_websockets and 
                   not self._stop_requested):
                await asyncio.sleep(0.1)
            
            if self._stop_requested:
                print("🛑 Stopped during chunk generation")
                return
            
            print(f"✅ All audio chunks generated ({self._websockets_completed} websockets)")
            
            # Then wait for queue to drain AND no active playback
            while ((self.audio_queue.qsize() > 0 or self._actively_playing_chunk) and 
                   not self._stop_requested):
                queue_size = self.audio_queue.qsize()
                if queue_size > 0 and queue_size % 10 == 0:
                    print(f"🔊 Audio queue: {queue_size} chunks remaining")
                elif queue_size < 10 and queue_size > 0:
                    print(f"🔊 Audio queue: {queue_size} chunks remaining")
                await asyncio.sleep(0.2)
            
            if self._stop_requested:
                print("🛑 Stopped during playback")
                return
                
            print("✅ Audio playback complete")
                
        except Exception as e:
            print(f"Audio completion wait error: {e}")

    async def _cleanup_multi_voice(self):
        """Clean up multi-voice streaming resources."""
        try:
            # Use the existing cleanup method for consistency
            await self._cleanup_stream()
                
        except Exception as e:
            print(f"Multi-voice cleanup error: {e}")

    async def speak_stream(self, text_generator: AsyncGenerator[str, None]) -> bool:
        """
        Speak streaming text with interruption support.
        
        Args:
            text_generator: Async generator yielding text chunks
            
        Returns:
            bool: True if completed successfully, False if interrupted
        """
        if self.is_streaming:
            await self.stop()
            
        try:
            self._stop_requested = False
            self._interrupted = False
            self._first_audio_received = False
            self._chunks_played = 0  # Reset chunks played counter
            
            # Create a new session for this speaking operation
            self.current_session = self._create_session()
            
            if self.current_session.whisper_tracker and self.track_spoken_content:
                self.current_session.whisper_tracker.start_tracking()
            
            await self._start_streaming_generator(text_generator)
            
            # If we reached here without stop being requested, it completed successfully
            completed_successfully = not self._stop_requested
            self._interrupted = self._stop_requested  # Set interrupted flag based on actual completion
            
            return completed_successfully
            
        except Exception as e:
            print(f"TTS streaming error: {e}")
            return False
        finally:
            await self._cleanup_stream()
    
    def get_precise_playback_position(self) -> float:
        """
        Get precise playback position in seconds (only available with mel-aec).
        
        Returns:
            Current playback position in seconds, or -1 if not available
        """
        if USING_MEL_AEC and self.stream and hasattr(self.stream, 'get_time'):
            try:
                return self.stream.get_time()
            except:
                pass
        return -1.0
    
    def get_buffered_audio_duration(self) -> float:
        """
        Get duration of audio currently buffered for output (mel-aec only).
        
        Returns:
            Buffered duration in seconds, or -1 if not available
        """
        if USING_MEL_AEC and self.stream and hasattr(self.stream, 'get_write_available'):
            try:
                # Estimate based on available write space
                available_frames = self.stream.get_write_available()
                buffered_frames = self.config.buffer_size * 4 - available_frames
                return max(0, buffered_frames) / self.config.sample_rate
            except:
                pass
        return -1.0
    
    def interrupt_with_position(self) -> Tuple[bool, float]:
        """
        Interrupt playback and get the exact position where it was interrupted.
        Only works with mel-aec.
        
        Returns:
            Tuple of (success, position_seconds)
        """
        if USING_MEL_AEC and self.is_playing:
            position = self.get_precise_playback_position()
            print(f"🎯 mel-aec: Interrupting at position {position:.3f}s")
            return True, position
        else:
            # Fallback for PyAudio - no precise position
            return True, -1.0
    
    async def stop(self):
        """Stop current TTS playback immediately and wait for complete shutdown."""
        print("🛑 TTS stop requested")
        self._stop_requested = True
        self._interrupted = True
        
        # Mark current session as interrupted
        if self.current_session:
            self.current_session.was_interrupted = True
            self.current_session.end_time = time.time()
        
        # Clear audio queue immediately and repeatedly
        for _ in range(3):  # Clear multiple times to ensure empty
            self._clear_audio_queue()
            await asyncio.sleep(0.01)  # Small delay between clears
        
        # Close WebSocket
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                print(f"Error closing TTS WebSocket: {e}")
            self.websocket = None
        
        # Cancel audio task and wait for it to finish
        if self.audio_task and not self.audio_task.done():
            self.audio_task.cancel()
            try:
                await self.audio_task
            except asyncio.CancelledError:
                pass
        
        # Don't force close the stream here - let the worker do it gracefully
        # This avoids the PortAudio errors
        
        # Stop audio playback and wait for thread to exit
        await self._stop_audio_playback_async()
        
        self.is_streaming = False
        print("✅ TTS stopped")
    
    def is_currently_playing(self) -> bool:
        """Check if TTS is currently playing (both streaming and audio output)."""
        return (self.is_streaming and 
                self.is_playing and 
                self.playback_thread and 
                self.playback_thread.is_alive() and
                not self._stop_requested)
    
    def has_played_audio(self) -> bool:
        """Check if any audio chunks have been played in the current session."""
        return self._chunks_played > 0
    
    async def _start_streaming(self, text: str):
        """Start streaming a single text."""
        await self._setup_websocket()
        
        # Send text
        if not self._stop_requested:
            message = {
                "text": text,
                "try_trigger_generation": True
            }
            await self.websocket.send(json.dumps(message))
            print(f"📤 Sent to TTS: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        # Signal completion
        if not self._stop_requested:
            await self.websocket.send(json.dumps({"text": ""}))
        
        # Wait for completion
        await self._wait_for_completion()
    
    async def _start_streaming_generator(self, text_generator: AsyncGenerator[str, None]):
        """Start streaming from a text generator."""
        await self._setup_websocket()
        
        text_buffer = ""
        
        async for text_chunk in text_generator:
            if self._stop_requested:
                break
                
            text_buffer += text_chunk
            # Accumulate for Whisper tracking
            if self.current_session:
                self.current_session.generated_text += text_chunk
            
            # Send chunks at natural breaks
            if any(punct in text_buffer for punct in ['.', '!', '?', ',', ';']) or len(text_buffer) > 40:
                if not self._stop_requested:
                    message = {
                        "text": text_buffer,
                        "try_trigger_generation": True
                    }
                    await self.websocket.send(json.dumps(message))
                    print(f"📤 Sent to TTS stream: '{text_buffer.strip()[:50]}{'...' if len(text_buffer.strip()) > 50 else ''}'")
                    text_buffer = ""
        
        # Send remaining text
        if text_buffer.strip() and not self._stop_requested:
            message = {
                "text": text_buffer,
                "try_trigger_generation": True
            }
            await self.websocket.send(json.dumps(message))
            print(f"📤 Sent final TTS chunk: '{text_buffer.strip()[:50]}{'...' if len(text_buffer.strip()) > 50 else ''}'")
        
        # Signal completion
        if not self._stop_requested:
            await self.websocket.send(json.dumps({"text": ""}))
        
        # Wait for completion
        await self._wait_for_completion()
    
    async def _setup_websocket(self):
        """Setup WebSocket connection and audio playback."""
        # Clear any leftover audio
        self._clear_audio_queue()
        
        # Start audio playback
        self._start_audio_playback()
        
        # Connect to WebSocket
        uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{self.config.voice_id}/stream-input"
        params = f"?model_id={self.config.model_id}&output_format={self.config.output_format}"
        
        self.websocket = await websockets.connect(uri + params)
        self.is_streaming = True
        
        # Send initial configuration
        initial_message = {
            "text": " ",
            "voice_settings": {
                "speed": self.config.speed,
                "stability": self.config.stability,
                "similarity_boost": self.config.similarity_boost
            },
            "xi_api_key": self.config.api_key
        }
        await self.websocket.send(json.dumps(initial_message))
        
        # Start audio response handler
        self.audio_task = asyncio.create_task(self._handle_audio_responses())
    
    async def _handle_audio_responses(self):
        """Handle incoming audio responses from ElevenLabs."""
        try:
            async for message in self.websocket:
                if self._stop_requested:
                    break
                
                if message is None:
                    continue
                
                try:
                    data = json.loads(message)
                except (json.JSONDecodeError, TypeError):
                    continue
                
                if "audio" in data and data["audio"] and not self._stop_requested:
                    try:
                        audio_data = base64.b64decode(data["audio"])
                        if len(audio_data) > 0 and not self._stop_requested:  # Double-check before queuing
                            self.audio_queue.put(audio_data)
                            
                            # Call first audio callback if not already called
                            if not self._first_audio_received and self.first_audio_callback:
                                self._first_audio_received = True
                                asyncio.create_task(self.first_audio_callback())
                    except Exception as e:
                        print(f"Failed to decode TTS audio: {e}")
                        continue
                
                if data.get("isFinal", False):
                    print("🏁 TTS generation completed")
                    break
                    
        except asyncio.CancelledError:
            print("🔇 TTS audio handler cancelled")
        except Exception as e:
            if not self._stop_requested:
                print(f"TTS audio handling error: {e}")
    
    async def _wait_for_completion(self):
        """Wait for audio processing to complete."""
        if self.audio_task and not self._stop_requested:
            try:
                await asyncio.wait_for(self.audio_task, timeout=15.0)
            except asyncio.TimeoutError:
                print("⏰ TTS audio task timed out")
                if self.audio_task:
                    self.audio_task.cancel()
            except asyncio.CancelledError:
                pass
        
        # Wait for queue to drain
        if not self._stop_requested:
            while not self.audio_queue.empty() and not self._stop_requested:
                await asyncio.sleep(0.1)
    
    def _start_audio_playback(self):
        """Start the audio playback thread."""
        if not self.is_playing:
            # Clean up any existing stream first
            if self.stream:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                except Exception:
                    pass
                self.stream = None
            
            # Open new stream
            try:
                stream_kwargs = {
                    'format': paInt16,
                    'channels': 1,
                    'rate': self.config.sample_rate,
                    'output': True,
                    'frames_per_buffer': self.config.buffer_size
                }
                
                # Add output device index if specified (resolve from name if needed)
                output_device_index = None
                if self.config.output_device_name is not None:
                    output_device_index = find_audio_device_by_name(self.config.output_device_name)
                    if output_device_index is not None:
                        stream_kwargs['output_device_index'] = output_device_index
                        print(f"🔊 Using output device: '{self.config.output_device_name}' (index {output_device_index})")
                
                self.stream = self.p.open(**stream_kwargs)
                
                self.is_playing = True
                self.playback_thread = threading.Thread(target=self._audio_playback_worker)
                self.playback_thread.daemon = True
                self.playback_thread.start()
                
            except Exception as e:
                print(f"Failed to start audio playback: {e}")
                self.is_playing = False
    
    def _stop_audio_playback(self):
        """Stop the audio playback thread (sync version)."""
        if self.is_playing:
            self.is_playing = False
            
            if self.playback_thread:
                self.playback_thread.join(timeout=2.0)
                
            if self.stream:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                except Exception as e:
                    print(f"Error stopping audio stream: {e}")
                self.stream = None
    
    async def _stop_audio_playback_async(self):
        """Stop the audio playback thread and wait for complete shutdown."""
        if not self.is_playing:
            return  # Already stopped
        
        print("🔄 Stopping audio playback...")
        self.is_playing = False
        
        # Wait for playback thread to exit in a non-blocking way
        if self.playback_thread and self.playback_thread.is_alive():
            # Use asyncio to wait for thread without blocking
            max_wait_time = 0.5  # Reduced to 500ms for faster stopping
            start_time = asyncio.get_event_loop().time()
            
            while (self.playback_thread.is_alive() and 
                   (asyncio.get_event_loop().time() - start_time) < max_wait_time):
                await asyncio.sleep(0.01)  # Check every 10ms
            
            if self.playback_thread.is_alive():
                print("⚠️ Audio thread didn't exit cleanly, forcing...")
                # Thread should exit on its own when is_playing = False
        
        # Clean up audio stream if it still exists
        if self.stream:
            try:
                # Check if stream is still active before trying to stop
                if hasattr(self.stream, '_stream') and self.stream._stream:
                    self.stream.stop_stream()
                    self.stream.close()
                    print("🔇 Audio stream closed")
            except Exception as e:
                # This is expected if stream was already closed in stop()
                if "Stream not open" not in str(e):
                    print(f"Error stopping audio stream: {e}")
            self.stream = None
        
        self.playback_thread = None
        print("✅ Audio playback fully stopped")
    
    def _audio_playback_worker(self):
        """Worker thread for audio playback with responsive stop handling and audio capture."""
        print("🎵 Audio playback worker started")
        
        # Capture the session this worker is associated with
        worker_session = self.current_session
        
        while self.is_playing and not self._stop_requested:
            try:
                # Use very short timeout for more responsive stopping
                audio_chunk = self.audio_queue.get(timeout=0.05)  # 50ms timeout
                
                # Double-check stop status and session before writing
                if audio_chunk and self.stream and not self._stop_requested and self.is_playing:
                    # Also check if this worker's session is still the current one
                    if worker_session != self.current_session:
                        print(f"🛑 Audio worker detected session change, stopping playback")
                        self.audio_queue.task_done()
                        break
                        
                    try:
                        # Mark that we're actively playing
                        self._actively_playing_chunk = True
                        
                        # Split chunk into smaller pieces for more responsive stopping
                        # Play in ~100ms segments (about 4410 samples at 22050Hz = 8820 bytes)
                        segment_size = min(8820, len(audio_chunk))
                        
                        offset = 0
                        while offset < len(audio_chunk):
                            # Check for interruption before each segment
                            if self._stop_requested or not self.is_playing or worker_session != self.current_session:
                                print("🛑 Stopping mid-chunk playback")
                                break
                                
                            # Calculate actual segment size for this iteration
                            current_segment_size = min(segment_size, len(audio_chunk) - offset)
                            audio_segment = audio_chunk[offset:offset + current_segment_size]
                            
                            # Play the segment with error handling
                            try:
                                if self.stream and self.stream._stream:
                                    self.stream.write(audio_segment)
                                    # Track that we've played audio
                                    if offset == 0:  # First segment of this chunk
                                        self._chunks_played += 1
                            except Exception as e:
                                # Stream was closed or error occurred
                                if "Stream not open" not in str(e) and "-9986" not in str(e):
                                    print(f"Audio segment write error: {e}")
                                break
                            offset += current_segment_size
                        
                        # Mark that we're done playing
                        self._actively_playing_chunk = False
                        
                        # Only do Whisper tracking if we played the full chunk
                        if offset >= len(audio_chunk) and worker_session and worker_session.whisper_tracker and self.track_spoken_content:
                            try:
                                # Convert bytes to numpy array for Whisper
                                # ElevenLabs returns PCM 22050Hz, Whisper needs 16kHz
                                audio_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
                                
                                # Proper resampling from 22050Hz to 16000Hz using scipy
                                if len(audio_np) > 0:
                                    # Use scipy's high-quality resampling with anti-aliasing
                                    original_rate = self.config.sample_rate  # 22050
                                    target_rate = 16000
                                    
                                    # Calculate target number of samples
                                    num_samples = int(len(audio_np) * target_rate / original_rate)
                                    if num_samples > 0:
                                        # Use scipy.signal.resample for high-quality resampling
                                        audio_16k = signal.resample(audio_np, num_samples)
                                        # Only add to tracker if it's still the current session's tracker
                                        if worker_session == self.current_session:
                                            worker_session.whisper_tracker.add_audio_chunk(audio_16k.astype(np.float32))
                            except Exception as e:
                                print(f"Whisper audio capture error: {e}")
                        
                        self.audio_queue.task_done()
                    except Exception as e:
                        print(f"Audio write error: {e}")
                        self.audio_queue.task_done()
                        break
                elif audio_chunk:
                    # If stopped, mark task as done but don't play
                    self.audio_queue.task_done()
                    
            except queue.Empty:
                # Check stop condition more frequently during silence
                if self._stop_requested or not self.is_playing:
                    break
                continue
            except Exception as e:
                print(f"Audio playback error: {e}")
                break
        
        print("🔇 Audio playback worker exited")
    
    def _clear_audio_queue(self):
        """Clear the audio queue thoroughly."""
        cleared_count = 0
        # Try multiple approaches to ensure queue is empty
        
        # First, use qsize to determine how many items to clear
        initial_size = self.audio_queue.qsize()
        
        # Clear using get_nowait
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                cleared_count += 1
            except queue.Empty:
                break
        
        # Double-check with qsize and clear any remaining
        remaining = self.audio_queue.qsize()
        for _ in range(remaining):
            try:
                self.audio_queue.get_nowait()
                cleared_count += 1
            except queue.Empty:
                break
        
        # Create a new queue to ensure it's truly empty
        if self.audio_queue.qsize() > 0:
            self.audio_queue = queue.Queue()
            cleared_count = initial_size  # Assume we cleared everything
        
        if cleared_count > 0:
            print(f"🗑️ Cleared {cleared_count} audio chunks from queue")
    
    async def _cleanup_stream(self):
        """Clean up streaming resources."""
        self.is_streaming = False
        
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception:
                pass
            self.websocket = None
        
        if self.audio_task and not self.audio_task.done():
            self.audio_task.cancel()
            try:
                await self.audio_task
            except asyncio.CancelledError:
                pass
            self.audio_task = None
        
        self._stop_audio_playback()
        
        # Stop Whisper tracking and collect spoken content
        if self.current_session and self.current_session.whisper_tracker and self.track_spoken_content:
            try:
                self.current_session.current_spoken_content = self.current_session.whisper_tracker.stop_tracking()
                print(f"🎙️ Captured {len(self.current_session.current_spoken_content)} spoken segments")
            except Exception as e:
                print(f"Error stopping Whisper tracking: {e}")
    
    def get_spoken_content(self) -> List[SpokenContent]:
        """
        Get the actual spoken content from the last TTS session.
        
        Returns:
            List of SpokenContent objects representing what was actually spoken
        """
        if not self.current_session:
            return []
        return self.current_session.current_spoken_content.copy()
    
    def get_spoken_text(self) -> str:
        """
        Get the actual spoken text as a single string.
        
        Returns:
            Combined text of what was actually spoken
        """
        if not self.current_session:
            return ""
        return " ".join(content.text for content in self.current_session.current_spoken_content)
    
    def get_spoken_text_heuristic(self) -> str:
        """
        Get spoken text using character-count heuristic from original generated text.
        Uses Whisper's character count but preserves exact LLM vocabulary/formatting.
        Only applies heuristic when there was an interruption - otherwise returns full text.
        
        Returns:
            Portion of original generated text that was likely spoken
        """
        if not self.current_session or not self.current_session.generated_text:
            return ""
        
        # If TTS was interrupted, use Whisper character count heuristic
        if self._interrupted and self.current_session.current_spoken_content:
            # Get total character count from Whisper
            whisper_char_count = sum(len(content.text) for content in self.current_session.current_spoken_content)
            
            # Debug logging
            whisper_texts = [content.text for content in self.current_session.current_spoken_content]
            print(f"🔍 [HEURISTIC DEBUG] Whisper segments: {len(whisper_texts)}")
            print(f"🔍 [HEURISTIC DEBUG] Whisper texts: {whisper_texts}")
            print(f"🔍 [HEURISTIC DEBUG] Whisper char count: {whisper_char_count}")
            print(f"🔍 [HEURISTIC DEBUG] Generated text length: {len(self.current_session.generated_text)}")
            print(f"🔍 [HEURISTIC DEBUG] Generated text: '{self.current_session.generated_text[:100]}...'")
            
            if whisper_char_count == 0:
                print("🔍 [HEURISTIC DEBUG] Returning empty - no Whisper characters")
                return ""
            
            # Use spoken_text_for_tts as the reference since that's what was actually sent to TTS
            reference_text = self.current_session.spoken_text_for_tts or self.current_session.generated_text
            
            # Find this position in the reference text
            target_position = min(whisper_char_count, len(reference_text))
            
            # Round up to nearest complete word boundary
            if target_position >= len(self.current_session.generated_text):
                # If Whisper captured more than generated (shouldn't happen), return full text
                return self.current_session.generated_text
            
            # Find the end of the word at target position
            word_end_position = target_position
            
            # If we're in the middle of a word, find the end
            while (word_end_position < len(self.current_session.generated_text) and 
                   self.current_session.generated_text[word_end_position] not in [' ', '.', ',', '!', '?', ';', ':', '\n']):
                word_end_position += 1
            
            # Check if we're in the middle of an emotive marker (between asterisks)
            result_text = self.current_session.generated_text[:word_end_position]
            asterisk_count = result_text.count('*')
            
            # If odd number of asterisks, we're in the middle of an emotive marker
            if asterisk_count % 2 == 1:
                # Find the closing asterisk
                close_pos = self.current_session.generated_text.find('*', word_end_position)
                if close_pos != -1:
                    word_end_position = close_pos + 1
            
            result = self.current_session.generated_text[:word_end_position].strip()
            print(f"🔍 [HEURISTIC DEBUG] Returning heuristic result: '{result}' ({len(result)} chars)")
            return result
        
        else:
            # TTS completed successfully - return full generated text
            return self.current_session.generated_text
    
    def get_generated_vs_spoken(self) -> Dict[str, str]:
        """
        Compare generated text vs actually spoken text.
        
        Returns:
            Dictionary with 'generated', 'spoken_whisper', and 'spoken_heuristic' keys
        """
        if not self.current_session:
            return {"generated": "", "spoken_whisper": "", "spoken_heuristic": ""}
            
        return {
            "generated": self.current_session.generated_text,
            "spoken_whisper": self.get_spoken_text(),
            "spoken_heuristic": self.get_spoken_text_heuristic()
        }
    
    def was_fully_spoken(self) -> bool:
        """
        Check if the generated text was fully spoken (not interrupted).
        Uses heuristic approach based on character count.
        
        Returns:
            True if likely fully spoken, False if interrupted
        """
        if not self.current_session:
            return False
            
        spoken_heuristic = self.get_spoken_text_heuristic()
        if not self.current_session.generated_text or not spoken_heuristic:
            return False
        
        # If heuristic captured at least 90% of generated text, consider it fully spoken
        return len(spoken_heuristic) >= (len(self.current_session.generated_text) * 0.9)

    async def cleanup(self):
        """Clean up all resources."""
        await self.stop()
        
        if self.p:
            try:
                self.p.terminate()
            except Exception as e:
                print(f"Error terminating PyAudio: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        if hasattr(self, 'p') and self.p:
            try:
                self.p.terminate()
            except Exception:
                pass

# Convenience function for quick usage
async def speak_text(text: str, api_key: str, voice_id: str = "T2KZm9rWPG5TgXTyjt7E") -> bool:
    """
    Convenience function to speak text with default settings.
    
    Args:
        text: Text to speak
        api_key: ElevenLabs API key
        voice_id: Voice ID to use
        
    Returns:
        bool: True if completed successfully, False if interrupted
    """
    config = TTSConfig(api_key=api_key, voice_id=voice_id)
    tts = AsyncTTSStreamer(config)
    
    try:
        result = await tts.speak_text(text)
        return result
    finally:
        await tts.cleanup()

# Example usage
async def main():
    """Example usage of the TTS module."""
    api_key = input("Enter your ElevenLabs API key: ").strip()
    
    config = TTSConfig(api_key=api_key)
    tts = AsyncTTSStreamer(config)
    
    try:
        print("🎤 Starting TTS test...")
        
        # Test single text
        result = await tts.speak_text("Hello! This is a test of the async TTS module.")
        print(f"✅ Completed: {result}")
        
        # Test interruption
        print("\n🛑 Testing interruption...")
        task = asyncio.create_task(
            tts.speak_text("This is a longer message that we will interrupt before it finishes playing completely.")
        )
        
        # Interrupt after 2 seconds
        await asyncio.sleep(2.0)
        await tts.stop()
        
        result = await task
        print(f"✅ Interrupted result: {result}")
        
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    finally:
        await tts.cleanup()

def find_audio_device_by_name(device_name: str) -> Optional[int]:
    """Find audio output device index by name (partial match)."""
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
            if info['maxOutputChannels'] > 0:  # Only check output devices
                if device_name_lower in info['name'].lower():
                    print(f"🎯 Found audio device: '{info['name']}' (index {i}) for name '{device_name}'")
                    return i
        
        print(f"⚠️ No audio device found matching '{device_name}'")
        print("Available output devices:")
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info['maxOutputChannels'] > 0:
                print(f"  - {info['name']}")
        return None
        
    finally:
        p.terminate()

def list_audio_output_devices():
    """List all available audio output devices with their indices."""
    p = pyaudio.PyAudio()
    print("\n🔊 Available Audio Output Devices:")
    print("=" * 50)
    
    output_devices = []
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxOutputChannels'] > 0:  # Only show output devices
            output_devices.append((i, info))
            default_marker = " (DEFAULT)" if i == p.get_default_output_device_info()['index'] else ""
            print(f"Index {i}: {info['name']}{default_marker}")
            print(f"   Channels: {info['maxOutputChannels']}")
            print(f"   Sample Rate: {info['defaultSampleRate']}")
            print()
    
    p.terminate()
    print(f"Total output devices found: {len(output_devices)}")
    return output_devices

if __name__ == "__main__":
    asyncio.run(main()) 