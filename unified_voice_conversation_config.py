#!/usr/bin/env python3
"""
Unified Voice Conversation System with YAML Configuration
Integrates modular STT and TTS systems with YAML-based configuration management.
"""

import asyncio
import re
import logging
import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, AsyncGenerator
from datetime import datetime, timedelta
from openai import OpenAI
import anthropic

# Import our modular systems and config loader
from async_stt_module import AsyncSTTStreamer, STTEventType, STTResult
from async_tts_module import AsyncTTSStreamer
from config_loader import load_config, VoiceAIConfig

@dataclass 
class ConversationState:
    """Tracks the current state of the conversation."""
    is_active: bool = False
    is_processing_llm: bool = False
    is_speaking: bool = False
    last_utterance_time: Optional[datetime] = None
    utterance_buffer: List[Dict[str, str]] = field(default_factory=list)  # Now stores {"text": str, "speaker": str}
    conversation_history: List[Dict[str, str]] = field(default_factory=list)  # Now includes "speaker" field
    current_speaker: Optional[int] = None
    current_speaker_name: Optional[str] = None
    connected_speakers: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # Track all connected speakers

class UnifiedVoiceConversation:
    """Unified voice conversation system with YAML configuration."""
    
    def __init__(self, config: VoiceAIConfig):
        self.config = config
        self.state = ConversationState()
        
        # Initialize LLM clients
        self.openai_client = None
        self.anthropic_client = None
        
        if config.conversation.llm_provider == "openai":
            self.openai_client = OpenAI(api_key=config.conversation.openai_api_key)
        elif config.conversation.llm_provider == "anthropic":
            if not config.conversation.anthropic_api_key:
                raise ValueError("Anthropic API key is required when using Anthropic provider")
            self.anthropic_client = anthropic.Anthropic(api_key=config.conversation.anthropic_api_key)
        
        # Initialize STT system
        self.stt = AsyncSTTStreamer(config.stt)
        
        # Initialize TTS system
        self.tts = AsyncTTSStreamer(config.tts)
        
        # Set up STT callbacks
        self._setup_stt_callbacks()
        
        # Apply logging configuration
        self._setup_logging()
        
        # Create llm_logs directory
        self.llm_logs_dir = Path("llm_logs")
        self.llm_logs_dir.mkdir(exist_ok=True)
        
        # Create conversation_logs directory and initialize conversation log
        self.conversation_logs_dir = Path("conversation_logs")
        self.conversation_logs_dir.mkdir(exist_ok=True)
        
        # Initialize conversation log file
        self._init_conversation_log()
        
        # Load conversation history if specified
        self._load_history_file()
        
        # Log any loaded history to the conversation log
        self._log_loaded_history()
        
        # Conversation management task
        self.conversation_task = None
    
    def _init_conversation_log(self):
        """Initialize conversation log file with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.conversation_log_file = self.conversation_logs_dir / f"conversation_{timestamp}.md"
        
        # Write header
        with open(self.conversation_log_file, 'w', encoding='utf-8') as f:
            f.write(f"# Conversation Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        print(f"📝 Conversation logging to: {self.conversation_log_file}")
    
    def _log_conversation_turn(self, role: str, content: str, speaker_name: str = None):
        """Log a conversation turn in history file format.
        
        Args:
            role: 'user' or 'assistant'
            content: The message content
            speaker_name: Optional speaker name (for user messages)
        """
        try:
            # Determine participant name based on role
            if role == "user":
                if speaker_name and speaker_name != "User":
                    participant = speaker_name
                else:
                    participant = "H"
            else:  # assistant
                # Use the AI participant name from prefill config if available
                participant = "Claude"
                if (hasattr(self.config.conversation, 'prefill_participants') and 
                    self.config.conversation.prefill_participants and 
                    len(self.config.conversation.prefill_participants) > 1):
                    participant = self.config.conversation.prefill_participants[1]
            
            # Append to conversation log
            with open(self.conversation_log_file, 'a', encoding='utf-8') as f:
                f.write(f"{participant}: {content}\n\n")
                
        except Exception as e:
            print(f"⚠️ Failed to log conversation turn: {e}")
            self.logger.error(f"Failed to log conversation turn: {e}")
    
    def _log_loaded_history(self):
        """Log any existing conversation history to the conversation log."""
        if not self.state.conversation_history:
            return
            
        try:
            with open(self.conversation_log_file, 'a', encoding='utf-8') as f:
                f.write("## Loaded History\n\n")
                
            # Log each message from loaded history
            for msg in self.state.conversation_history:
                self._log_conversation_turn(msg["role"], msg["content"])
                
            with open(self.conversation_log_file, 'a', encoding='utf-8') as f:
                f.write("## New Conversation\n\n")
                
            print(f"📜 Logged {len(self.state.conversation_history)} history messages to conversation log")
            
        except Exception as e:
            print(f"⚠️ Failed to log loaded history: {e}")
            self.logger.error(f"Failed to log loaded history: {e}")
    
    def _setup_logging(self):
        """Configure logging based on configuration."""
        
        # Set logging level
        level = getattr(logging, self.config.logging.level.upper(), logging.INFO)
        logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
        
        self.logger = logging.getLogger(__name__)
        
        # Store logging preferences
        self.show_interim = self.config.logging.show_interim_results
        self.show_tts_chunks = self.config.logging.show_tts_chunks
        self.show_audio_debug = self.config.logging.show_audio_debug
    
    def _generate_log_filename(self, log_type: str, timestamp: float = None) -> str:
        """Generate a unique filename for LLM logs."""
        if timestamp is None:
            timestamp = time.time()
        
        dt = datetime.fromtimestamp(timestamp)
        formatted_time = dt.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        
        return f"{formatted_time}_{log_type}.json"
    
    def _log_llm_request(self, messages: List[Dict[str, str]], model: str, timestamp: float, provider: str = None) -> str:
        """Log LLM request to a file and return the filename."""
        filename = self._generate_log_filename("request", timestamp)
        filepath = self.llm_logs_dir / filename
        
        request_data = {
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp).isoformat(),
            "provider": provider or self.config.conversation.llm_provider,
            "model": model,
            "messages": messages,
            "max_tokens": self.config.conversation.max_tokens,
            "stream": True,
            "conversation_mode": self.config.conversation.conversation_mode,
            "request_type": "llm_completion"
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(request_data, f, indent=2, ensure_ascii=False)
            print(f"📝 LLM request logged: {filename}")
        except Exception as e:
            print(f"❌ Failed to log LLM request: {e}")
        
        return filename
    
    def _log_llm_response(self, response_content: str, request_filename: str, timestamp: float, 
                         was_interrupted: bool = False, error: str = None, provider: str = None) -> str:
        """Log LLM response to a file and return the filename."""
        filename = self._generate_log_filename("response", timestamp)
        filepath = self.llm_logs_dir / filename
        
        response_data = {
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp).isoformat(),
            "provider": provider or self.config.conversation.llm_provider,
            "request_file": request_filename,
            "response_content": response_content,
            "was_interrupted": was_interrupted,
            "error": error,
            "content_length": len(response_content) if response_content else 0,
            "response_type": "llm_completion"
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(response_data, f, indent=2, ensure_ascii=False)
            print(f"📝 LLM response logged: {filename}")
        except Exception as e:
            print(f"❌ Failed to log LLM response: {e}")
        
        return filename

    def _convert_to_prefill_format(self, messages: List[Dict[str, str]]) -> tuple[str, str]:
        """Convert chat messages to prefill format.
        Returns (user_message, assistant_message_prefix).
        """
        user_message = self.config.conversation.prefill_user_message
        
        # Build conversation turns from history
        conversation_turns = []
        human_name = self.config.conversation.prefill_participants[0]  # Default: 'H'
        ai_name = self.config.conversation.prefill_participants[1]     # Default: 'Claude'
        
        for msg in messages:
            if msg["role"] == "system":
                continue  # Skip system messages in prefill mode
            elif msg["role"] == "user":
                conversation_turns.append(f"{human_name}: {msg['content']}")
            elif msg["role"] == "assistant":
                conversation_turns.append(f"{ai_name}: {msg['content']}")
        
        # Join turns with double newlines
        assistant_content = "\n\n".join(conversation_turns)
        if assistant_content:
            assistant_content += f"\n\n{ai_name}:"
        else:
            assistant_content = f"{ai_name}:"
            
        return user_message, assistant_content

    def _convert_from_prefill_format(self, prefill_response: str) -> str:
        """Extract the actual response from prefill format.
        Removes the participant prefix if present.
        """
        ai_name = self.config.conversation.prefill_participants[1]  # Default: 'Claude'
        
        # Remove leading AI name prefix if present
        if prefill_response.startswith(f"{ai_name}:"):
            return prefill_response[len(f"{ai_name}:"):].strip()
        
        return prefill_response.strip()

    def _parse_history_file(self, file_path: str) -> List[Dict[str, str]]:
        """Parse history file and convert to message format.
        
        Expects format:
        H: human message
        
        Claude: assistant response
        
        H: next human message
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            messages = []
            current_role = None
            current_content = []
            
            for line in content.split('\n'):
                line = line.strip()
                
                if line.startswith('H: '):
                    # Save previous message if exists
                    if current_role and current_content:
                        messages.append({
                            "role": "user" if current_role == "H" else "assistant",
                            "content": '\n'.join(current_content).strip()
                        })
                    
                    # Start new human message
                    current_role = "H"
                    current_content = [line[3:]]  # Remove "H: " prefix
                    
                elif line.startswith('Claude: '):
                    # Save previous message if exists
                    if current_role and current_content:
                        messages.append({
                            "role": "user" if current_role == "H" else "assistant",
                            "content": '\n'.join(current_content).strip()
                        })
                    
                    # Start new assistant message
                    current_role = "Claude"
                    current_content = [line[8:]]  # Remove "Claude: " prefix
                    
                elif line and current_role:
                    # Continue current message
                    current_content.append(line)
                # Skip empty lines or lines without a role
            
            # Add final message if exists
            if current_role and current_content:
                messages.append({
                    "role": "user" if current_role == "H" else "assistant",
                    "content": '\n'.join(current_content).strip()
                })
            
            return messages
            
        except Exception as e:
            print(f"❌ Failed to parse history file {file_path}: {e}")
            return []

    def _load_history_file(self):
        """Load conversation history from file if specified."""
        if not self.config.conversation.history_file:
            return
            
        file_path = self.config.conversation.history_file
        print(f"📜 Loading conversation history from: {file_path}")
        
        history_messages = self._parse_history_file(file_path)
        
        if history_messages:
            self.state.conversation_history.extend(history_messages)
            print(f"✅ Loaded {len(history_messages)} messages from history file")
            print(f"📊 Conversation context: {sum(len(msg['content']) for msg in history_messages)} characters")
        else:
            print(f"⚠️  No messages loaded from history file")

    def _setup_stt_callbacks(self):
        """Set up callbacks for STT events."""
        
        # Handle completed utterances
        self.stt.on(STTEventType.UTTERANCE_COMPLETE, self._on_utterance_complete)
        
        # Handle interim results (for interruption detection)
        self.stt.on(STTEventType.INTERIM_RESULT, self._on_interim_result)
        
        # Handle speech events
        self.stt.on(STTEventType.SPEECH_STARTED, self._on_speech_started)
        self.stt.on(STTEventType.SPEECH_ENDED, self._on_speech_ended)
        
        # Handle speaker changes
        self.stt.on(STTEventType.SPEAKER_CHANGE, self._on_speaker_change)
        
        # Handle errors
        self.stt.on(STTEventType.ERROR, self._on_error)
    
    async def start_conversation(self):
        """Start the voice conversation system."""
        print("🎙️ Starting Unified Voice Conversation System (YAML Config)")
        print("=" * 60)
        
        # Show configuration summary
        self._show_config_summary()
        
        try:
            # Start STT
            print("🎤 Starting speech recognition...")
            if not await self.stt.start_listening():
                print("❌ Failed to start STT")
                return False
            
            self.state.is_active = True
            
            # Start conversation management
            self.conversation_task = asyncio.create_task(self._conversation_manager())
            
            print("✅ Conversation system active!")
            print("💡 Tips:")
            print("   - Speak naturally and pause when done")
            print("   - You can interrupt the AI by speaking while it talks")
            print("   - Press Ctrl+C to exit")
            print()
            
            # Keep running until stopped
            while self.state.is_active:
                await asyncio.sleep(1)
            
            return True
            
        except Exception as e:
            print(f"❌ Error starting conversation: {e}")
            return False
    
    def _show_config_summary(self):
        """Show a summary of the loaded configuration."""
        print(f"📋 Configuration Summary:")
        print(f"   🎯 Voice ID: {self.config.conversation.voice_id}")
        print(f"   🎤 STT Model: {self.config.stt.model} ({self.config.stt.language})")
        print(f"   🔊 TTS Model: {self.config.tts.model_id}")
        print(f"   🤖 LLM Model: {self.config.conversation.llm_model}")
        print(f"   ⏱️  Pause Threshold: {self.config.conversation.pause_threshold}s")
        print(f"   📝 Min Words: {self.config.conversation.min_words_for_submission}")
        print(f"   🔇 Interruption Confidence: {self.config.conversation.interruption_confidence}")
        
        if self.config.development.enable_debug_mode:
            print(f"   🐛 Debug Mode: Enabled")
        
        print()
    
    async def stop_conversation(self):
        """Stop the conversation system."""
        print("🛑 Stopping conversation...")
        self.state.is_active = False
        
        # Cancel conversation management
        if self.conversation_task and not self.conversation_task.done():
            self.conversation_task.cancel()
            try:
                await self.conversation_task
            except asyncio.CancelledError:
                pass
        
        # Stop TTS if speaking
        if self.state.is_speaking:
            await self.tts.stop()
        
        # Stop STT
        await self.stt.stop_listening()
        
        print("✅ Conversation stopped")
    
    async def _on_utterance_complete(self, result: STTResult):
        """Handle completed utterances from STT."""
        speaker_info = f" (Speaker {result.speaker_id})" if result.speaker_id is not None else ""
        print(f"🎯 Final{speaker_info}: {result.text}")
        
        # Check for interruption at ANY stage of AI response
        interrupted = False
        
        if self.tts.is_currently_playing():
            print(f"🛑 Interrupting TTS playback with: {result.text}")
            await self.tts.stop()
            interrupted = True
            
        elif self.state.is_processing_llm:
            print(f"🛑 Interrupting LLM generation with: {result.text}")
            # Cancel any ongoing LLM tasks (will be handled in stream_llm_to_tts)
            interrupted = True
            
        elif self.state.is_speaking:
            print(f"🛑 Interrupting TTS setup with: {result.text}")
            # Stop any TTS preparation
            await self.tts.stop()
            interrupted = True
        
        if interrupted:
            # Clear ALL state flags to ensure next utterance processes
            self.state.is_speaking = False
            self.state.is_processing_llm = False
            print("🔄 All AI processing interrupted and state cleared")
            print(f"🔄 Utterance buffer now has: {len(self.state.utterance_buffer)} items")
            
            # Force immediate processing of the interrupting utterance
            if len(self.state.utterance_buffer) > 0:
                print("🚀 Force processing interrupting utterance immediately")
                asyncio.create_task(self._force_process_buffer())
        
        # Add to utterance buffer
        speaker_name = result.speaker_name or f"Speaker_{result.speaker_id}" if result.speaker_id is not None else "Unknown"
        self.state.utterance_buffer.append({"text": result.text, "speaker": speaker_name})
        self.state.last_utterance_time = result.timestamp
        self.state.current_speaker = result.speaker_id
        self.state.current_speaker_name = speaker_name
        
        print(f"📝 Added to buffer: '{result.text}' at {result.timestamp}")
        print(f"📊 Buffer size: {len(self.state.utterance_buffer)}, Last time: {self.state.last_utterance_time}")
        
        if self.config.development.enable_debug_mode:
            self.logger.debug(f"Utterance added to buffer. Buffer size: {len(self.state.utterance_buffer)}")
    
    async def _on_interim_result(self, result: STTResult):
        """Handle interim results for showing progress."""
        # Interruption is now handled only in _on_utterance_complete for final utterances
        # This is just for showing interim results
        
        # Show interim results based on configuration
        if self.show_interim:
            print(f"💭 Interim: {result.text}")
    
    async def _on_speech_started(self, data):
        """Handle speech start events."""
        # Just show detection, don't interrupt yet - wait for complete utterance
        if self.tts.is_currently_playing():
            print("🚫 Speech detected during TTS - waiting for complete utterance")
        elif self.state.is_speaking:
            print("🚫 Speech detected during conversation processing")
        
        if self.show_audio_debug:
            self.logger.debug("Speech started event received")
    
    async def _on_speech_ended(self, data):
        """Handle speech end events."""
        print("🔇 Speech ended")
        self.state.last_utterance_time = data['timestamp']
        
        if self.show_audio_debug:
            self.logger.debug("Speech ended event received")
    
    async def _on_speaker_change(self, data):
        """Handle speaker change events."""
        speaker_id = data['speaker_id']
        speaker_name = data.get('speaker_name', f"Speaker {speaker_id}")
        print(f"👤 Speaker changed to: {speaker_name}")
        
        if self.config.development.enable_debug_mode:
            self.logger.debug(f"Speaker change: {speaker_id} -> {speaker_name}")
    
    async def _on_error(self, data):
        """Handle STT errors."""
        error_msg = data['error']
        print(f"❌ STT Error: {error_msg}")
        self.logger.error(f"STT Error: {error_msg}")
        
        # Check if this is a connection error that might require restart
        if "Connection send failed" in error_msg or "ConnectionClosed" in error_msg:
            print("🔄 Detected connection failure, attempting to restart STT...")
            try:
                # Try to restart STT connection
                await self.stt.stop_listening()
                await asyncio.sleep(0.01)  # Minimal pause
                success = await self.stt.start_listening()
                if success:
                    print("✅ STT connection restarted successfully")
                else:
                    print("❌ Failed to restart STT connection")
                    await self._speak_text("I'm having trouble with speech recognition. Please try again.")
            except Exception as e:
                print(f"❌ Error restarting STT: {e}")
                self.logger.error(f"STT restart error: {e}")
    
    async def _conversation_manager(self):
        """Manages conversation flow and LLM submission decisions."""
        print("🔄 Conversation manager started")
        last_debug_time = datetime.now()
        
        while self.state.is_active:
            try:
                await asyncio.sleep(0.01)  # Check every 10ms for maximum responsiveness
                
                # Show debug info every 5 seconds if we have data but no submission
                now = datetime.now()
                if (now - last_debug_time).total_seconds() >= 5.0:
                    if self.state.utterance_buffer and self.state.last_utterance_time:
                        combined_transcript = " ".join(msg['text'] for msg in self.state.utterance_buffer).strip()
                        word_count = len(combined_transcript.split()) if combined_transcript else 0
                        time_since_speech = now - self.state.last_utterance_time
                        
                        print(f"🐛 Debug: Buffer='{combined_transcript}' ({word_count} words), "
                              f"Time since speech: {time_since_speech.total_seconds():.1f}s, "
                              f"Processing: {self.state.is_processing_llm}, "
                              f"Speaking: {self.state.is_speaking}")
                    last_debug_time = now
                
                # Skip if currently processing or speaking
                if (self.state.is_processing_llm or 
                    self.state.is_speaking or 
                    not self.state.last_utterance_time):
                    # Debug: Show why we're skipping (but less frequently)
                    if self.state.utterance_buffer and (now - last_debug_time).total_seconds() >= 2.0:
                        skip_reason = []
                        if self.state.is_processing_llm:
                            skip_reason.append("processing_llm")
                        if self.state.is_speaking:
                            skip_reason.append("speaking")
                        if not self.state.last_utterance_time:
                            skip_reason.append("no_utterance_time")
                        
                        if skip_reason:
                            print(f"⏸️ Skipping utterance buffer due to: {', '.join(skip_reason)}")
                            print(f"   Buffer: '{' '.join(msg['text'] for msg in self.state.utterance_buffer)}'")
                    continue
                
                # Calculate time since last speech
                time_since_speech = datetime.now() - self.state.last_utterance_time
                
                # Get combined transcript
                combined_transcript = " ".join(msg['text'] for msg in self.state.utterance_buffer).strip()
                word_count = len(combined_transcript.split()) if combined_transcript else 0
                
                # Decision logic for LLM submission
                should_submit = False
                reason = ""
                
                # IMMEDIATE submission - no waiting for pauses!
                if word_count >= 1:  # Process any speech immediately
                    should_submit = True
                    if self._seems_complete(combined_transcript):
                        reason = "Statement appears complete"
                    else:
                        reason = f"Immediate processing of {word_count} words"
                
                # Submit to LLM if criteria met
                if should_submit:
                    print(f"🧠 Submitting to LLM: {reason}")
                    print(f"📝 Input: {combined_transcript}")
                    
                    # Determine primary speaker from buffer
                    speakers = [msg['speaker'] for msg in self.state.utterance_buffer]
                    primary_speaker = max(set(speakers), key=speakers.count) if speakers else "Unknown"
                    
                    if len(set(speakers)) > 1:
                        print(f"👥 Multiple speakers detected: {set(speakers)}, primary: {primary_speaker}")
                    
                    if self.config.development.enable_debug_mode:
                        self.logger.debug(f"LLM submission triggered: {reason}")
                    
                    await self._process_with_llm(combined_transcript, primary_speaker)
                    
                    # Clear buffer and reset timing
                    self.state.utterance_buffer.clear()
                    self.state.last_utterance_time = None
                    print(f"🧹 Buffer cleared after successful LLM submission")
                else:
                    # Debug: show why we're not submitting
                    if self.state.utterance_buffer and (now - last_debug_time).total_seconds() >= 2.0:
                        print(f"🤔 Not submitting: '{combined_transcript}' ({word_count} words, {time_since_speech.total_seconds():.1f}s ago)")
                        print(f"   Reason: word_count={word_count}, time_since={time_since_speech.total_seconds():.1f}s")
                
            except Exception as e:
                print(f"Conversation manager error: {e}")
                self.logger.error(f"Conversation manager error: {e}")
                import traceback
                traceback.print_exc()
    
    def _seems_complete(self, text: str) -> bool:
        """Heuristic to determine if a statement seems complete."""
        if not text:
            return False
        
        # Check for ending punctuation
        if text.rstrip().endswith(('.', '!', '?')):
            return True
        
        # Check for common complete phrases
        complete_patterns = [
            r'\bthat(?:[\'s]|s) (it|all|everything)\b',
            r'\b(okay|alright|thanks?|thank you)\b\s*$',
            r'\b(done|finished|complete)\b\s*$'
        ]
        
        for pattern in complete_patterns:
            if re.search(pattern, text.lower()):
                return True
        
        return False
    
    async def _process_with_llm(self, user_input: str, speaker_name: str = "User"):
        """Process user input with LLM and speak response."""
        if self.state.is_processing_llm:
            print(f"⚠️ Already processing LLM, skipping: {user_input}")
            return
        
        try:
            self.state.is_processing_llm = True
            print(f"🔄 Starting LLM processing for: '{user_input}' (from {speaker_name})")
            
            # Add to conversation history with speaker information
            self.state.conversation_history.append({
                "role": "user",
                "content": user_input,
                "speaker": speaker_name
            })
            
            # Log conversation turn
            self._log_conversation_turn("user", user_input, speaker_name)
            
            print("🤖 Getting LLM response...")
            
            # Prepare messages with speaker-aware formatting
            system_prompt = self.config.conversation.system_prompt
            if any('speaker' in msg for msg in self.state.conversation_history):
                system_prompt += "\n\nNote: This is a multi-speaker conversation. Speaker names will be indicated."
            
            messages = [{"role": "system", "content": system_prompt}]
            
            # Format conversation history with speaker information
            for msg in self.state.conversation_history[-200:]:  # Keep last 200 exchanges
                content = msg["content"]
                if msg["role"] == "user" and "speaker" in msg and msg["speaker"] not in ["User", "H"]:
                    content = f"[{msg['speaker']}]: {content}"
                messages.append({
                    "role": msg["role"],
                    "content": content
                })
            
            # Stream LLM response to TTS
            await self._stream_llm_to_tts(messages)
            
        except Exception as e:
            print(f"LLM processing error: {e}")
            self.logger.error(f"LLM processing error: {e}")
            await self._speak_text("Sorry, I encountered an error processing your request.")
        finally:
            self.state.is_processing_llm = False
            print(f"🔄 LLM processing finished, state cleared")
    
    async def _stream_llm_to_tts(self, messages: List[Dict[str, str]]):
        """Stream LLM completion to TTS."""
        assistant_response = ""  # Initialize at method level to fix scoping
        request_timestamp = time.time()
        request_filename = ""
        was_interrupted = False
        error_msg = None
        
        try:
            self.state.is_speaking = True
            
            # Create async generator for LLM response
            async def llm_generator():
                nonlocal assistant_response, request_filename, was_interrupted, error_msg  # Access the method-level variables
                
                if self.config.conversation.llm_provider == "openai":
                    async for chunk in self._stream_openai_response(messages, request_timestamp):
                        # Check for interruption during LLM generation
                        if not self.state.is_speaking or not self.state.is_processing_llm:
                            print("🛑 LLM streaming interrupted by user")
                            was_interrupted = True
                            break
                        
                        if chunk is None:  # Error occurred
                            error_msg = "OpenAI API error"
                            yield "Sorry, I encountered an error with the language model."
                            return
                            
                        assistant_response += chunk
                        if self.show_tts_chunks:
                            print(f"🔊 TTS chunk: {chunk}")
                        yield chunk
                        
                elif self.config.conversation.llm_provider == "anthropic":
                    async for chunk in self._stream_anthropic_response(messages, request_timestamp):
                        # Check for interruption during LLM generation
                        if not self.state.is_speaking or not self.state.is_processing_llm:
                            print("🛑 LLM streaming interrupted by user")
                            was_interrupted = True
                            break
                        
                        if chunk is None:  # Error occurred
                            error_msg = "Anthropic API error"
                            yield "Sorry, I encountered an error with the language model."
                            return
                            
                        assistant_response += chunk
                        if self.show_tts_chunks:
                            print(f"🔊 TTS chunk: {chunk}")
                        yield chunk
                        
                else:
                    error_msg = f"Unsupported LLM provider: {self.config.conversation.llm_provider}"
                    yield f"Sorry, unsupported language model provider: {self.config.conversation.llm_provider}"
                

            
            # Use TTS to speak the streaming response
            result = await self.tts.speak_stream(llm_generator())
            
            # Use heuristic approach for conversation history (preserves exact LLM text)
            spoken_heuristic = self.tts.get_spoken_text_heuristic().strip()
            generated_vs_spoken = self.tts.get_generated_vs_spoken()
            was_fully_spoken = self.tts.was_fully_spoken()
            
            if spoken_heuristic or generated_vs_spoken.get('spoken_whisper'):
                print(f"🎙️ WHISPER TRACKING RESULTS:")
                print(f"   Generated: {len(generated_vs_spoken['generated'])} chars")
                print(f"   Whisper raw: {len(generated_vs_spoken['spoken_whisper'])} chars")
                print(f"   Heuristic: {len(spoken_heuristic)} chars")
                print(f"   Fully spoken: {was_fully_spoken}")
                print(f"   Heuristic text: '{spoken_heuristic[:200]}...'")
            
            # Determine content for history using heuristic approach
            content_for_history = spoken_heuristic if spoken_heuristic else assistant_response
            
            # Only add to history if we have meaningful content
            if content_for_history.strip():
                if was_fully_spoken:
                    print(f"💬 History updated with full generated content: '{content_for_history[:100]}...'")
                else:
                    print(f"💬 History updated with HEURISTIC content: '{content_for_history[:100]}...'")
                
                self.state.conversation_history.append({
                    "role": "assistant",
                    "content": content_for_history
                })
                
                # Log conversation turn
                self._log_conversation_turn("assistant", content_for_history)
            elif was_interrupted and not content_for_history:
                print("🛑 Response was interrupted - no spoken content to add")
            
        except Exception as e:
            error_msg = str(e)
            print(f"Error in _stream_llm_to_tts: {e}")
            self.logger.error(f"Error in _stream_llm_to_tts: {e}")
            await self._speak_text("Sorry, I encountered an error processing your request.")
        finally:
            # Log the response (or error) after completion
            response_timestamp = time.time()
            if request_filename:  # Only log if we made a request
                self._log_llm_response(
                    assistant_response, 
                    request_filename, 
                    response_timestamp, 
                    was_interrupted, 
                    error_msg,
                    self.config.conversation.llm_provider
                )
            
            self.state.is_speaking = False
            print("✅ Response completed successfully")

    async def _stream_openai_response(self, messages: List[Dict[str, str]], request_timestamp: float):
        """Stream response from OpenAI API."""
        models_to_try = [self.config.conversation.llm_model]
        
        for model in models_to_try:
            try:
                # Log the request before making the API call
                request_filename = self._log_llm_request(messages, model, request_timestamp, "openai")
                
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    max_tokens=self.config.conversation.max_tokens
                )
                
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                return  # Success
                
            except Exception as e:
                print(f"Failed with OpenAI model {model}: {e}")
                if model == models_to_try[-1]:
                    yield None  # Signal error
                    return

    async def _stream_anthropic_response(self, messages: List[Dict[str, str]], request_timestamp: float):
        """Stream response from Anthropic API."""
        models_to_try = [self.config.conversation.llm_model]
        
        for model in models_to_try:
            try:
                # Convert to appropriate format based on conversation mode
                if self.config.conversation.conversation_mode == "prefill":
                    user_message, assistant_prefix = self._convert_to_prefill_format(messages)
                    
                    # Log the request before making the API call
                    prefill_messages = [
                        {"role": "user", "content": user_message},
                        {"role": "assistant", "content": assistant_prefix}
                    ]
                    request_filename = self._log_llm_request(prefill_messages, model, request_timestamp, "anthropic")
                    
                    # Use Anthropic's completion API with prefill and CLI system prompt
                    response = self.anthropic_client.messages.create(
                        model=model,
                        max_tokens=self.config.conversation.max_tokens,
                        system=self.config.conversation.prefill_system_prompt,
                        messages=[
                            {"role": "user", "content": user_message},
                            {"role": "assistant", "content": assistant_prefix}
                        ],
                        stop_sequences=["\n\nH:", "\n\nClaude:"],
                        stream=True
                    )
                else:
                    # Chat mode - convert system message if present
                    anthropic_messages = []
                    system_message = None
                    
                    for msg in messages:
                        if msg["role"] == "system":
                            system_message = msg["content"]
                        else:
                            anthropic_messages.append(msg)
                    
                    # Log the request before making the API call
                    request_filename = self._log_llm_request(anthropic_messages, model, request_timestamp, "anthropic")
                    
                    # Use Anthropic's chat API
                    kwargs = {
                        "model": model,
                        "max_tokens": self.config.conversation.max_tokens,
                        "messages": anthropic_messages,
                        "stream": True
                    }
                    
                    if system_message:
                        kwargs["system"] = system_message
                    
                    response = self.anthropic_client.messages.create(**kwargs)
                
                # Stream the response
                accumulated_response = ""
                for chunk in response:
                    if chunk.type == "content_block_delta":
                        if hasattr(chunk.delta, 'text'):
                            content = chunk.delta.text
                            accumulated_response += content
                            yield content
                
                # Convert from prefill format if needed
                if self.config.conversation.conversation_mode == "prefill":
                    # The response should already be clean since we prefilled with the participant name
                    pass  # No additional processing needed
                    
                return  # Success
                
            except Exception as e:
                print(f"Failed with Anthropic model {model}: {e}")
                if model == models_to_try[-1]:
                    yield None  # Signal error
                    return
    
    async def _speak_text(self, text: str):
        """Speak a simple text message."""
        try:
            self.state.is_speaking = True
            result = await self.tts.speak_text(text)
            
            # Log Whisper tracking for testing
            spoken_heuristic = self.tts.get_spoken_text_heuristic().strip()
            if spoken_heuristic:
                print(f"🎙️ Simple speech heuristic result: '{spoken_heuristic[:100]}...'")
            
            if not result:
                print("🛑 Speech was interrupted")
        finally:
            self.state.is_speaking = False
    
    async def _force_process_buffer(self):
        """Force immediate processing of buffer after interruption."""
        # Wait a tiny bit for state to settle
        await asyncio.sleep(0.01)
        
        if not self.state.utterance_buffer or not self.state.last_utterance_time:
            return
            
        # Get combined transcript
        combined_transcript = " ".join(msg['text'] for msg in self.state.utterance_buffer).strip()
        
        if combined_transcript:
            # Determine primary speaker from buffer
            speakers = [msg['speaker'] for msg in self.state.utterance_buffer]
            primary_speaker = max(set(speakers), key=speakers.count) if speakers else "Unknown"
            
            print(f"🚀 Force processing buffer: '{combined_transcript}' (from {primary_speaker})")
            await self._process_with_llm(combined_transcript, primary_speaker)
            
            # Clear buffer and reset timing
            self.state.utterance_buffer.clear()
            self.state.last_utterance_time = None
            print(f"🧹 Buffer cleared after forced processing")
    
    async def cleanup(self):
        """Clean up all resources."""
        try:
            await self.stop_conversation()
        except Exception as e:
            print(f"Error stopping conversation: {e}")
        
        try:
            await self.stt.cleanup()
        except Exception as e:
            print(f"Error cleaning up STT: {e}")
        
        try:
            await self.tts.cleanup()
        except Exception as e:
            print(f"Error cleaning up TTS: {e}")

async def main():
    """Main function for the unified voice conversation system."""
    print("🎙️ Unified Voice Conversation System (YAML Config)")
    print("==================================================")
    
    conversation = None
    
    try:
        # Load configuration from YAML
        print("📁 Loading configuration...")
        config = load_config()
        print("✅ Configuration loaded successfully!")
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("\n💡 To create a config file:")
        print("   1. Copy config.yaml.example to config.yaml")
        print("   2. Fill in your API keys")
        print("   3. Adjust settings as needed")
        print("\n   OR run: python config_loader.py create-example")
        return
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        return
    except Exception as e:
        print(f"❌ Unexpected error loading config: {e}")
        return
    
    try:
        # Create conversation system
        conversation = UnifiedVoiceConversation(config)
        
        # Setup signal handling for graceful shutdown
        import signal
        
        def signal_handler():
            print("\n🛑 Received shutdown signal")
            if conversation:
                conversation.state.is_active = False
        
        # Register signal handlers
        for sig in [signal.SIGINT, signal.SIGTERM]:
            try:
                signal.signal(sig, lambda s, f: signal_handler())
            except ValueError:
                # Signal may not be available on all platforms
                pass
        
        success = await conversation.start_conversation()
        if not success:
            print("❌ Failed to start conversation")
            return
            
        # Keep running until interrupted
        print("🎯 Conversation is active. Press Ctrl+C to exit.")
        try:
            while conversation.state.is_active:
                await asyncio.sleep(0.5)  # Shorter sleep for more responsive shutdown
        except asyncio.CancelledError:
            print("\n⏹️ Task cancelled")
            
    except KeyboardInterrupt:
        print("\n👋 Conversation ended by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if conversation:
            print("🧹 Cleaning up...")
            try:
                await conversation.cleanup()
            except Exception as e:
                print(f"⚠️ Cleanup error: {e}")
            print("✅ Cleanup complete")

if __name__ == "__main__":
    asyncio.run(main()) 
