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
from typing import Optional, List, Dict, Any, AsyncGenerator, Union
from datetime import datetime, timedelta
from openai import OpenAI
import anthropic
from anthropic import AsyncAnthropic

# Import our modular systems and config loader
from async_stt_module import AsyncSTTStreamer, STTEventType # , STTResult
from async_tts_module import AsyncTTSStreamer
from config_loader import load_config, VoiceAIConfig
from tools import create_tool_registry
from character_system import create_character_manager, CharacterManager
from camera_capture import CameraCapture, CameraConfig as CameraCaptureConfig
from synchronized_thinking_sound import SynchronizedThinkingSoundPlayer
from websocket_ui_server import VoiceUIServer, UIMessage
from echo_filter import EchoFilter
from context_manager import ContextManager
from mel_aec_audio import stop_stream # needed for shutdown

INTERRUPTED_STR = "[Interrupted by user]"

@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation."""
    role: str  # "user", "assistant", "system", or "tool"
    content: Union[str, List[Dict[str, Any]]]  # String or list of content blocks (for images)
    timestamp: datetime
    status: str = "completed"  # "pending", "processing", "completed", "interrupted"
    speaker_id: Optional[int] = None
    speaker_name: Optional[str] = None  # Identified speaker name from voice fingerprinting
    character: Optional[str] = None  # Character name for multi-character conversations
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata for special handling
    id: Optional[str] = None  # Unique message ID (msg_{index}), assigned when added to history

@dataclass 
class ConversationState:
    """Tracks the current state of the conversation."""
    is_active: bool = False
    is_processing_llm: bool = False
    is_speaking: bool = False
    conversation_history: List[ConversationTurn] = field(default_factory=list)
    # Track what utterance is currently being processed
    current_processing_turn: Optional[ConversationTurn] = None
    # Track the current LLM streaming task so we can cancel it
    current_llm_task: Optional[asyncio.Task] = None
    # Track pending tool response to speak after interruption
    pending_tool_response: Optional[str] = None
    # Track request generation to handle concurrent requests
    request_generation: int = 0
    # Track active AI stream session for UI corrections
    current_ui_session_id: Optional[str] = None
    # Global speaker management
    next_speaker: Optional[str] = None  # Who should speak next
    current_speaker: Optional[str] = None  # Who is currently speaking  
    last_ai_speaker: Optional[str] = None  # Last AI character that spoke (for same_model mode)
    current_generation: int = 0  # The active generation number
    speaker_lock: asyncio.Lock = field(default_factory=asyncio.Lock)  # Prevent concurrent speaker selection


class UnifiedVoiceConversation:
    """Unified voice conversation system with YAML configuration."""
    
    def __init__(self, config: VoiceAIConfig):
        self.config = config
        self.state = ConversationState()

        self.llm_output_task = None
        
        # Initialize LLM clients
        self.openai_client = None
        self.anthropic_client = None
        self.async_anthropic_client = None
        self.bedrock_client = None
        self.async_bedrock_client = None
        
        # Track processing tasks to prevent race conditions
        self._processing_task = None
        # Removed _processing_generation - using state.current_generation instead
        # Track director requests separately
        self._director_generation = 0
        
        # Debounce timer for utterance processing (wait for multiple utterances)
        self._utterance_debounce_timer = None
        self._utterance_debounce_task = None
        
        # LLM clients will be initialized after character manager is created
        
        # Initialize STT system with participant names as keywords
        self._setup_stt_keywords(config)
        
        # Add echo cancellation settings to STT config if enabled
        if config.conversation.enable_echo_cancellation:
            config.stt.enable_echo_cancellation = True
            config.stt.aec_frame_size = config.conversation.aec_frame_size
            config.stt.aec_filter_length = config.conversation.aec_filter_length
            config.stt.aec_delay_ms = config.conversation.aec_delay_ms
        
        self.stt = AsyncSTTStreamer(config.stt, config.speakers)
        
        # Initialize TTS system
        self.tts = AsyncTTSStreamer(config.tts)
        
        # Connect TTS to STT for echo cancellation if enabled
        if config.conversation.enable_echo_cancellation:
            # Check if echo cancellation is actually available in STT
            if hasattr(self.stt, 'echo_canceller') and self.stt.echo_canceller is not None:
                # Set callback for TTS to send reference audio to STT
        
                print("🔊 Echo cancellation connected: TTS -> STT")
            else:
                print("⚠️  Echo cancellation requested but not available - please install speexdsp")
        
        # Initialize tool registry
        self.tool_registry = create_tool_registry(config.conversation.tools_config)
        
        # Initialize OSC client if enabled
        self.osc_client = None
        if config.osc:
            print(f"📡 OSC config found - enabled: {config.osc.enabled}")
            if config.osc.enabled:
                try:
                    from pythonosc import udp_client
                    self.osc_client = udp_client.SimpleUDPClient(config.osc.host, config.osc.port)
                    print(f"✅ OSC client initialized successfully!")
                    print(f"   Host: {config.osc.host}")
                    print(f"   Port: {config.osc.port}")
                    print(f"   Start address: {config.osc.speaking_start_address}")
                    print(f"   Stop address: {config.osc.speaking_stop_address}")
                except ImportError as e:
                    print(f"❌ OSC library not installed: {e}")
                    print("   Run: pip install python-osc")
                    self.osc_client = None
                except Exception as e:
                    print(f"❌ Failed to initialize OSC client: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                    self.osc_client = None
            else:
                print("🔇 OSC is disabled in config")
        else:
            print("🔇 No OSC configuration found")
        
        # Initialize camera if enabled
        self.camera: Optional[CameraCapture] = None
        if config.camera and config.camera.enabled:
            camera_config = CameraCaptureConfig(
                device_id=config.camera.device_id,
                resolution=tuple(config.camera.resolution),
                capture_on_speech=config.camera.capture_on_speech,
                save_captures=config.camera.save_captures,
                capture_dir=config.camera.capture_dir,
                jpeg_quality=config.camera.jpeg_quality
            )
            self.camera = CameraCapture(camera_config)
            if self.camera.start():
                print("📷 Camera capture enabled")
            else:
                print("⚠️ Failed to start camera, continuing without camera capture")
                self.camera = None
        
        # Initialize character manager (always use character mode)
        characters_config = {
            "characters": config.conversation.characters_config or {},
            "director": config.conversation.director_config or {},
            "api_keys": {
                "openai": config.conversation.openai_api_key,
                "anthropic": config.conversation.anthropic_api_key,
                "deepgram": config.conversation.deepgram_api_key,
                "elevenlabs": config.conversation.elevenlabs_api_key
            }
        }
        
        # If no characters defined, create a default Assistant character
        if not characters_config["characters"]:
            characters_config["characters"] = {
                "Assistant": {
                    "llm_provider": config.conversation.llm_provider,
                    "llm_model": config.conversation.llm_model,
                    "voice_id": config.conversation.voice_id,
                    "voice_settings": {
                        "speed": config.tts.speed,
                        "stability": config.tts.stability,
                        "similarity_boost": config.tts.similarity_boost
                    },
                    "system_prompt": config.conversation.system_prompt,
                    "max_tokens": config.conversation.max_tokens,
                    "temperature": 0.7,  # Default temperature
                    "max_prompt_tokens": 8000  # Default token limit for prompts
                }
            }
            # Simple director for single character
            if not characters_config["director"]:
                characters_config["director"] = {
                    "llm_provider": config.conversation.llm_provider,
                    "llm_model": config.conversation.llm_model,
                    "system_prompt": "You are directing a conversation. Since there is only one AI assistant, always respond with 'Assistant' when asked who should speak next."
                }
        
        self.character_manager: CharacterManager = create_character_manager(characters_config)
        
        # Initialize LLM clients based on conversation default and actual character providers
        self._initialize_llm_clients(config)
        
        # Initialize context manager if enabled
        self.context_manager = None
        if config.contexts and config.contexts.enabled:
            contexts_list = [
                {
                    'name': ctx.name,
                    'history_file': ctx.history_file,
                    'description': ctx.description,
                    'metadata': ctx.metadata,
                    'character_histories': ctx.character_histories
                }
                for ctx in config.contexts.contexts
            ]
            self.context_manager = ContextManager(
                contexts_config=contexts_list,
                state_dir=config.contexts.state_dir
            )
            self.context_manager.auto_save_enabled = config.contexts.auto_save_enabled
            self.context_manager.auto_save_interval = config.contexts.auto_save_interval
        
        # Initialize thinking sound player (use same sample rate as STT for echo cancellation)
        self.thinking_sound = SynchronizedThinkingSoundPlayer()
        
        # Connect thinking sound to echo cancellation if enabled
        if config.conversation.enable_echo_cancellation:
            if hasattr(self.stt, 'echo_canceller') and self.stt.echo_canceller is not None:
        
                print("🔊 Echo cancellation connected: Thinking sound -> STT")
        
        # Initialize echo filter
        if config.echo_filter and config.echo_filter.enabled:
            self.echo_filter = EchoFilter(
                similarity_threshold=config.echo_filter.similarity_threshold,
                time_window=config.echo_filter.time_window,
                min_length=config.echo_filter.min_length
            )
        else:
            self.echo_filter = None
        
        # Initialize UI server
        ui_port = config.ui_port if hasattr(config, 'ui_port') else 8795
        self.ui_server = VoiceUIServer(self, host='0.0.0.0', port=ui_port)
        
        # Show appropriate message
        if len(characters_config["characters"]) > 1:
            print(f"🎭 Multi-character mode: {', '.join(characters_config['characters'].keys())}")
        else:
            print("🎭 Character mode enabled")
        
        # Set up tool execution callback
        #self.tts.on_tool_execution = self._handle_tool_execution
        
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
        
        state_file = getattr(self.config.conversation, "state_file", None)
        self._explicit_state_file = bool(state_file)
        if state_file:
            self.conversation_state_file = Path(state_file)
        elif self.context_manager:
            self.conversation_state_file = None
            print("🗂️ Context manager detected; skipping global last_state.json restore")
        else:
            self.conversation_state_file = self.conversation_logs_dir / "last_state.json"
        if self.conversation_state_file:
            self.conversation_state_file.parent.mkdir(parents=True, exist_ok=True)
        self._suppress_state_save = True
        
        # Initialize conversation log file
        self._init_conversation_log()
        
        # Load conversation history
        if self.context_manager:
            # Load original histories for all contexts
            self.context_manager.load_original_histories(self._parse_history_file)
            # Load saved states for all contexts
            self.context_manager.load_all_states()
            # Sync history from active context
            self._sync_history_from_context()
        else:
            # Legacy: Load history directly if no context manager
            self._load_history_file()
        
        # Attempt to restore previous conversation state from JSON
        if self.conversation_state_file:
            self._load_conversation_state()
        self._suppress_state_save = False
        self._save_conversation_state()
        
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
    
    def _log_conversation_turn(self, role: str, content: Union[str, List[Dict[str, Any]]], speaker_name: Optional[str] = None):
        """Log a conversation turn in history file format.
        
        Args:
            role: 'user' or 'assistant'
            content: The message content (may already include speaker prefix) or image content
            speaker_name: The identified speaker name (for user messages)
        """
        try:
            # Handle image content
            if isinstance(content, list):
                with open(self.conversation_log_file, 'a', encoding='utf-8') as f:
                    for item in content:
                        if item.get("type") == "text":
                            f.write(f"{item.get('text', '')}\n\n")
                        elif item.get("type") == "image":
                            f.write(f"[Image: {item.get('source', {}).get('media_type', 'unknown')}]\n\n")
                return
                
            # Handle string content
            # Check if content already has a speaker prefix (for multi-character mode)
            if role == "assistant" and ": " in content and content.index(": ") < 50:
                # Content already includes character name, use as-is
                with open(self.conversation_log_file, 'a', encoding='utf-8') as f:
                    f.write(f"{content}\n\n")
            else:
                # Determine participant name based on role
                if role == "user":
                    # Use speaker name if available, otherwise fallback to "H"
                    participant = speaker_name if speaker_name else "H"
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
    
    def add_image_to_conversation(self, image_path: str, text: Optional[str] = None) -> bool:
        """Add an image to the conversation history.
        
        Args:
            image_path: Path to the image file
            text: Optional text to accompany the image
            
        Returns:
            bool: True if image was successfully added
        """
        try:
            # Read the image file
            image_path = Path(image_path).resolve()
            if not image_path.exists():
                print(f"❌ Image file not found: {image_path}")
                return False
                
            # Determine media type
            suffix = image_path.suffix.lower()
            media_type_map = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp'
            }
            media_type = media_type_map.get(suffix, 'image/jpeg')
            
            # Read image data
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            # Encode to base64
            import base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # Create content blocks
            content = []
            if text:
                content.append({"type": "text", "text": text})
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_base64
                }
            })
            
            # Add to conversation history
            image_turn = ConversationTurn(
                role="user",
                content=content,
                timestamp=datetime.now(),
                status="completed"
            )
            self._add_turn_to_history(image_turn)
            
            # Log the image
            self._log_conversation_turn("user", content)
            
            print(f"📸 Added image to conversation: {image_path.name}")
            
            # Process pending utterances (which will include this image)
            asyncio.create_task(self._process_pending_utterances())
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to add image: {e}")
            self.logger.error(f"Failed to add image: {e}")
            return False
    
    def _sync_history_from_context(self):
        """Sync conversation history from active context."""
        if not self.context_manager:
            return
        
        context = self.context_manager.get_active_context()
        if context:
            # Clear current history and load from context
            self.state.conversation_history = context.current_history.copy()
            print(f"📋 Synced history from context '{context.config.name}': {len(self.state.conversation_history)} turns")
            
            # Rebuild detected_speakers from both original history and current conversation state
            if not hasattr(self, 'detected_speakers'):
                self.detected_speakers = set()
            
            # First, get speakers from original history file
            if context.config.history_file:
                try:
                    # Re-parse to rebuild detected_speakers (but don't use the messages)
                    _ = self._parse_history_file(context.config.history_file)
                    print(f"🔄 Rebuilt detected speakers from '{context.config.history_file}': {sorted(self.detected_speakers)}")
                except Exception as e:
                    print(f"⚠️ Failed to rebuild detected speakers from history file: {e}")
            
            # Also extract speakers from the current conversation state (where Unknown Speakers would be)
            additional_speakers = set()
            for turn in self.state.conversation_history:
                if hasattr(turn, 'speaker_name') and turn.speaker_name:
                    additional_speakers.add(turn.speaker_name)
                # Also check for speaker names in content for turns that might have embedded speaker info
                if hasattr(turn, 'content') and turn.content:
                    # Look for patterns like "Unknown Speaker 1:" in the content
                    import re
                    speaker_matches = re.findall(r'^([A-Za-z0-9_\s\.]+):', turn.content, re.MULTILINE)
                    for match in speaker_matches:
                        speaker_name = match.strip()
                        if self._is_valid_speaker_name(speaker_name):
                            additional_speakers.add(speaker_name)
            
            if additional_speakers:
                self.detected_speakers.update(additional_speakers)
                print(f"🔄 Added speakers from conversation state: {sorted(additional_speakers)}")
                print(f"🔄 Total detected speakers: {sorted(self.detected_speakers)}")
        
        self._save_conversation_state()
    
    def _sync_history_to_context(self):
        """Sync conversation history to active context."""
        if not self.context_manager:
            return
        
        context = self.context_manager.get_active_context()
        if context:
            # Update context with current history
            context.current_history = self.state.conversation_history.copy()
            context.is_modified = True
    
    def _add_turn_to_history(self, turn: ConversationTurn):
        """Add a turn to conversation history and sync with context."""
        # Add to local history

        # check if we are editing an existing message
        editing_i = None
        if turn.metadata and turn.metadata.get("message_id", None):
            message_id = turn.metadata['message_id']
            for i, prev_turn in list(enumerate(self.state.conversation_history))[::-1][:10]:
                if prev_turn.metadata and prev_turn.metadata.get("message_id", None) == message_id:
                    editing_i = i
                    break


        if editing_i:
            # Keep the existing ID when editing
            if not turn.id:
                turn.id = f"msg_{editing_i}"
            self.state.conversation_history[editing_i] = turn
        else:
            # Assign a proper ID based on the new index, unless already set
            new_index = len(self.state.conversation_history)
            if not turn.id:
                turn.id = f"msg_{new_index}"
            self.state.conversation_history.append(turn)
        
        # Sync entire history TO context (don't use context.add_turn to avoid sync issues)
        self._sync_history_to_context()
        
        self._save_conversation_state()
    
    def _log_loaded_history(self):
        """Log any existing conversation history to the conversation log."""
        if not self.state.conversation_history:
            return
            
        try:
            with open(self.conversation_log_file, 'a', encoding='utf-8') as f:
                f.write("## Loaded History\n\n")
                
            # Log each message from loaded history
            for turn in self.state.conversation_history:
                self._log_conversation_turn(turn.role, turn.content)
                
            with open(self.conversation_log_file, 'a', encoding='utf-8') as f:
                f.write("## New Conversation\n\n")
                
            print(f"📜 Logged {len(self.state.conversation_history)} history messages to conversation log")
            
        except Exception as e:
            print(f"⚠️ Failed to log loaded history: {e}")
            self.logger.error(f"Failed to log loaded history: {e}")
    
    def _serialize_conversation_turn(self, turn: ConversationTurn) -> Dict[str, Any]:
        """Convert a conversation turn to JSON-serializable data."""
        return {
            "role": turn.role,
            "content": turn.content,
            "timestamp": turn.timestamp.isoformat(),
            "status": turn.status,
            "speaker_id": turn.speaker_id,
            "speaker_name": turn.speaker_name,
            "character": turn.character,
            "metadata": turn.metadata,
        }
    
    def _deserialize_conversation_turn(self, data: Dict[str, Any]) -> ConversationTurn:
        """Create a ConversationTurn from persisted data."""
        timestamp = datetime.now()
        ts_value = data.get("timestamp")
        if ts_value:
            try:
                timestamp = datetime.fromisoformat(ts_value)
            except ValueError:
                pass
        
        return ConversationTurn(
            role=data.get("role", "assistant"),
            content=data.get("content"),
            timestamp=timestamp,
            status=data.get("status", "completed"),
            speaker_id=data.get("speaker_id"),
            speaker_name=data.get("speaker_name"),
            character=data.get("character"),
            metadata=data.get("metadata"),
        )
    
    def _get_current_processing_turn_index(self) -> Optional[int]:
        """Return the index of the current processing turn if available."""
        if self.state.current_processing_turn and self.state.current_processing_turn in self.state.conversation_history:
            return self.state.conversation_history.index(self.state.current_processing_turn)
        return None
    
    def _serialize_conversation_state(self) -> Dict[str, Any]:
        """Build a JSON representation of the current conversation state."""
        return {
            "conversation_history": [
                self._serialize_conversation_turn(turn)
                for turn in self.state.conversation_history
            ],
            "state": {
                "current_generation": self.state.current_generation,
                "request_generation": self.state.request_generation,
                "pending_tool_response": self.state.pending_tool_response,
                "next_speaker": self.state.next_speaker,
                "current_speaker": self.state.current_speaker,
                "last_ai_speaker": self.state.last_ai_speaker,
                "is_processing_llm": self.state.is_processing_llm,
                "is_speaking": self.state.is_speaking,
            },
            "current_processing_turn_index": self._get_current_processing_turn_index(),
            "saved_at": datetime.now().isoformat(),
        }
    
    def _save_conversation_state(self):
        """Persist the current conversation state to JSON."""
        if not hasattr(self, "conversation_state_file") or not self.conversation_state_file:
            return
        if getattr(self, "_suppress_state_save", False):
            return
        
        try:
            data = self._serialize_conversation_state()
            temp_file = self.conversation_state_file.with_suffix(self.conversation_state_file.suffix + ".tmp")
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            temp_file.replace(self.conversation_state_file)
        except Exception as e:
            print(f"⚠️ Failed to save conversation state: {e}")
            if hasattr(self, "logger"):
                self.logger.error(f"Failed to save conversation state: {e}")
    
    def _rebuild_detected_speakers_from_history(self):
        """Refresh detected speaker list from current conversation history."""
        if not hasattr(self, 'detected_speakers'):
            self.detected_speakers = set()
        else:
            self.detected_speakers.clear()
        
        for turn in self.state.conversation_history:
            if turn.speaker_name:
                self.detected_speakers.add(turn.speaker_name)
            if isinstance(turn.content, str):
                matches = re.findall(r'^([A-Za-z0-9_\s\.]+):', turn.content, re.MULTILINE)
                for match in matches:
                    speaker_name = match.strip()
                    if self._is_valid_speaker_name(speaker_name):
                        self.detected_speakers.add(speaker_name)
        
        if self.detected_speakers:
            print(f"🔄 Rebuilt detected speakers from state: {sorted(self.detected_speakers)}")
    
    def _load_conversation_state(self):
        """Load persisted conversation state from JSON if available."""
        if not hasattr(self, "conversation_state_file") or not self.conversation_state_file:
            return
        
        if not self.conversation_state_file.exists():
            return
        
        try:
            with open(self.conversation_state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            history_data = data.get("conversation_history", [])
            if not history_data:
                return
            
            loaded_turns = [self._deserialize_conversation_turn(item) for item in history_data]
            self.state.conversation_history = loaded_turns
            
            state_data = data.get("state", {})
            self.state.current_generation = state_data.get("current_generation", 0)
            self.state.request_generation = state_data.get("request_generation", 0)
            self.state.pending_tool_response = state_data.get("pending_tool_response")
            self.state.next_speaker = state_data.get("next_speaker")
            self.state.current_speaker = state_data.get("current_speaker")
            self.state.last_ai_speaker = state_data.get("last_ai_speaker")
            self.state.is_processing_llm = False
            self.state.is_speaking = False
            
            idx = data.get("current_processing_turn_index")
            if isinstance(idx, int) and 0 <= idx < len(self.state.conversation_history):
                self.state.current_processing_turn = self.state.conversation_history[idx]
            else:
                self.state.current_processing_turn = None
            
            self._rebuild_detected_speakers_from_history()
            self._sync_history_to_context()
            
            print(f"💾 Restored {len(loaded_turns)} turns from {self.conversation_state_file}")
        except Exception as e:
            print(f"⚠️ Failed to load conversation state: {e}")
            if hasattr(self, "logger"):
                self.logger.error(f"Failed to load conversation state: {e}")
    
    def _setup_stt_keywords(self, config: VoiceAIConfig):
        """Setup STT keywords including character names."""
        keywords = []
        
        # Add character names from character configs
        if config.conversation.characters_config:
            for char_name, char_config in config.conversation.characters_config.items():
                # Add character key name with high weight
                keywords.append((char_name, 10.0))
                
                # Add prefill name if different
                if isinstance(char_config, dict) and 'prefill_name' in char_config:
                    prefill_name = char_config['prefill_name']
                    if prefill_name != char_name:
                        # For Nova-3, we can use multi-word keyterms
                        if config.stt.model == "nova-3":
                            keywords.append((prefill_name, 10.0))
                        else:
                            # For other models, only add individual words
                            for word in prefill_name.split():
                                if len(word) > 2:  # Skip short words
                                    keywords.append((word, 10.0))
        
        # Add prefill participant names from regular mode
        elif config.conversation.prefill_participants:
            for participant in config.conversation.prefill_participants:
                if config.stt.model == "nova-3":
                    # Nova-3 can handle full phrases
                    keywords.append((participant, 10.0))
                else:
                    # Other models: add individual words
                    for word in participant.split():
                        if len(word) > 2:  # Skip short words
                            keywords.append((word, 10.0))
        
        # Add any existing keywords from config
        if hasattr(config.stt, 'keywords') and config.stt.keywords:
            keywords.extend(config.stt.keywords)
        
        # Remove duplicates while keeping highest weight
        keyword_dict = {}
        for word, weight in keywords:
            if word not in keyword_dict or weight > keyword_dict[word]:
                keyword_dict[word] = weight
        
        # Convert back to list of tuples
        config.stt.keywords = [(word, weight) for word, weight in keyword_dict.items()]
        
        if config.stt.keywords:
            print(f"🔤 Added {len(config.stt.keywords)} keywords to STT including: {list(keyword_dict.keys())[:5]}")
    
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
    
    def _sanitize_messages_for_logging(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create a sanitized copy of messages suitable for logging.
        Replaces image data with metadata to avoid logging raw base64 data.
        """
        sanitized = []
        for msg in messages:
            sanitized_msg = {"role": msg["role"]}
            
            # Handle content
            content = msg.get("content", "")
            if isinstance(content, list):
                # Content is a list of blocks (text/image)
                sanitized_content = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            # Keep text as-is
                            sanitized_content.append(item)
                        elif item.get("type") == "image":
                            # Replace image data with metadata
                            source = item.get("source", {})
                            sanitized_content.append({
                                "type": "image",
                                "source": {
                                    "type": source.get("type", "unknown"),
                                    "media_type": source.get("media_type", "unknown"),
                                    "data": f"[BASE64_IMAGE_DATA - {len(source.get('data', ''))} chars]"
                                }
                            })
                        else:
                            # Keep other types as-is
                            sanitized_content.append(item)
                    else:
                        # Non-dict items, keep as-is
                        sanitized_content.append(item)
                sanitized_msg["content"] = sanitized_content
            else:
                # Simple string content
                sanitized_msg["content"] = content
            
            # Copy any metadata fields
            for key in ["_is_prefill", "_prefill_name"]:
                if key in msg:
                    sanitized_msg[key] = msg[key]
            
            sanitized.append(sanitized_msg)
        
        return sanitized
    
    def _log_llm_request(self, messages: List[Dict[str, str]], model: str, timestamp: float, provider: str = None, stop_sequences: List[str] = None) -> str:
        """Log LLM request to a file and return the filename."""
        filename = self._generate_log_filename("request", timestamp)
        filepath = self.llm_logs_dir / filename
        
        # Sanitize messages to avoid logging raw image data
        sanitized_messages = self._sanitize_messages_for_logging(messages)
        
        request_data = {
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp).isoformat(),
            "provider": provider or self.config.conversation.llm_provider,
            "model": model,
            "messages": sanitized_messages,
            "max_tokens": self.config.conversation.max_tokens,
            "stream": True,
            "conversation_mode": self.config.conversation.conversation_mode,
            "request_type": "llm_completion"
        }
        
        # Add stop sequences if provided
        if stop_sequences:
            request_data["stop_sequences"] = stop_sequences
        
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

    def _convert_to_prefill_format(self, messages: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], str]:
        """Convert chat messages to prefill format with image support.
        Returns (user_messages, assistant_message_prefix).
        
        Images are handled by splitting the conversation into blocks:
        - Text spans between images become text user messages
        - Images become image user messages  
        - Everything after the last image goes in the assistant message
        """
        human_name = self.config.conversation.prefill_participants[0]  # Default: 'H'
        ai_name = self.config.conversation.prefill_participants[1]     # Default: 'Claude'
        
        # Process messages to find images and create blocks
        user_messages = []
        current_text_block = []
        last_image_index = -1
        
        # Find the index of the last image
        for i, msg in enumerate(messages):
            if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                # Check if this message contains an image
                for content_item in msg["content"]:
                    if content_item.get("type") == "image":
                        last_image_index = i
                        break
        
        # Process messages up to and including the last image
        for i, msg in enumerate(messages):
            if msg["role"] == "system":
                continue  # Skip system messages in prefill mode
                
            # Check if this is an image message
            is_image_message = False
            if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                for content_item in msg["content"]:
                    if content_item.get("type") == "image":
                        is_image_message = True
                        break
            
            if is_image_message and i <= last_image_index:
                # First, add any accumulated text block
                if current_text_block:
                    text_content = "\n\n".join(current_text_block)
                    user_messages.append({"role": "user", "content": text_content})
                    current_text_block = []
                
                # Then add the image message as-is
                user_messages.append(msg)
            elif i <= last_image_index:
                # Regular text message before the last image - accumulate it
                if msg["role"] == "user":
                    content = msg.get('content', '')
                    if isinstance(content, str):
                        if not re.match(r'^[^:]+:', content):
                            content = f"{human_name}: {content}"
                        current_text_block.append(content)
                elif msg["role"] == "assistant":
                    content = msg.get('content', '')
                    if isinstance(content, str):
                        current_text_block.append(content)
        
        # Add any remaining text block before the last image
        if current_text_block and last_image_index >= 0:
            text_content = "\n\n".join(current_text_block)
            user_messages.append({"role": "user", "content": text_content})
            current_text_block = []
        
        # Build assistant content from everything after the last image
        assistant_turns = []
        for i, msg in enumerate(messages):
            if i > last_image_index:
                if msg["role"] == "system":
                    continue
                elif msg["role"] == "user":
                    content = msg.get('content', '')
                    if isinstance(content, str):
                        if not re.match(r'^[^:]+:', content):
                            content = f"{human_name}: {content}"
                        assistant_turns.append(content)
                elif msg["role"] == "assistant":
                    content = msg.get('content', '')
                    if isinstance(content, str):
                        assistant_turns.append(content)
        
        # If no images were found, put everything in assistant message
        if last_image_index == -1:
            for msg in messages:
                if msg["role"] == "system":
                    continue
                elif msg["role"] == "user":
                    content = msg.get('content', '')
                    if isinstance(content, str):
                        if not re.match(r'^[^:]+:', content):
                            content = f"{human_name}: {content}"
                        assistant_turns.append(content)
                elif msg["role"] == "assistant":
                    content = msg.get('content', '')
                    if isinstance(content, str):
                        assistant_turns.append(content)
        
        # Join assistant turns and add prefill
        assistant_content = "\n\n".join(assistant_turns)
        if assistant_content:
            assistant_content += f"\n\n{ai_name}:"
        else:
            assistant_content = f"{ai_name}:"
        
        # If no user messages were created, use the default prefill message
        if not user_messages:
            user_messages = [{"role": "user", "content": self.config.conversation.prefill_user_message}]
            
        return user_messages, assistant_content

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
        
        Supports multi-speaker conversations. All speakers are preserved in the conversation
        history for LLM context, but for voice interaction only the configured participants
        are used for the actual conversation flow.
        
        Expected format:
        SpeakerName: message content
        
        AnotherSpeaker: response content
        
        Returns messages with 'character' field for multi-character conversations.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # Get configured participant names for the current voice conversation
            human_name = self.config.conversation.prefill_participants[0]  # e.g., 'H'
            ai_name = self.config.conversation.prefill_participants[1]     # e.g., 'Claude 3 Opus'
            
            # Store all detected speakers for stop sequences
            self.detected_speakers = set()
            
            import re
            messages = []
            current_speaker = None
            current_content = []
            
            for line in content.split('\n'):
                line = line.strip()
                
                # Match speaker pattern: "SpeakerName: content" but be much more restrictive
                # Only match lines that look like actual speaker names, not formatting or bullet points
                speaker_match = re.match(r'^([A-Za-z0-9_\s\.]+):\s*(.*)', line)
                
                if speaker_match:
                    speaker_name = speaker_match.group(1).strip()
                    message_content = speaker_match.group(2).strip()
                    
                    # Filter out obvious non-speaker patterns
                    if self._is_valid_speaker_name(speaker_name):
                        # Track all detected speakers
                        self.detected_speakers.add(speaker_name)
                        
                        # Check if this is the same speaker continuing
                        if speaker_name == current_speaker:
                            # Same speaker - just add the message content (not the speaker name again)
                            if message_content:
                                current_content.append(message_content)
                            # Skip empty continuations from same speaker
                        else:
                            # Different speaker - save previous and start new
                            if current_speaker and current_content:
                                # Determine role based on configured participants
                                # For multi-speaker: classify as 'user' if it's the human participant,
                                # otherwise as 'assistant' to preserve the conversation context
                                role = "user" if current_speaker == human_name else "assistant"
                                
                                # In character mode, we need to preserve the character name separately
                                full_content = '\n'.join(current_content).strip()
                                
                                message = {
                                    "role": role,
                                    "content": full_content
                                }
                                
                                # Add character name if it's an assistant message and not the default AI
                                if role == "assistant" and current_speaker != ai_name:
                                    # Store the speaker name for later character matching
                                    message["_speaker_name"] = current_speaker
                                
                                messages.append(message)
                            
                            # Start new message
                            current_speaker = speaker_name
                            current_content = [message_content] if message_content else []
                    else:
                        # Debug: log filtered speakers
                        if "Unknown Speaker" in speaker_name:
                            print(f"🚫 Filtered out speaker: '{speaker_name}' (validation failed)")
                            # Additional debug for Unknown Speaker cases
                            print(f"   Length: {len(speaker_name)} chars, Words: {len(speaker_name.split())}")
                            invalid_chars = ['*', '[', ']', '(', ')', '"', "'"]
                            has_invalid = any(char in speaker_name for char in invalid_chars)
                            print(f"   Contains invalid chars: {has_invalid}")
                        
                        # Invalid speaker - treat as continuation of current message or skip
                        if current_speaker:
                            # Add the full line including the colon as continuation
                            current_content.append(line)
                        # If no current speaker, just skip the line
                    
                elif line and current_speaker:
                    # Continue current message (current_speaker is already validated)
                    current_content.append(line)
                # Skip empty lines or lines without a speaker
            
            # Add final message if exists (current_speaker is already validated)
            if current_speaker and current_content:
                role = "user" if current_speaker == human_name else "assistant"
                full_content = '\n'.join(current_content).strip()
                
                message = {
                    "role": role,
                    "content": full_content
                }
                
                # Add character name if it's an assistant message and not the default AI
                if role == "assistant" and current_speaker != ai_name:
                    # Store the speaker name for later character matching
                    message["_speaker_name"] = current_speaker
                
                messages.append(message)
            
            print(f"📊 Detected speakers in history: {sorted(self.detected_speakers)}")
            return messages
            
        except Exception as e:
            print(f"❌ Failed to parse history file {file_path}: {e}")
            return []

    def _is_valid_speaker_name(self, speaker_name: str) -> bool:
        """Validate if a string looks like a legitimate speaker name."""
        # Exclude obvious non-speaker patterns
        invalid_patterns = [
            r'^\d+\.',  # Numbers like "1.", "2."
            r'^-\s',    # Bullet points like "- Panel"
            r'^\*\*',   # Bold markdown like "**Notable"
            r'^Step\s', # Step instructions
            r'^Panel\s', # Panel descriptions
            r'^\[',     # Bracket descriptions
            r'translation$', # Translation notes
            r'^The\s.*(part|truth|question|boundary)',  # Descriptive phrases
            r'^A\s(verse|poem)',  # Poetry descriptions
            r'Dynamics$',  # Ends with "Dynamics"
            r'Interactions$',  # Ends with "Interactions"
            r'Elements$',   # Ends with "Elements"
            r'Themes$',     # Ends with "Themes"
            r'^haha\s',     # Casual expressions
            r'https$',      # URLs
            r'Metaphors$',  # Ends with "Metaphors"
            r'Rigidity$',   # Ends with "Rigidity"
            r'Rejection$',  # Ends with "Rejection"
            r'Performance$', # Ends with "Performance"
            r'Engagement$',  # Ends with "Engagement"
        ]
        
        # Check against invalid patterns
        for pattern in invalid_patterns:
            if re.search(pattern, speaker_name, re.IGNORECASE):
                return False
        
        # Additional filters
        # Too long (probably a sentence)
        if len(speaker_name) > 50:
            return False
            
        # Contains too many words (probably descriptive text)
        if len(speaker_name.split()) > 4:
            return False
            
        # Contains certain punctuation that indicates it's not a name
        if any(char in speaker_name for char in ['*', '[', ']', '(', ')', '"', "'"]):
            return False
            
        return True

    def _load_history_file(self):
        """Load conversation history from file if specified."""
        if not self.config.conversation.history_file:
            return
            
        file_path = self.config.conversation.history_file
        print(f"📜 Loading conversation history from: {file_path}")
        
        history_messages = self._parse_history_file(file_path)
        
        if history_messages:
            # Convert old message format to ConversationTurn format
            for msg in history_messages:
                # Try to resolve character name from speaker name
                character_name = None
                if msg.get("_speaker_name") and hasattr(self, 'character_manager') and self.character_manager:
                    speaker = msg["_speaker_name"]
                    # Try to match by character name or prefill name
                    for char_name, char_config in self.character_manager.characters.items():
                        if (speaker == char_name or 
                            (char_config.prefill_name and speaker == char_config.prefill_name)):
                            character_name = char_name
                            break
                
                turn = ConversationTurn(
                    role=msg["role"],
                    content=msg["content"], 
                    timestamp=datetime.now(),  # We don't have original timestamps
                    status="completed",  # Historical messages are completed
                    character=character_name
                )
                self._add_turn_to_history(turn)
                
                # Debug logging for character resolution
                if msg["role"] == "assistant" and msg.get("_speaker_name"):
                    print(f"   📝 {msg['_speaker_name']} → character: {character_name or 'None'}")
                
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
        #self.stt.on(STTEventType.SPEECH_STARTED, self._on_speech_started)
        #self.stt.on(STTEventType.SPEECH_ENDED, self._on_speech_ended)
        
        # Handle speaker changes
        #self.stt.on(STTEventType.SPEAKER_CHANGE, self._on_speaker_change)
        
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
            
            # Start UI server
            asyncio.create_task(self.ui_server.start())
            
            # Start context auto-save if enabled
            if self.context_manager:
                await self.context_manager.start_auto_save()
            
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
        self._save_conversation_state()
        
        # Cancel conversation management
        if self.conversation_task and not self.conversation_task.done():
            self.conversation_task.cancel()
            try:
                await self.conversation_task
            except asyncio.CancelledError:
                pass
        
        # Stop TTS if speaking
        if self.state.is_speaking:
            await self.tts.interrupt()
        
        # Stop STT
        await self.stt.stop_listening()
        
        # Stop context auto-save and save final state
        if self.context_manager:
            await self.context_manager.stop_auto_save()
            self.context_manager.save_all_states()
        
        print("✅ Conversation stopped")
    
    def _get_spoken_text_with_fallback(self, candidate: str = "") -> str:
        """Retrieve spoken text using heuristic with sane fallbacks when Whisper data is unavailable."""
        candidate = (candidate or "").strip()
        return self.tts.generated_text
        session = getattr(self.tts, "current_session", None)
        if not session:
            return candidate
        
        generated = (session.generated_text or "").strip()
        
        # Primary heuristic (uses Whisper character counts when available)
        heuristic_text = ""
        try:
            heuristic_text = self.tts.get_spoken_text_heuristic().strip()
        except Exception as e:
            print(f"⚠️ Failed to compute spoken text heuristic: {e}")
        
        if heuristic_text:
            if getattr(self.tts, "_interrupted", False) and generated and heuristic_text == generated:
                fallback = (session.spoken_text_for_tts or "").strip()
                if fallback and fallback != generated:
                    print("🔍 [DEBUG] Using spoken_text_for_tts fallback (heuristic matched full generated text)")
                    return fallback
            return heuristic_text
        
        # Whisper segments if tracking is active
        if session.current_spoken_content:
            fallback = " ".join(segment.text for segment in session.current_spoken_content).strip()
            if fallback:
                print("🔍 [DEBUG] Using Whisper spoken segments fallback")
                return fallback
        
        # Fallback to text actually dispatched to TTS
        fallback = (session.spoken_text_for_tts or "").strip()
        if fallback:
            print("🔍 [DEBUG] Using spoken_text_for_tts fallback")
            return fallback
        
        # Fall back to provided candidate or full generated text
        if candidate:
            return candidate
        
        if getattr(self.tts, "_interrupted", False) and generated:
            return generated
        
        return generated
    
    async def _stop_tts_and_notify_ui(self) -> str:
        """Stop TTS playback and update the UI with any spoken content."""
        interrupted_text = await self.tts.interrupt()
        
        if hasattr(self, 'ui_server'):
            session_id = getattr(self.state, "current_ui_session_id", None)
            if session_id:
                await self.ui_server.broadcast(UIMessage(
                    type="ai_stream_correction",
                    data={
                        "session_id": session_id,
                        "corrected_text": interrupted_text,
                        "was_interrupted": True
                    }
                ))
                print(f"🔄 Sent AI stream correction for session '{session_id}'")
        
        return interrupted_text
    

    def get_llm_context(self, next_speaker, request_timestamp):
        character_config = self.character_manager.get_character_config(next_speaker)
        # Check if we should use prefill format
        use_prefill = (self.config.conversation.conversation_mode == "prefill" and 
                        character_config.llm_provider in ["anthropic", "bedrock"])
        
        if use_prefill:
            # For prefill mode, convert directly from raw history
            prefill_name = character_config.prefill_name or character_config.name
            raw_history = self._get_conversation_history_for_character()
            
            # Get character histories from current context if available
            context_char_histories = None
            if self.context_manager:
                active_context = self.context_manager.get_active_context()
                if active_context:
                    context_char_histories = active_context.character_histories
                    print(f"🔍 DEBUG: Active context '{active_context.config.name}' has character_histories: {list(context_char_histories.keys()) if context_char_histories else None}")
            
            print(f"🔍 DEBUG: Creating prefill messages for character '{next_speaker}' (prefill_name: '{prefill_name}')")
            print(f"🔍 DEBUG: Context has character histories: {context_char_histories is not None}")
            if context_char_histories:
                print(f"🔍 DEBUG: Character histories available for: {list(context_char_histories.keys())}")
            
            # Create messages in prefill format directly
            messages = self._create_character_prefill_messages(
                raw_history, 
                next_speaker, 
                prefill_name,
                character_config.system_prompt,
                context_char_histories
            )
        else:
            # Standard chat format
            # Get character histories from current context if available
            context_char_histories = None
            if self.context_manager:
                active_context = self.context_manager.get_active_context()
                if active_context:
                    context_char_histories = active_context.character_histories
            
            messages = self.character_manager.format_messages_for_character(
                next_speaker,
                self._get_conversation_history_for_character(),
                context_char_histories
            )
        
        # Check if we're in prefill format to generate stop sequences
        stop_sequences = None
        if self.config.conversation.conversation_mode == "prefill":
            stop_sequences = []
            # Add all character names as stop sequences
            for char_name in self.character_manager.characters.keys():
                stop_sequences.append(f"\n\n{char_name}:")
                # Also add prefill names if different
                char_config_temp = self.character_manager.get_character_config(char_name)
                if char_config_temp and char_config_temp.prefill_name and char_config_temp.prefill_name != char_name:
                    stop_sequences.append(f"\n\n{char_config_temp.prefill_name}:")
            
            # Add human names
            if hasattr(self.config.conversation, 'prefill_participants'):
                human_name = self.config.conversation.prefill_participants[0]
                stop_sequences.append(f"\n\n{human_name}:")
            
            # Add any detected speakers
            if hasattr(self, 'detected_speakers') and self.detected_speakers:
                for speaker in self.detected_speakers:
                    speaker_stop = f"\n\n{speaker}:"
                    if speaker_stop not in stop_sequences:
                        stop_sequences.append(speaker_stop)
            
            # Add System: to stop sequences
            stop_sequences.append("\n\nSystem:")
            stop_sequences.append("\n\nA:")
            
            # Add User 1 through User 10
            for i in range(1, 11):
                stop_sequences.append(f"\n\nUser {i}:")
            
            # Add interrupted marker
            stop_sequences.append("[Interrupted by user]")
            
            # Remove duplicates while preserving order
            stop_sequences = list(dict.fromkeys(stop_sequences))
        # Log the request
        request_filename = self._log_llm_request(
            messages, 
            character_config.llm_model, 
            request_timestamp,
            character_config.llm_provider,
            stop_sequences
        )
        return messages, character_config, stop_sequences, request_filename

    def get_llm_text_stream(self, messages, character_config, request_timestamp, ui_session_id, pending_message_id: Optional[str] = None):
        if character_config.llm_provider == "openai":
            return self._stream_character_openai_response(messages, character_config, request_timestamp, ui_session_id, pending_message_id)
        elif character_config.llm_provider == "anthropic":
            return self._stream_character_anthropic_response(messages, character_config, request_timestamp, ui_session_id, pending_message_id)
        elif character_config.llm_provider == "bedrock":
            return self._stream_character_bedrock_response(messages, character_config, request_timestamp, ui_session_id, pending_message_id)
        else:
            raise ValueError(f"Unknown llm provider {character_config.llm_provider}")
    
    async def _get_llm_output_helper(self, speaker=None):
        request_filename = None
        completed = True
        character_config = None
        next_speaker = None
        original_config = None
        pending_message_id = None  # Will be set before streaming starts
        request_timestamp = time.time()
        ui_session_id = f"session_{request_timestamp}"
        try:
            print("Playing thinking sound")
            await self.thinking_sound.play()
            if speaker is None:
                next_speaker = await self.character_manager.select_next_speaker(
                    self._get_conversation_history_for_director()
                )
            else:
                next_speaker = speaker
             
            messages, character_config, stop_sequences, request_filename = self.get_llm_context(next_speaker, request_timestamp)
            original_config = self._set_character_voice(character_config)
            print("Got context")
            print(messages)
            # Pre-calculate message ID for UI edit/delete consistency
            pending_message_id = f"msg_{len(self.state.conversation_history)}"
            text_stream = self.get_llm_text_stream(messages, character_config, request_timestamp, ui_session_id, pending_message_id)


            async def stop_thinking_for_generation():
                await self.thinking_sound.interrupt()
                if hasattr(self, 'ui_server'):
                    await self.ui_server.broadcast_speaker_status(
                        thinking_sound=False,
                        is_speaking=True,  # Now actually speaking
                        current_speaker=next_speaker
                    )
            speak_text = self.tts.speak_text(
                text_stream,
                stop_thinking_for_generation,
                self.osc_client,
                self.config.osc.speaking_start_address
            )
            self.speak_text_task = asyncio.create_task(speak_text)
            print("waiting for speak text")
            await self.speak_text_task
            print("done with speak text")
            

        except asyncio.CancelledError:
            completed = False
            await self.thinking_sound.interrupt()
            await self.tts.interrupt()
            if self.speak_text_task:
                self.speak_text_task.cancel()
                await self.speak_text_task
                self.speak_text_task = None
            print("Canceled get llm output")
            raise
        except Exception as e:
            import traceback
            print(traceback.print_exc())
            print("error in get llm output")
            # Make sure thinking sound stops on any error
            await self.thinking_sound.interrupt()
        finally:
            # Safety: always stop thinking sound when exiting
            await self.thinking_sound.interrupt()
            print("finalizae get llm output")
            if original_config is not None:
                self._restore_voice_config(original_config)
            assistant_response = self.tts.generated_text
            # Log the response
            if len(assistant_response.strip()) > 0:
                assistant_response += ("" if completed else INTERRUPTED_STR)
                assistant_response = assistant_response.strip()
                if character_config is not None and request_filename is not None:
                    response_timestamp = time.time()
                    self._log_llm_response(
                        assistant_response,
                        request_filename,
                        response_timestamp,
                        was_interrupted=not completed,
                        error=None,
                        provider=character_config.llm_provider
                    )
                    status = "completed" if completed else "interrupted"
                    assistant_turn = ConversationTurn(
                        role="assistant",
                        content=assistant_response,
                        timestamp=datetime.now(),
                        status=status,
                        character=next_speaker,
                        id=pending_message_id  # Use pre-calculated ID to match UI
                    )
                    self._add_turn_to_history(assistant_turn)
                    # For multi-character mode, log with character name prefix (no brackets)
                    self._log_conversation_turn("assistant", f"{next_speaker}: {assistant_response}")
                    await self.ui_server.broadcast(UIMessage(
                        type="ai_stream_correction",
                        data={
                            "session_id": ui_session_id,
                            "corrected_text": assistant_response,
                            "was_interrupted": False
                        }
                    ))
            else:
                await self.ui_server.broadcast(UIMessage(
                    type="ai_stream_correction",
                    data={
                        "session_id": ui_session_id,
                        "corrected_text": "",
                        "was_interrupted": False
                    }
                ))



    async def _interrupt_llm_output(self):
        try:
            if self.llm_output_task:
                print("Cancelling old llm output")
                self.llm_output_task.cancel()
                await self.llm_output_task
                self.llm_output_task = None
        except asyncio.CancelledError:
            pass # intentional
    async def _get_llm_output(self, speaker=None):
        await self._interrupt_llm_output()        
        self.llm_output_task = asyncio.create_task(self._get_llm_output_helper(speaker))

        
    
    async def _on_utterance_complete(self, result, skip_processing: bool = False):
        """Handle completed utterances from STT."""
        # Use speaker name if available from voice fingerprinting, otherwise fall back to speaker ID
        ui_speaker_name = result.speaker_name

        # Create the turn first, then add to history to get proper ID assignment
        user_turn = ConversationTurn(
            role="user",
            content=result.text,
            timestamp=datetime.fromtimestamp(result.timestamp) if isinstance(result.timestamp, (int, float)) else datetime.now(),
            status="pending",
            speaker_id=result.speaker_id,
            speaker_name=result.speaker_name,
            metadata= {"message_id": result.message_id}
        )
        self._add_turn_to_history(user_turn)  # This assigns turn.id = "msg_{index}"
        self._log_conversation_turn("user", result.text, speaker_name=result.speaker_name)

        # Broadcast AFTER adding to history so we use the correct msg_N ID
        if hasattr(self, 'ui_server'):
            await self.ui_server.broadcast_transcription(
                speaker=ui_speaker_name,
                text=result.text,
                is_final=True,
                is_edit=result.is_edit,
                message_id=user_turn.id  # Use the assigned turn.id, not the STT result UUID
            )

        # Get director mode (supports legacy director_enabled boolean)
        director_mode = getattr(self.config.conversation, 'director_mode', None)
        if director_mode is None:
            director_mode = "director" if self.config.conversation.director_enabled else "off"
        
        if director_mode == "director":
            print("Getting llm output (director mode)")
            await self._get_llm_output()
            return
        elif director_mode == "same_model":
            # Use the last AI character that spoke
            if self.state.last_ai_speaker:
                print(f"Getting llm output (same_model mode) - using {self.state.last_ai_speaker}")
                self.state.next_speaker = self.state.last_ai_speaker
                await self._get_llm_output(speaker=self.state.last_ai_speaker)
            else:
                # No previous speaker set, use default
                default_speaker = self.character_manager.get_default_character()
                print(f"Getting llm output (same_model mode) - no previous, using default: {default_speaker}")
                await self._get_llm_output(speaker=default_speaker)
            return
        else:
            # director_mode == "off" - no automatic response
            return



        if result.speaker_name:
            speaker_info = f" ({result.speaker_name})"
            ui_speaker_name = result.speaker_name
            
            # Add new speaker to detected speakers for stop sequences
            if not hasattr(self, 'detected_speakers'):
                self.detected_speakers = set()
            # Always add the speaker name if it's not None and not empty
            if result.speaker_name and result.speaker_name.strip():
                if result.speaker_name not in self.detected_speakers:
                    self.detected_speakers.add(result.speaker_name)
                    print(f"🎯 Added '{result.speaker_name}' to detected speakers for stop sequences")
                    print(f"📋 Current detected speakers: {sorted(self.detected_speakers)}")
                    
        elif result.speaker_id is not None:
            speaker_info = f" (Speaker {result.speaker_id})"
            ui_speaker_name = f"Speaker {result.speaker_id}"
        else:
            speaker_info = ""
            ui_speaker_name = "USER"
        
        # Check if this is an echo of TTS output
        # Check echo filter
        '''
        is_echo = False
        matched_tts = None
        similarity = 0.0
        if self.echo_filter:
            is_echo, matched_tts, similarity = self.echo_filter.is_echo(result.text)
        
        if is_echo:
            print(f"🔇 Ignoring echo{speaker_info}: {result.text} (matched {similarity:.0%})")
            # Still broadcast to UI but mark as echo
            if hasattr(self, 'ui_server'):
                await self.ui_server.broadcast_transcription(
                    speaker=ui_speaker_name,
                    text=f"[Echo filtered] {result.text}",
                    is_final=True
                )
            return  # Don't process as real user input
        '''

        print(f"🎯 Final{speaker_info}: {result.text}")
        
        # Broadcast final transcription
        if hasattr(self, 'ui_server'):
            await self.ui_server.broadcast_transcription(
                speaker=ui_speaker_name,
                text=result.text,
                is_final=True,
                is_edit= result.is_edit,
                message_id = result.message_id
            )
        
        # Capture image if camera is enabled
        captured_image = None
        if self.camera and self.config.camera.capture_on_speech:
            capture_result = self.camera.capture_image()
            if capture_result:
                _, jpeg_base64 = capture_result
                captured_image = jpeg_base64
                print("📸 Captured image with user speech")
        
        # Check for interruption at ANY stage of AI response
        interrupted = False
        
        if self.config.conversation.interruptions_enabled and self.tts.is_currently_playing():
            print(f"🛑 Interrupting TTS playback with: {result.text}")
            
            # Stop TTS, capture spoken content, and update UI
            spoken_content = await self._stop_tts_and_notify_ui()
            print(f"🔍 [DEBUG] Interruption - truncated spoken content: '{spoken_content}' ({len(spoken_content)} chars)")
            
            # Don't add assistant response here - character processing will handle it
            # This avoids duplicates
            interrupted = True
            
        elif self.state.is_processing_llm and self.config.conversation.interruptions_enabled:
            print(f"🛑 Interrupting LLM generation with: {result.text}")
            # Mark current processing as interrupted
            if self.state.current_processing_turn:
                self.state.current_processing_turn.status = "interrupted"
                # Add metadata to indicate this was a user speech interruption
                if not self.state.current_processing_turn.metadata:
                    self.state.current_processing_turn.metadata = {}
                self.state.current_processing_turn.metadata['interrupted_by'] = 'user_speech'
            interrupted = True
            # Stop any ongoing TTS that might be starting
            if self.tts.is_currently_playing():
                await self._stop_tts_and_notify_ui()
            
        elif self.state.is_speaking and self.config.conversation.interruptions_enabled:
            print(f"🛑 Interrupting TTS setup with: {result.text}")
            await self._stop_tts_and_notify_ui()
            # Mark current processing as interrupted by user speech
            if self.state.current_processing_turn:
                self.state.current_processing_turn.status = "interrupted"
                if not self.state.current_processing_turn.metadata:
                    self.state.current_processing_turn.metadata = {}
                self.state.current_processing_turn.metadata['interrupted_by'] = 'user_speech'
            interrupted = True
        
        if interrupted:
            # Cancel any ongoing LLM streaming task BEFORE adding the user utterance
            if self.state.current_llm_task and not self.state.current_llm_task.done():
                print("🚫 Cancelling LLM streaming task")
                self.state.current_llm_task.cancel()
                # Wait a moment for cancellation to take effect and history to be updated
                try:
                    await asyncio.wait_for(self.state.current_llm_task, timeout=0.5)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
                self.state.current_llm_task = None
            
            # Clear processing state
            self.state.is_speaking = False
            self.state.is_processing_llm = False
            self.state.current_processing_turn = None
            print("🔄 All AI processing interrupted and state cleared")
            self._save_conversation_state()
            
            # Broadcast state cleared
            if hasattr(self, 'ui_server'):
                await self.ui_server.broadcast_speaker_status(
                    current_speaker=None,
                    is_speaking=False,
                    is_processing=False,
                    thinking_sound=False
                )
        
        # Add the new user utterance to conversation history AFTER handling interruption
        # Check if we should append to an existing recent pending turn from the same speaker
        # This handles cases where Deepgram splits a continuous speech into multiple utterance events
        appended_to_existing = False
        if not captured_image and result.speaker_name:
            # Look for recent pending turns from the same speaker (within last 5 seconds)
            current_time = result.timestamp if isinstance(result.timestamp, (int, float)) else time.time()
            for turn in reversed(self.state.conversation_history):
                if turn.role == "user" and turn.status == "pending":
                    # Check if it's from the same speaker
                    if turn.speaker_name == result.speaker_name:
                        # Check if it's recent (within 5 seconds)
                        turn_time = turn.timestamp.timestamp() if hasattr(turn.timestamp, 'timestamp') else turn.timestamp
                        if isinstance(turn_time, (int, float)) and (current_time - turn_time) < 5.0:
                            # Append to existing turn
                            if isinstance(turn.content, str):
                                turn.content = turn.content + " " + result.text
                                appended_to_existing = True
                                print(f"➕ Appended to existing utterance: '{result.text}' → Full: '{turn.content}'")
                                break
        if result.is_edit:
            appended_to_existing = False
        
        if not appended_to_existing:
            # Create new turn
            # Include captured image if available
            if captured_image:
                # Create content with both text and image
                content = [
                    {"type": "text", "text": result.text},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": captured_image
                        }
                    }
                ]
            else:
                content = result.text
                
            user_turn = ConversationTurn(
                role="user",
                content=content,
                timestamp=datetime.fromtimestamp(result.timestamp) if isinstance(result.timestamp, (int, float)) else datetime.now(),
                status="pending",
                speaker_id=result.speaker_id,
                speaker_name=result.speaker_name,
                metadata= {"message_id": result.message_id}
            )
            self._add_turn_to_history(user_turn)
            self._log_conversation_turn("user", content, speaker_name=result.speaker_name)
            print(f"💬 Added new user utterance: '{result.text}'" + (" with image" if captured_image else ""))
        
        # Debounce processing to allow multiple utterances to accumulate
        # This ensures that multiple sentences get processed together
        self._schedule_debounced_processing()
        
        if self.config.development.enable_debug_mode:
            self.logger.debug(f"Utterance added to history")
    
    async def _on_interim_result(self, result):

        if result.speaker_name:
            ui_speaker_name = result.speaker_name
        elif result.speaker_id is not None:
            ui_speaker_name = f"Speaker {result.speaker_id}"
        else:
            ui_speaker_name = "USER"
        
        # Broadcast to UI
        if hasattr(self, 'ui_server'):
            await self.ui_server.broadcast_transcription(
                speaker=ui_speaker_name,
                text=result.text,
                is_interim=True
            )
        if self.config.conversation.interruptions_enabled:
           await self._interrupt_llm_output()            
        return
        
        """Handle interim results for showing progress and immediate interruption."""
        # Determine speaker name for UI
        
        # Show interim results based on configuration
        if self.show_interim:
            print(f"💭 Interim: {result.text}")
            
        # Broadcast to UI
        if hasattr(self, 'ui_server'):
            await self.ui_server.broadcast_transcription(
                speaker=ui_speaker_name,
                text=result.text,
                is_interim=True
            )
        
        # IMMEDIATE INTERRUPTION: Interrupt as soon as ANY user text is detected
        if self.config.conversation.interruptions_enabled and result.text.strip():
            # Check if something is actively playing/processing that we should interrupt
            should_interrupt = (self.tts.is_currently_playing() or 
                              self.state.is_processing_llm or 
                              self.state.is_speaking)
            
            if should_interrupt:
                return # stap we don't need this
                # Only interrupt if audio has actually played (allow pre-audio speech)
                if self.tts.is_currently_playing():
                    print(f"🛑 INSTANT INTERRUPTION triggered by: '{result.text}' (conf: {result.confidence:.2f})")
                    
                    # Stop TTS if playing
                    if self.tts.is_currently_playing():
                        await self._stop_tts_and_notify_ui()
                    
                    # Cancel ongoing LLM streaming task
                    if self.state.current_llm_task and not self.state.current_llm_task.done():
                        self.state.current_llm_task.cancel()
                        try:
                            await self.state.current_llm_task
                        except asyncio.CancelledError:
                            pass
                    
                    # Mark current processing as interrupted
                    if self.state.current_processing_turn:
                        self.state.current_processing_turn.status = "interrupted"
                        if not self.state.current_processing_turn.metadata:
                            self.state.current_processing_turn.metadata = {}
                        self.state.current_processing_turn.metadata['interrupted_by'] = 'user_speech_interim'
                        self.state.current_processing_turn.metadata['interim_text'] = result.text
                    
                    # Clear processing flags
                    self.state.is_processing_llm = False
                    self.state.is_speaking = False
    
    async def _on_speech_started(self, data):
        """Handle speech start events."""
        # Interruption now happens in _on_interim_result when sufficient confidence is detected
        # This event is just for logging/debugging
        if self.tts.is_currently_playing():
            print("🎤 Speech detected during TTS - interruption will trigger on interim results")
        elif self.state.is_speaking:
            print("🎤 Speech detected during conversation processing")
        
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
        
        # Add new speaker to detected speakers for stop sequences
        if not hasattr(self, 'detected_speakers'):
            self.detected_speakers = set()
        # Always add the speaker name if it's not None and not empty
        if speaker_name and speaker_name.strip():
            if speaker_name not in self.detected_speakers:
                self.detected_speakers.add(speaker_name)
                print(f"🎯 Added '{speaker_name}' to detected speakers for stop sequences")
                print(f"📋 Current detected speakers: {sorted(self.detected_speakers)}")
        
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
    
    async def _handle_tool_execution(self, tool_call):
        """Handle tool execution callback from TTS module."""
        from async_tts_module import ToolCall, ToolResult
        
        print(f"🔧 Executing tool: {tool_call.tag_name}")
        print(f"   Content: {tool_call.content}")
        print(f"   Position: {tool_call.start_position}-{tool_call.end_position}")
        
        try:
            # Create context for tool execution
            context = {
                'conversation_history': self.state.conversation_history,
                'current_session': self.tts.current_session,
                'config': self.config,
                'current_speaker': self.state.current_speaker,
                'osc_client': self.osc_client if hasattr(self, 'osc_client') else None,
                'send_osc_color_change': self._send_osc_color_change
            }
            
            # Execute tool using registry
            result = await self.tool_registry.execute_tool(tool_call, context)
            
            # Log tool execution result to conversation history
            if result.content:
                tool_turn = ConversationTurn(
                    role="tool",
                    content=f"[{tool_call.tag_name}] {result.content}",
                    timestamp=datetime.now(),
                    status="completed"
                )
                self._add_turn_to_history(tool_turn)
                self._log_conversation_turn("tool", tool_turn.content)
            
            # If tool wants to interrupt and has content, we may need to speak it
            if result.should_interrupt and result.content:
                # Store for later processing after current TTS is interrupted
                self.state.pending_tool_response = result.content
                self._save_conversation_state()
            
            return result
                    
        except Exception as e:
            print(f"❌ Error executing tool: {e}")
            self.logger.error(f"Tool execution error: {e}")
            return ToolResult(should_interrupt=False, content=None)
    
    def _schedule_debounced_processing(self):
        """Schedule processing after a debounce delay to accumulate multiple utterances."""
        # Cancel existing debounce timer if any
        if self._utterance_debounce_task and not self._utterance_debounce_task.done():
            self._utterance_debounce_task.cancel()
        
        # Get debounce delay from config (use pause_threshold)
        debounce_delay = self.config.conversation.pause_threshold
        
        async def _debounced_trigger():
            """Wait for debounce delay, then trigger processing."""
            try:
                pending_before = sum(1 for turn in self.state.conversation_history 
                                   if turn.role == "user" and turn.status == "pending")
                await asyncio.sleep(debounce_delay)
                # After delay, trigger processing
                pending_after = sum(1 for turn in self.state.conversation_history 
                                  if turn.role == "user" and turn.status == "pending")
                print(f"⏰ Debounce timer expired ({debounce_delay}s) - processing {pending_after} accumulated utterance(s)")
                await self._process_pending_utterances()
            except asyncio.CancelledError:
                # Timer was cancelled by a new utterance - this is expected
                pending_now = sum(1 for turn in self.state.conversation_history 
                                if turn.role == "user" and turn.status == "pending")
                print(f"⏰ Debounce timer reset - now {pending_now} utterance(s) pending")
                pass
        
        # Start new debounce timer
        self._utterance_debounce_task = asyncio.create_task(_debounced_trigger())
        pending_count = sum(1 for turn in self.state.conversation_history 
                          if turn.role == "user" and turn.status == "pending")
        print(f"⏰ Debounce timer started ({debounce_delay}s) - {pending_count} utterance(s) pending")
    
    async def _process_pending_utterances(self):
        """Process pending utterances from conversation history."""
        # Always increment generation for new utterance processing
        self.state.current_generation += 1
        current_generation = self.state.current_generation
        print(f"📍 New utterance processing generation: {current_generation}")
        
        # Cancel any existing processing task
        if self._processing_task and not self._processing_task.done():
            print("🚫 Cancelling previous processing task")
            self._processing_task.cancel()
            # Don't wait for it - let it cancel in background
            
        # Create new processing task
        self._processing_task = asyncio.create_task(
            self._do_process_pending_utterances(current_generation)
        )
        
    async def _do_process_pending_utterances(self, generation: int):
        """Actually process pending utterances with generation tracking."""
        try:
            # Find pending user utterances
            pending_turns = [turn for turn in self.state.conversation_history 
                            if turn.role == "user" and turn.status == "pending"]
            
            if not pending_turns:
                return
                
            # Check if we're still the current generation
            if generation != self.state.current_generation:
                print(f"🚫 Processing cancelled - newer request exists (gen {generation} vs {self.state.current_generation})")
                return
                
            # Process all pending turns
            # Extract text content for display
            text_contents = []
            for turn in pending_turns:
                if isinstance(turn.content, str):
                    text_contents.append(turn.content)
                elif isinstance(turn.content, list):
                    # Extract text from content blocks
                    for item in turn.content:
                        if item.get("type") == "text":
                            text_contents.append(item.get("text", ""))
            
            combined_text = " ".join(text_contents)
            print(f"🧠 Processing {len(pending_turns)} pending utterances: '{combined_text}'")
            
            # Mark all pending turns as processing
            for turn in pending_turns:
                turn.status = "processing"
            self._save_conversation_state()
            
            # Check generation again before processing
            if generation != self.state.current_generation:
                print(f"🚫 Processing cancelled before LLM - newer request exists")
                # Reset status for all turns
                for turn in pending_turns:
                    turn.status = "pending"
                self._save_conversation_state()
                return
                
            # For now, pass the combined text. The actual content (including images) 
            # will be properly handled when building messages for the LLM
            await self._process_with_llm(combined_text, pending_turns[0], generation)  # Pass first turn for reference
            
        except asyncio.CancelledError:
            print("🚫 Processing task cancelled")
            # Reset status for pending turns
            for turn in self.state.conversation_history:
                if turn.role == "user" and turn.status == "processing":
                    turn.status = "pending"
            self._save_conversation_state()
            raise
        except Exception as e:
            print(f"❌ Error in processing: {e}")
            import traceback
            traceback.print_exc()
    
    async def _process_with_character_llm(self, user_input: str, reference_turn: ConversationTurn, generation: int = None, prepared_statement_name: str = None):
        """Process user input with multi-character system."""
        ui_session_id: Optional[str] = None
        try:
            self.state.is_processing_llm = True
            
            # Check if this is the current generation
            if generation is not None:
                if generation != self.state.current_generation:
                    print(f"🚫 Character LLM processing cancelled - stale generation {generation} (current: {self.state.current_generation})")
                    self.state.is_processing_llm = False
                    return
            else:
                # If no generation specified, this is a new request
                # Only increment if we're not already processing
                if not self.state.is_processing_llm:
                    self.state.current_generation += 1
                generation = self.state.current_generation
                print(f"📍 Generation: {generation}")

            # Ensure the most recent turn is from the user before generating
            reference_metadata = reference_turn.metadata if getattr(reference_turn, "metadata", None) else {}
            is_manual_override = bool(
                reference_metadata.get("is_manual_trigger")
                or reference_metadata.get("is_prepared_statement")
                or reference_metadata.get("force_generation")
                or prepared_statement_name is not None
            )

            if not is_manual_override:
                last_turn = next(
                    (turn for turn in reversed(self.state.conversation_history)
                     if turn.role not in {"system"}),
                    None
                )
                if False and (not last_turn or last_turn.role != "user"):
                    print("🚫 Skipping LLM generation - last message not from user")
                    for turn in self.state.conversation_history:
                        if turn.role == "user" and turn.status == "processing":
                            turn.status = "pending"
                    async with self.state.speaker_lock:
                        self.state.next_speaker = None
                    self.state.is_processing_llm = False
                    self._save_conversation_state()
                    if hasattr(self, 'ui_server'):
                        await self.ui_server.broadcast_speaker_status(
                            is_processing=False,
                            pending_speaker=None,
                            thinking_sound=False
                        )
                    return
            else:
                print("🔓 Manual override: proceeding with LLM generation despite missing user turn")
            
            # Increment director generation and store it
            self._director_generation += 1
            director_gen = self._director_generation
            
            # Start thinking sound with director generation
            await self.thinking_sound.play()
            
            # Broadcast thinking sound status
            if hasattr(self, 'ui_server'):
                await self.ui_server.broadcast_speaker_status(thinking_sound=True)
            
            # Determine next speaker with global lock
            async with self.state.speaker_lock:
                # Check if we already have a next speaker set globally
                if self.state.next_speaker:
                    next_speaker = self.state.next_speaker
                    print(f"📢 Using globally set next speaker: {next_speaker}")
                    # Clear it after use
                    self.state.next_speaker = None
                else:
                    # Check if this is a manual trigger from UI
                    next_speaker = None
                    if hasattr(reference_turn, 'metadata') and reference_turn.metadata:
                        if reference_turn.metadata.get('is_manual_trigger'):
                            next_speaker = reference_turn.metadata.get('triggered_speaker')
                            print(f"🎯 Using manually triggered speaker: {next_speaker}")
                    
                    # If no manual trigger, check director mode
                    if not next_speaker:
                        # Get director mode (supports legacy director_enabled boolean)
                        director_mode = getattr(self.config.conversation, 'director_mode', None)
                        if director_mode is None:
                            director_mode = "director" if self.config.conversation.director_enabled else "off"
                        
                        if director_mode == "director":
                            # Pass detected speakers to character manager for validation
                            if hasattr(self, 'detected_speakers'):
                                self.character_manager._detected_speakers = self.detected_speakers
                                
                            next_speaker = await self.character_manager.select_next_speaker(
                                self._get_conversation_history_for_director()
                            )
                        elif director_mode == "same_model":
                            # Use the last AI character that spoke
                            if self.state.last_ai_speaker:
                                next_speaker = self.state.last_ai_speaker
                                print(f"🔄 Same model mode - using {next_speaker}")
                            else:
                                # No previous speaker, use first character
                                next_speaker = self.character_manager.get_default_character()
                                print(f"🔄 Same model mode - no previous, using default: {next_speaker}")
                        else:
                            # Director is disabled, default to USER
                            next_speaker = "USER"
                            print("🚫 Director disabled - defaulting to USER")
                
                # Set as current speaker
                self.state.current_speaker = next_speaker
                # Track last AI speaker for same_model mode
                if next_speaker and next_speaker != "USER" and not next_speaker.startswith("User"):
                    self.state.last_ai_speaker = next_speaker
                    print(f"📌 Tracking last AI speaker: {next_speaker}")
            
            # Broadcast pending speaker
            if hasattr(self, 'ui_server'):
                await self.ui_server.broadcast_speaker_status(
                    pending_speaker=next_speaker,
                    is_processing=True
                )
            
            # Check both processing generation and director generation
            if generation is not None and generation != self.state.current_generation:
                print(f"🚫 Processing cancelled after director - newer utterance exists (gen {generation} vs current {self.state.current_generation})")
                await self.thinking_sound.interrupt()
                self.state.is_processing_llm = False
                return
                
            if director_gen != self._director_generation:
                print(f"🚫 Processing cancelled after director - newer director request exists")
                await self.thinking_sound.interrupt()
                self.state.is_processing_llm = False
                return
            
            if next_speaker == "USER" or next_speaker is None:
                print("🎭 Director: User should speak next")
                await self.thinking_sound.interrupt()
                self.state.is_processing_llm = False
                return
            
            # Set active character
            self.character_manager.set_active_character(next_speaker)
            character_config = self.character_manager.get_character_config(next_speaker)
            
            if not character_config:
                print(f"❌ Unknown character: {next_speaker}")
                await self.thinking_sound.interrupt()
                self.state.is_processing_llm = False
                return
            
            print(f"🎭 {next_speaker} is responding...")
            
            # Send OSC message for character speaking start
            self._send_osc_speaking_start(next_speaker)
            
            # Broadcast current speaker
            if hasattr(self, 'ui_server'):
                await self.ui_server.broadcast_speaker_status(
                    current_speaker=next_speaker,
                    is_speaking=False,  # Not speaking yet, just processing
                    is_processing=True
                )
            
            # Request timestamp for logging
            request_timestamp = time.time()
            ui_session_id = f"session_{request_timestamp}"
            self.state.current_ui_session_id = ui_session_id
            
            # Check if we should use prefill format
            use_prefill = (self.config.conversation.conversation_mode == "prefill" and 
                          character_config.llm_provider in ["anthropic", "bedrock"])
            
            if use_prefill:
                # For prefill mode, convert directly from raw history
                prefill_name = character_config.prefill_name or character_config.name
                raw_history = self._get_conversation_history_for_character()
                
                # Get character histories from current context if available
                context_char_histories = None
                if self.context_manager:
                    active_context = self.context_manager.get_active_context()
                    if active_context:
                        context_char_histories = active_context.character_histories
                        print(f"🔍 DEBUG: Active context '{active_context.config.name}' has character_histories: {list(context_char_histories.keys()) if context_char_histories else None}")
                
                print(f"🔍 DEBUG: Creating prefill messages for character '{next_speaker}' (prefill_name: '{prefill_name}')")
                print(f"🔍 DEBUG: Context has character histories: {context_char_histories is not None}")
                if context_char_histories:
                    print(f"🔍 DEBUG: Character histories available for: {list(context_char_histories.keys())}")
                
                # Create messages in prefill format directly
                messages = self._create_character_prefill_messages(
                    raw_history, 
                    next_speaker, 
                    prefill_name,
                    character_config.system_prompt,
                    context_char_histories
                )
            else:
                # Standard chat format
                # Get character histories from current context if available
                context_char_histories = None
                if self.context_manager:
                    active_context = self.context_manager.get_active_context()
                    if active_context:
                        context_char_histories = active_context.character_histories
                
                messages = self.character_manager.format_messages_for_character(
                    next_speaker,
                    self._get_conversation_history_for_character(),
                    context_char_histories
                )
            
            # Check if we're in prefill format to generate stop sequences
            stop_sequences = None
            if self.config.conversation.conversation_mode == "prefill":
                stop_sequences = []
                # Add all character names as stop sequences
                for char_name in self.character_manager.characters.keys():
                    stop_sequences.append(f"\n\n{char_name}:")
                    # Also add prefill names if different
                    char_config_temp = self.character_manager.get_character_config(char_name)
                    if char_config_temp and char_config_temp.prefill_name and char_config_temp.prefill_name != char_name:
                        stop_sequences.append(f"\n\n{char_config_temp.prefill_name}:")
                
                # Add human names
                if hasattr(self.config.conversation, 'prefill_participants'):
                    human_name = self.config.conversation.prefill_participants[0]
                    stop_sequences.append(f"\n\n{human_name}:")
                
                # Add any detected speakers
                if hasattr(self, 'detected_speakers') and self.detected_speakers:
                    for speaker in self.detected_speakers:
                        speaker_stop = f"\n\n{speaker}:"
                        if speaker_stop not in stop_sequences:
                            stop_sequences.append(speaker_stop)
                
                # Add System: to stop sequences
                stop_sequences.append("\n\nSystem:")
                stop_sequences.append("\n\nA:")
                
                # Add User 1 through User 10
                for i in range(1, 11):
                    stop_sequences.append(f"\n\nUser {i}:")
                
                # Add interrupted marker
                stop_sequences.append("[Interrupted by user]")
                
                # Remove duplicates while preserving order
                stop_sequences = list(dict.fromkeys(stop_sequences))
            
            # Log the request
            request_filename = self._log_llm_request(
                messages, 
                character_config.llm_model, 
                request_timestamp,
                character_config.llm_provider,
                stop_sequences
            )
            active_context_name = None
            context_snapshot = None
            if self.context_manager:
                active_context_name = self.context_manager.active_context_name
                context = self.context_manager.get_active_context()
                if context and getattr(context, "metadata", None):
                    context_snapshot = dict(context.metadata)
            print(f"🧭 Sending request for {next_speaker} via {character_config.llm_provider} (model={character_config.llm_model})")
            if active_context_name:
                print(f"   📂 Active context: {active_context_name}")
                if context_snapshot:
                    print(f"   📝 Context metadata: {context_snapshot}")
            print(f"   📄 Request log: {request_filename}")
            print(messages)
            
            # Check both generations before starting LLM
            if generation is not None and generation != self.state.current_generation:
                print(f"🚫 Processing cancelled before LLM call - newer utterance exists (gen {generation} vs current {self.state.current_generation})")
                await self.thinking_sound.interrupt()
                self.state.is_processing_llm = False
                return
                
            if director_gen != self._director_generation:
                print(f"🚫 Processing cancelled before LLM call - newer director request exists")
                await self.thinking_sound.interrupt()
                self.state.is_processing_llm = False
                return
            
            # Set callback to stop thinking sound when first audio arrives
            # Create a closure that captures the current director generation and speaker
            async def stop_thinking_for_generation():
                await self.thinking_sound.interrupt()
                if hasattr(self, 'ui_server'):
                    await self.ui_server.broadcast_speaker_status(
                        thinking_sound=False,
                        is_speaking=True,  # Now actually speaking
                        current_speaker=next_speaker
                    )
            
            self.tts.first_audio_callback = stop_thinking_for_generation

            # Stream response based on provider
            assistant_response = ""

            # Pre-calculate the message ID that will be assigned when the turn is added to history
            # This ensures the UI gets the correct ID for edit/delete operations
            pending_message_id = f"msg_{len(self.state.conversation_history)}"
            
            # Check if this is a prepared statement request
            if prepared_statement_name and prepared_statement_name in character_config.prepared_statements:
                print(f"📜 Using prepared statement '{prepared_statement_name}' for {next_speaker}")
                prepared_text = character_config.prepared_statements[prepared_statement_name]
                
                # Temporarily set character voice
                original_config = self._set_character_voice(character_config)
                try:
                    # Create a simple async generator that yields the prepared text
                    async def prepared_text_generator():
                        yield prepared_text
                        '''
                        # Yield the text in chunks for natural streaming
                        chunk_size = 50  # Characters per chunk
                        for i in range(0, len(prepared_text), chunk_size):
                            chunk = prepared_text[i:i + chunk_size]
                            yield chunk
                            # Small delay to simulate natural streaming
                            await asyncio.sleep(0.02)
                        '''
                    # Create TTS task for the prepared statement
                    self.state.current_llm_task = asyncio.create_task(
                        self.tts.speak_text(prepared_text_generator(), stop_thinking_for_generation)
                    )
                    
                    # Wait for completion
                    try:
                        completed = await self.state.current_llm_task
                    except asyncio.CancelledError:
                        print("⚠️ Prepared statement playback was interrupted")
                        completed = False
                    
                    # Update echo filter
                    session_id = f"prepared_{character_config.name}_{request_timestamp}"
                    #if self.echo_filter:
                    #    if completed:
                    #        self.echo_filter.on_tts_complete(session_id)
                    #    else:
                    #        spoken_text = self.tts.get_spoken_text_heuristic() if hasattr(self.tts, 'get_spoken_text_heuristic') else None
                    #        self.echo_filter.on_tts_interrupted(session_id, spoken_text)
                    
                    # Capture the response
                    #assistant_response = prepared_text if completed else self.tts.get_spoken_text_heuristic().strip()
                    assistant_response = prepared_text
                    
                    # Send to UI with the pre-calculated message_id
                    if hasattr(self, 'ui_server'):
                        await self.ui_server.broadcast_ai_stream(
                            speaker=next_speaker,
                            text=prepared_text,
                            session_id=ui_session_id,
                            is_complete=True,
                            message_id=pending_message_id
                        )
                finally:
                    # Restore original voice config
                    self._restore_voice_config(original_config)

            elif character_config.llm_provider == "openai":
                # Temporarily set character voice
                original_config = self._set_character_voice(character_config)
                try:
                    # Create TTS task for this character
                    self.state.current_llm_task = asyncio.create_task(
                        self.tts.speak_text(
                            self._stream_character_openai_response(messages, character_config, request_timestamp, ui_session_id, pending_message_id),
                            stop_thinking_for_generation
                        )
                    )
                    
                    # Wait for completion
                    try:
                        completed = await self.state.current_llm_task
                    except asyncio.CancelledError:
                        print("⚠️ Character response was interrupted")
                        completed = False
                    
                    # Update echo filter with completion status
                    session_id = f"openai_{character_config.name}_{request_timestamp}"
                    if completed:
                        if self.echo_filter:
                            self.echo_filter.on_tts_complete(session_id)
                    else:
                        # Get spoken text if interrupted
                        spoken_text = None
                        if hasattr(self.tts, 'get_spoken_text_heuristic'):
                            spoken_text = self.tts.get_spoken_text_heuristic()
                        if self.echo_filter:
                            self.echo_filter.on_tts_interrupted(session_id, spoken_text)
                    
                    # Get the appropriate text based on whether it was interrupted
                    if hasattr(self.tts, 'current_session') and self.tts.current_session:
                        if completed:
                            # Use full generated text for completed responses
                            assistant_response = self.tts.generated_text.strip()
                        else:
                            # Use spoken heuristic for interrupted responses
                            assistant_response = self.tts.get_spoken_text_heuristic().strip()
                        print(f"📝 Captured assistant response: {len(assistant_response)} chars")
                    else:
                        print("⚠️ No TTS session or generated text available")
                finally:
                    # Restore original voice config
                    self._restore_voice_config(original_config)
            elif character_config.llm_provider == "anthropic":
                # Temporarily set character voice
                original_config = self._set_character_voice(character_config)
                try:
                    # Create TTS task for this character
                    self.state.current_llm_task = asyncio.create_task(
                        self.tts.speak_text(
                            self._stream_character_anthropic_response(messages, character_config, request_timestamp, ui_session_id, pending_message_id),
                            stop_thinking_for_generation
                        )
                    )
                    
                    # Wait for completion
                    try:
                        completed = await self.state.current_llm_task
                    except asyncio.CancelledError:
                        print("⚠️ Character response was interrupted")
                        completed = False
                    
                    # Update echo filter with completion status
                    session_id = f"anthropic_{character_config.name}_{request_timestamp}"
                    if completed:
                        if self.echo_filter:
                            self.echo_filter.on_tts_complete(session_id)
                    else:
                        # Get spoken text if interrupted
                        spoken_text = None
                        if hasattr(self.tts, 'get_spoken_text_heuristic'):
                            spoken_text = self.tts.get_spoken_text_heuristic()
                        if self.echo_filter:
                            self.echo_filter.on_tts_interrupted(session_id, spoken_text)
                    
                    # Store assistant response
                    assistant_response = ""
                    if hasattr(self.tts, 'current_session') and self.tts.current_session:
                        if completed:
                            assistant_response = (self.tts.generated_text or "").strip()
                        else:
                            assistant_response = self.tts.get_spoken_text_heuristic().strip()
                            if not assistant_response and hasattr(self.tts, 'get_spoken_text'):
                                assistant_response = (self.tts.get_spoken_text() or "").strip()
                    elif completed:
                        assistant_response = (getattr(self.tts, 'last_session_generated_text', '') or "").strip()
                    
                    if assistant_response:
                        print(f"📝 Captured assistant response: {len(assistant_response)} chars (completed={completed})")
                    else:
                        print("⚠️ No TTS session or generated text available (anthropic)")
                finally:
                    # Restore original voice config
                    self._restore_voice_config(original_config)
            elif character_config.llm_provider == "bedrock":
                # Temporarily set character voice
                original_config = self._set_character_voice(character_config)
                try:
                    # Create TTS task for this character
                    self.state.current_llm_task = asyncio.create_task(
                        self.tts.speak_text(
                            self._stream_character_bedrock_response(messages, character_config, request_timestamp, ui_session_id, pending_message_id),
                            stop_thinking_for_generation
                        )
                    )
                    
                    # Wait for completion
                    try:
                        completed = await self.state.current_llm_task
                    except asyncio.CancelledError:
                        print("⚠️ Character response was interrupted")
                        completed = False
                    
                    # Update echo filter with completion status
                    session_id = f"anthropic_{character_config.name}_{request_timestamp}"
                    if completed:
                        if self.echo_filter:
                            self.echo_filter.on_tts_complete(session_id)
                    else:
                        # Get spoken text if interrupted
                        spoken_text = None
                        if hasattr(self.tts, 'get_spoken_text_heuristic'):
                            spoken_text = self.tts.get_spoken_text_heuristic()
                        if self.echo_filter:
                            self.echo_filter.on_tts_interrupted(session_id, spoken_text)
                    
                    # Get the appropriate text based on whether it was interrupted
                    if hasattr(self.tts, 'current_session') and self.tts.current_session:
                        if completed:
                            # Use full generated text for completed responses
                            assistant_response = self.tts.generated_text.strip()
                        else:
                            # Use spoken heuristic for interrupted responses
                            assistant_response = self.tts.get_spoken_text_heuristic().strip()
                        print(f"📝 Captured assistant response: {len(assistant_response)} chars")
                    else:
                        print("⚠️ No TTS session or generated text available")
                finally:
                    # Restore original voice config
                    self._restore_voice_config(original_config)
            
            if not completed:
                assistant_response = self._get_spoken_text_with_fallback(assistant_response)
            
            # Log the response
            if assistant_response.strip():
                response_timestamp = time.time()
                self._log_llm_response(
                    assistant_response,
                    request_filename,
                    response_timestamp,
                    was_interrupted=not completed,
                    error=None,
                    provider=character_config.llm_provider
                )
                print(f"✅ Logged assistant response: {len(assistant_response)} chars")
            else:
                print(f"⚠️ No assistant response to log (empty or whitespace)")
            
            # Add character's response to conversation history
            if assistant_response.strip():
                print(f"💬 Adding assistant response to history: {len(assistant_response)} chars from {next_speaker}")
                # Determine status based on completion
                status = "completed" if completed else "interrupted"
                assistant_turn = ConversationTurn(
                    role="assistant",
                    content=assistant_response,
                    timestamp=datetime.now(),
                    status=status,
                    character=next_speaker,
                    id=pending_message_id  # Use the pre-calculated ID to match UI
                )
                self._add_turn_to_history(assistant_turn)
                # For multi-character mode, log with character name prefix (no brackets)
                self._log_conversation_turn("assistant", f"{next_speaker}: {assistant_response}")
                print(f"✅ Added assistant response to conversation history with ID {pending_message_id}")

                # If the response was interrupted, add a system message only if it was interrupted by user speech
                if status == "interrupted":
                    # Check if this was interrupted by user speech
                    interrupted_by_user = False
                    if self.state.current_processing_turn and self.state.current_processing_turn.metadata:
                        interrupted_by_user = self.state.current_processing_turn.metadata.get('interrupted_by') == 'user_speech'

                    if interrupted_by_user:
                        system_message = f"[{next_speaker} was interrupted by user speaking]"
                        system_turn = ConversationTurn(
                            role="system",
                            content=system_message,
                            timestamp=datetime.now(),
                            status="completed"
                        )
                        self._add_turn_to_history(system_turn)
                        self._log_conversation_turn("system", system_message)
                        print(f"📝 Added interruption notice to conversation history")
                
                # Add to character manager context
                self.character_manager.add_turn_to_context(next_speaker, assistant_response)
            else:
                print(f"⚠️ Skipping empty assistant response")
            
            # After character speaks, check if another character should speak
            await asyncio.sleep(0.5)  # Brief pause
            
            # Only continue if this is still the current generation
            if generation == self.state.current_generation:
                # Recursively check for next speaker (with depth limit)
                if not hasattr(self, '_character_depth'):
                    self._character_depth = 0
                
                self._character_depth += 1
                if self._character_depth < 3:  # Max 3 characters in a row
                    # Use director mode to determine next speaker if needed
                    async with self.state.speaker_lock:
                        # Get director mode (supports legacy director_enabled boolean)
                        director_mode = getattr(self.config.conversation, 'director_mode', None)
                        if director_mode is None:
                            director_mode = "director" if self.config.conversation.director_enabled else "off"
                        
                        if not self.state.next_speaker:
                            if director_mode == "director":
                                # Director determines next speaker
                                if hasattr(self, 'detected_speakers'):
                                    self.character_manager._detected_speakers = self.detected_speakers
                                self.state.next_speaker = await self.character_manager.select_next_speaker(
                                    self._get_conversation_history_for_director()
                                )
                                print(f"🎭 Director selected next speaker: {self.state.next_speaker}")
                            elif director_mode == "same_model":
                                # Keep the last AI character
                                if self.state.last_ai_speaker:
                                    self.state.next_speaker = self.state.last_ai_speaker
                                    print(f"🔄 Same model - continuing with: {self.state.next_speaker}")
                                else:
                                    self.state.next_speaker = "USER"
                            else:
                                # No director, default to USER
                                self.state.next_speaker = "USER"
                                print(f"📢 No director - next speaker set to USER")
                    
                    # Continue with the same generation number
                    await self._process_with_character_llm("", reference_turn, generation)
                else:
                    self._character_depth = 0
            else:
                print(f"⏭️ Skipping recursive call - generation {generation} is stale (current: {self.state.current_generation})")
                
        except Exception as e:
            import traceback
            print(traceback.print_exc())
            print(f"❌ Character LLM error: {e}")
            self.logger.error(f"Character LLM error: {e}")
        finally:
            print("Character LLM Finished processing")
            # Always stop thinking sound (no generation means force stop)
            await self.thinking_sound.interrupt()
            if getattr(self.state, "current_ui_session_id", None) == ui_session_id:
                self.state.current_ui_session_id = None
            self.state.is_processing_llm = False
            
            # Clear current speaker when done
            async with self.state.speaker_lock:
                # Send OSC stop message before clearing current speaker
                if self.state.current_speaker:
                    self._send_osc_speaking_stop(self.state.current_speaker)
                    # Send blank output message for TouchDesigner laser rendering
                    self._send_osc_blank_output()
                self.state.current_speaker = None
            
            # Broadcast that speaking/processing has ended
            if hasattr(self, 'ui_server'):
                await self.ui_server.broadcast_speaker_status(
                    current_speaker=None,
                    is_speaking=False,
                    is_processing=False,
                    thinking_sound=False
                )
    
    async def _stream_character_openai_response(self, messages, character_config, request_timestamp, ui_session_id: Optional[str] = None, pending_message_id: Optional[str] = None):
        """Stream response from OpenAI for a specific character."""
        client = self.character_manager.get_character_client(character_config.name)
        ui_session_id = ui_session_id or f"session_{request_timestamp}"

        # Create session ID for echo tracking
        session_id = f"openai_{character_config.name}_{request_timestamp}"
        if self.echo_filter:
            self.echo_filter.on_tts_start(session_id, character_config.name)

        try:
            response = await client.chat.completions.create(
                model=character_config.llm_model,
                messages=messages,
                stream=True,
                max_tokens=character_config.max_tokens,
                temperature=character_config.temperature
            )

            async for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    # Track for echo filter
                    if self.echo_filter:
                        self.echo_filter.on_tts_chunk(session_id, content)
                    yield content
                    # Broadcast to UI
                    if hasattr(self, 'ui_server'):
                        await self.ui_server.broadcast_ai_stream(
                            speaker=character_config.name,
                            text=content,
                            session_id=ui_session_id
                        )

        except Exception as e:
            print(f"❌ OpenAI streaming error for {character_config.name}: {e}")
            raise
        finally:
            # Send completion signal to UI with the pre-calculated message_id
            if hasattr(self, 'ui_server'):
                await self.ui_server.broadcast_ai_stream(
                    speaker=character_config.name,
                    text="",
                    session_id=ui_session_id,
                    is_complete=True,
                    message_id=pending_message_id
                )
    
    async def _stream_character_anthropic_response(self, messages, character_config, request_timestamp, ui_session_id: Optional[str] = None, pending_message_id: Optional[str] = None):
        """Stream response from Anthropic for a specific character."""
        client = self.character_manager.get_character_client(character_config.name)
        ui_session_id = ui_session_id or f"session_{request_timestamp}"

        # Create session ID for echo tracking
        session_id = f"anthropic_{character_config.name}_{request_timestamp}"
        if self.echo_filter:
            self.echo_filter.on_tts_start(session_id, character_config.name)
        
        try:
            # Check if messages are already in prefill format
            is_prefill_format = False
            prefill_name = None
            
            # Check if last message is assistant (typical prefill pattern)
            if len(messages) > 0:
                last_msg = messages[-1]
                if last_msg.get("role") == "assistant":
                    is_prefill_format = True
                    # Extract prefill name from the assistant message content
                    content = last_msg.get("content", "")
                    if content.endswith(":"):
                        # Extract the name before the final colon
                        parts = content.rsplit("\n", 1)
                        if len(parts) > 0:
                            last_line = parts[-1].strip()
                            if last_line.endswith(":"):
                                prefill_name = last_line[:-1]
            
            messages_to_send = messages
            
            # Extract system content and prepare messages
            system_content = ""
            anthropic_messages = []
            
            for msg in messages_to_send:
                if msg["role"] == "system":
                    system_content = msg["content"]
                else:
                    # Create a clean copy without metadata
                    clean_msg = {"role": msg["role"], "content": msg["content"]}
                    anthropic_messages.append(clean_msg)
            
            # Generate stop sequences for multi-character conversations
            stop_sequences = []
            if is_prefill_format:
                # Add all character names as stop sequences
                for char_name in self.character_manager.characters.keys():
                    stop_sequences.append(f"\n\n{char_name}:")
                    # Also add prefill names if different
                    char_config = self.character_manager.get_character_config(char_name)
                    if char_config and char_config.prefill_name and char_config.prefill_name != char_name:
                        stop_sequences.append(f"\n\n{char_config.prefill_name}:")
                
                # Add human names
                if hasattr(self.config.conversation, 'prefill_participants'):
                    human_name = self.config.conversation.prefill_participants[0]
                    stop_sequences.append(f"\n\n{human_name}:")
                
                # Add any detected speakers
                if hasattr(self, 'detected_speakers') and self.detected_speakers:
                    for speaker in self.detected_speakers:
                        speaker_stop = f"\n\n{speaker}:"
                        if speaker_stop not in stop_sequences:
                            stop_sequences.append(speaker_stop)
                
                # Add System: to stop sequences
                stop_sequences.append("\n\nSystem:")
                stop_sequences.append("\n\nA:")
                
                # Add User 1 through User 10
                for i in range(1, 11):
                    stop_sequences.append(f"\n\nUser {i}:")
                
                # Add interrupted marker
                stop_sequences.append("[Interrupted by user]")
                
                # Remove duplicates while preserving order
                stop_sequences = list(dict.fromkeys(stop_sequences))
                print(f"🛑 Character using stop sequences: {stop_sequences}")
            
            response = await client.messages.create(
                model=character_config.llm_model,
                messages=anthropic_messages,
                system=system_content,
                stream=True,
                max_tokens=character_config.max_tokens,
                temperature=character_config.temperature,
                stop_sequences=stop_sequences if stop_sequences else None
            )
            
            # Track if we need to skip the prefill prefix
            skip_prefix = is_prefill_format
            prefix_buffer = "" if skip_prefix else None
            async for chunk in response:
                if chunk.type == "content_block_delta" and chunk.delta.text:
                    text = chunk.delta.text
                    
                    # Skip the character name prefix in prefill mode
                    if skip_prefix and prefix_buffer is not None:
                        prefix_buffer += text
                        # Check if we've seen the full prefix
                        expected_prefix = f"{prefill_name}: "
                        if len(prefix_buffer) >= len(expected_prefix):
                            # We've seen enough, check if it matches
                            if prefix_buffer.startswith(expected_prefix):
                                # Skip the prefix, yield the rest
                                text_to_yield = prefix_buffer[len(expected_prefix):]
                                if text_to_yield:
                                    if self.echo_filter:
                                        self.echo_filter.on_tts_chunk(session_id, text_to_yield)
                                    yield text_to_yield
                                    # Broadcast to UI
                                    if hasattr(self, 'ui_server'):
                                        await self.ui_server.broadcast_ai_stream(
                                            speaker=character_config.name,
                                            text=text_to_yield,
                                            session_id=ui_session_id
                                        )
                            else:
                                # Doesn't match expected prefix, yield everything
                                if self.echo_filter:
                                    self.echo_filter.on_tts_chunk(session_id, prefix_buffer)
                                yield prefix_buffer
                                # Broadcast to UI
                                if hasattr(self, 'ui_server'):
                                    await self.ui_server.broadcast_ai_stream(
                                        speaker=character_config.name,
                                        text=prefix_buffer,
                                        session_id=ui_session_id
                                    )
                            prefix_buffer = None  # Stop checking
                        elif not expected_prefix.startswith(prefix_buffer):
                            # Buffer doesn't match the prefix pattern at all
                            # Output everything immediately and stop checking
                            if self.echo_filter:
                                self.echo_filter.on_tts_chunk(session_id, prefix_buffer)
                            yield prefix_buffer
                            # Broadcast to UI
                            if hasattr(self, 'ui_server'):
                                await self.ui_server.broadcast_ai_stream(
                                    speaker=character_config.name,
                                    text=prefix_buffer,
                                    session_id=ui_session_id
                                )
                            prefix_buffer = None  # Stop checking
                        # else: buffer is a partial match, keep accumulating
                    else:
                        # Track for echo filter
                        if self.echo_filter:
                            self.echo_filter.on_tts_chunk(session_id, text)
                        yield text
                        # Broadcast to UI
                        if hasattr(self, 'ui_server'):
                            await self.ui_server.broadcast_ai_stream(
                                speaker=character_config.name,
                                text=text,
                                session_id=ui_session_id
                            )
                    
        except Exception as e:
            print(f"❌ Anthropic streaming error for {character_config.name}: {e}")
            raise
        finally:
            # Send completion signal to UI with the pre-calculated message_id
            if hasattr(self, 'ui_server'):
                await self.ui_server.broadcast_ai_stream(
                    speaker=character_config.name,
                    text="",
                    session_id=ui_session_id,
                    is_complete=True,
                    message_id=pending_message_id
                )

    async def _stream_character_bedrock_response(self, messages, character_config, request_timestamp, ui_session_id: Optional[str] = None, pending_message_id: Optional[str] = None):
        """Stream response from Bedrock for a specific character."""
        # For Bedrock, we use the async bedrock client
        client = self.async_bedrock_client
        ui_session_id = ui_session_id or f"session_{request_timestamp}"
        
        # Create session ID for echo tracking
        session_id = f"bedrock_{character_config.name}_{request_timestamp}"
        if self.echo_filter:
            self.echo_filter.on_tts_start(session_id, character_config.name)
        
        try:
            # Check if messages are already in prefill format
            is_prefill_format = False
            prefill_name = None
            
            # Check if last message is assistant (typical prefill pattern)
            if len(messages) > 0:
                last_msg = messages[-1]
                if last_msg.get("role") == "assistant":
                    is_prefill_format = True
                    # Extract prefill name from the assistant message content
                    content = last_msg.get("content", "")
                    if content.endswith(":"):
                        # Extract the name before the final colon
                        parts = content.rsplit("\n", 1)
                        if len(parts) > 0:
                            last_line = parts[-1].strip()
                            if last_line.endswith(":"):
                                prefill_name = last_line[:-1]
            
            messages_to_send = messages
            
            # Extract system content and prepare messages
            system_content = ""
            anthropic_messages = []
            
            for msg in messages_to_send:
                if msg["role"] == "system":
                    system_content = msg["content"]
                else:
                    # Create a clean copy without metadata
                    clean_msg = {"role": msg["role"], "content": msg["content"]}
                    anthropic_messages.append(clean_msg)
            
            # Generate stop sequences for multi-character conversations
            stop_sequences = []
            if is_prefill_format:
                # Add all character names as stop sequences
                for char_name in self.character_manager.characters.keys():
                    stop_sequences.append(f"\n\n{char_name}:")
                    # Also add prefill names if different
                    char_config = self.character_manager.get_character_config(char_name)
                    if char_config and char_config.prefill_name and char_config.prefill_name != char_name:
                        stop_sequences.append(f"\n\n{char_config.prefill_name}:")
                
                # Add human names
                if hasattr(self.config.conversation, 'prefill_participants'):
                    human_name = self.config.conversation.prefill_participants[0]
                    stop_sequences.append(f"\n\n{human_name}:")
                
                # Add any detected speakers
                if hasattr(self, 'detected_speakers') and self.detected_speakers:
                    for speaker in self.detected_speakers:
                        speaker_stop = f"\n\n{speaker}:"
                        if speaker_stop not in stop_sequences:
                            stop_sequences.append(speaker_stop)
                
                # Add default system stop sequence
                stop_sequences.append("\n\nSystem:")
                stop_sequences.append("\n\nA:")
                
                # Remove duplicates while preserving order
                stop_sequences = list(dict.fromkeys(stop_sequences))
                print(f"🛑 Character using stop sequences: {stop_sequences}")
            
            # For Bedrock, model names start with "anthropic." or "us.anthropic." (cross-region)
            model_name = character_config.llm_model
            # Only prepend anthropic. if the model doesn't already have the right prefix
            if not model_name.startswith("anthropic.") and not model_name.startswith("us.anthropic."):
                model_name = f"anthropic.{model_name}"
            
            # Prepare parameters for Bedrock - remove None values as Bedrock doesn't like them
            bedrock_params = {
                "model": model_name,
                "messages": anthropic_messages,
                "stream": True,
                "max_tokens": character_config.max_tokens,
                "temperature": character_config.temperature,
                "stop_sequences": stop_sequences or []  # Use empty list instead of None
            }
            
            # Add system prompt if provided
            if system_content:
                bedrock_params["system"] = system_content
            
            # Remove any None values that might cause issues
            bedrock_params = {k: v for k, v in bedrock_params.items() if v is not None}
            
            response = await client.messages.create(**bedrock_params)
            
            # Track if we need to skip the prefill prefix
            skip_prefix = is_prefill_format
            prefix_buffer = "" if skip_prefix else None
            async for chunk in response:
                if chunk.type == "content_block_delta" and chunk.delta.text:
                    text = chunk.delta.text
                    
                    # For prefill mode, skip the initial character name
                    if prefix_buffer is not None:
                        prefix_buffer += text
                        # Check eagerly after each chunk - look for the prefix pattern
                        if prefill_name:
                            prefix_pattern = f"{prefill_name}: "
                            prefix_len = len(prefix_pattern)
                            
                            # If we have enough chars to check for the full prefix
                            if len(prefix_buffer) >= prefix_len:
                                if prefix_buffer.startswith(prefix_pattern):
                                    # Skip the prefix and output the rest
                                    remaining = prefix_buffer[prefix_len:]
                                    if remaining:
                                        # Track for echo filter
                                        if self.echo_filter:
                                            self.echo_filter.on_tts_chunk(session_id, remaining)
                                        yield remaining
                                        # Broadcast to UI
                                        if hasattr(self, 'ui_server'):
                                            await self.ui_server.broadcast_ai_stream(
                                                speaker=character_config.name,
                                                text=remaining,
                                                session_id=ui_session_id
                                            )
                                    prefix_buffer = None  # Done checking for prefix
                                elif not prefix_pattern.startswith(prefix_buffer):
                                    # Buffer doesn't match the prefix pattern at all
                                    # Output everything and stop checking
                                    if self.echo_filter:
                                        self.echo_filter.on_tts_chunk(session_id, prefix_buffer)
                                    yield prefix_buffer
                                    # Broadcast to UI
                                    if hasattr(self, 'ui_server'):
                                        await self.ui_server.broadcast_ai_stream(
                                            speaker=character_config.name,
                                            text=prefix_buffer,
                                            session_id=ui_session_id
                                        )
                                    prefix_buffer = None  # Done checking for prefix
                                # else: buffer is a partial match, keep accumulating
                            # else: not enough chars yet, keep accumulating
                    else:
                        # Track for echo filter
                        if self.echo_filter:
                            self.echo_filter.on_tts_chunk(session_id, text)
                        yield text
                        # Broadcast to UI
                        if hasattr(self, 'ui_server'):
                            await self.ui_server.broadcast_ai_stream(
                                speaker=character_config.name,
                                text=text,
                                session_id=ui_session_id
                            )
                    
        except Exception as e:
            print(f"❌ Bedrock streaming error for {character_config.name}: {e}")
            raise
        finally:
            # Send completion signal to UI with the pre-calculated message_id
            if hasattr(self, 'ui_server'):
                await self.ui_server.broadcast_ai_stream(
                    speaker=character_config.name,
                    text="",
                    session_id=ui_session_id,
                    is_complete=True,
                    message_id=pending_message_id
                )

    def _get_conversation_history_for_director(self):
        """Get conversation history formatted for director.
        Excludes images since director only needs text to decide who speaks next.
        """
        history = []
        for turn in self.state.conversation_history[-20:]:  # Last 20 turns
            # Include both completed and interrupted turns
            if turn.status in ["completed", "interrupted"]:
                # Extract text content only
                if isinstance(turn.content, str):
                    content = turn.content
                elif isinstance(turn.content, list):
                    # Extract text from content blocks
                    text_parts = []
                    for item in turn.content:
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                    content = " ".join(text_parts)
                else:
                    content = str(turn.content)
                
                entry = {
                    "role": turn.role,
                    "content": content
                }
                if turn.character:
                    entry["character"] = turn.character
                if turn.speaker_name:
                    entry["speaker_name"] = turn.speaker_name
                history.append(entry)
        return history
    
    def _set_character_voice(self, character_config):
        """Set TTS voice for a specific character."""
        if not character_config:
            return None
            
        voice_settings = self.character_manager.get_character_voice_settings(
            character_config.name
        )
        
        # Save original config
        original_config = {
            "voice_id": self.tts.config.voice_id,
            "speed": self.tts.config.speed,
            "stability": self.tts.config.stability,
            "similarity_boost": self.tts.config.similarity_boost
        }
        
        # Apply character voice settings
        if voice_settings.get("voice_id"):
            self.tts.config.voice_id = voice_settings["voice_id"]
            self.tts.config.speed = voice_settings.get("speed", 1.0)
            self.tts.config.stability = voice_settings.get("stability", 0.5)
            self.tts.config.similarity_boost = voice_settings.get("similarity_boost", 0.8)
            
            print(f"🎤 Set voice for {character_config.name}: {voice_settings['voice_id']}")
        
        return original_config
    
    def _restore_voice_config(self, original_config):
        """Restore original TTS voice configuration."""
        if original_config:
            self.tts.config.voice_id = original_config["voice_id"]
            self.tts.config.speed = original_config["speed"]
            self.tts.config.stability = original_config["stability"]
            self.tts.config.similarity_boost = original_config["similarity_boost"]
    
    def _create_character_prefill_messages(self, raw_history: List[Dict[str, Any]], character_name: str, 
                                          prefill_name: str, system_prompt: str, 
                                          context_char_histories: Optional[Dict[str, List[Any]]] = None) -> List[Dict[str, Any]]:
        """Create prefill format messages directly from raw conversation history for a character.
        
        This creates the proper prefill structure with images:
        1. System message
        2. User messages with pre-image history 
        3. User messages with images
        4. Assistant message with post-image history and character's prefill
        """
        print(f"🔍 DEBUG: _create_character_prefill_messages called with:")
        print(f"  - character_name: '{character_name}'")
        print(f"  - prefill_name: '{prefill_name}'")
        print(f"  - context_char_histories keys: {list(context_char_histories.keys()) if context_char_histories else None}")
        
        # Find last image in history
        last_image_index = -1
        for i, turn in enumerate(raw_history):
            if turn.get("role") == "user" and isinstance(turn.get("content"), list):
                for item in turn["content"]:
                    if isinstance(item, dict) and item.get("type") == "image":
                        last_image_index = i
                        break
        
        # Build prefill messages
        
        # Add character-specific history if available
        char_history_parts = []
        if context_char_histories and character_name in context_char_histories:
            char_history = context_char_histories[character_name]
            print(f"📚 Prepending {len(char_history)} messages from {character_name}'s context-specific history")
            
            # Format character history turns for prefill
            for turn in char_history:
                # In prefill format, we need to format these as conversation turns
                formatted_text = self._format_turn_for_prefill(
                    turn.role,
                    turn.content,
                    character_name if turn.role == "assistant" else None,
                    character_name,
                    prefill_name,
                    speaker_name=turn.speaker_name if hasattr(turn, 'speaker_name') else None
                )
                if formatted_text:
                    char_history_parts.append(formatted_text)
        else:
            print(f"🔍 DEBUG: No character history found for '{character_name}'")
            if context_char_histories:
                print(f"🔍 DEBUG: Available keys are: {list(context_char_histories.keys())}")
        
        if last_image_index >= 0:
            # Has images - use image-aware format
            user_messages = []
            text_blocks = []
            
            # Start with character history if available
            if char_history_parts:
                text_blocks.extend(char_history_parts)
            
            # Process messages up to and including last image
            for i, turn in enumerate(raw_history[:last_image_index + 1]):
                role = turn.get("role")
                content = turn.get("content")
                character = turn.get("character")
                
                # Check if this turn has an image
                has_image = False
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "image":
                            has_image = True
                            break
                
                if has_image:
                    # First, add any accumulated text as a user message
                    if text_blocks:
                        user_messages.append({"role": "user", "content": "\n\n".join(text_blocks)})
                        text_blocks = []
                    
                    # Then add the image message
                    user_messages.append({"role": "user", "content": content})
                else:
                    # Regular text message - format and accumulate
                    formatted_text = self._format_turn_for_prefill(role, content, character, character_name, prefill_name)
                    if formatted_text:
                        text_blocks.append(formatted_text)
            
            # Add any remaining pre-image text
            if text_blocks:
                user_messages.append({"role": "user", "content": "\n\n".join(text_blocks)})
            
            # Build assistant prefix from post-image history
            assistant_parts = []
            for turn in raw_history[last_image_index + 1:]:
                formatted_text = self._format_turn_for_prefill(
                    turn.get("role"), 
                    turn.get("content"), 
                    turn.get("character"),
                    character_name,
                    prefill_name
                )
                if formatted_text:
                    assistant_parts.append(formatted_text)
            
            # Create assistant message with prefill
            assistant_content = "\n\n".join(assistant_parts) + f"\n\n{prefill_name}:" if assistant_parts else f"{prefill_name}:"
            
            return [{"role": "system", "content": system_prompt}] + user_messages + [{"role": "assistant", "content": assistant_content}]
        else:
            # No images - simpler format
            conversation_parts = []
            
            # Start with character history if available
            if char_history_parts:
                conversation_parts.extend(char_history_parts)
            
            for turn in raw_history:
                if turn.get("content").strip() == "" or turn.get("content").strip() == INTERRUPTED_STR:
                    print(f"Ignoring turn {repr(turn)}")
                    continue
                formatted_text = self._format_turn_for_prefill(
                    turn.get("role"),
                    turn.get("content"),
                    turn.get("character"),
                    character_name,
                    prefill_name,
                    speaker_name=turn.get("speaker_name")
                )
                if formatted_text:
                    conversation_parts.append(formatted_text)
            
            # Create prefill format
            assistant_content = "\n\n".join(conversation_parts) + f"\n\n{prefill_name}:" if conversation_parts else f"{prefill_name}:"
            
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self.config.conversation.prefill_user_message},
                {"role": "assistant", "content": assistant_content}
            ]
    
    def _format_turn_for_prefill(self, role: str, content: Any, character: Optional[str], 
                                 current_character: str, prefill_name: str, speaker_name: Optional[str] = None) -> Optional[str]:
        """Format a single turn for prefill conversation."""
        # Extract text from content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            content = " ".join(text_parts)
        
        if not content or not isinstance(content, str):
            return None
        
        if role == "user":
            # Human messages - use speaker name if available, otherwise fallback to "H"
            participant = speaker_name if speaker_name else "H"
            return f"{participant}: {content}"
        elif role == "assistant":
            # Character messages
            if character == current_character:
                # This character's own messages
                return f"{prefill_name}: {content}"
            elif character:
                # Other character's messages - get their prefill name
                other_config = self.character_manager.get_character_config(character)
                if other_config and other_config.prefill_name:
                    return f"{other_config.prefill_name}: {content}"
                else:
                    return f"{character}: {content}"
            else:
                # Generic assistant
                return f"Assistant: {content}"
        elif role == "system":
            # System messages (like interruptions)
            return f"System: {content}"
        
        return None
    
    def _convert_character_messages_to_prefill(self, messages: List[Dict[str, str]], character_prefill_name: str) -> List[Dict[str, str]]:
        """Convert character messages to prefill format with character-specific name, preserving images.
        
        This handles messages that have already been formatted by format_messages_for_character,
        which means other characters' messages are already prefixed with their names.
        """
        # Extract system prompt
        system_prompt = ""
        chat_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                chat_messages.append(msg)
        
        # Use the same image-aware conversion as regular prefill
        # First, find the last image index
        last_image_index = -1
        for i, msg in enumerate(chat_messages):
            if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                # Check if this message contains an image
                for content_item in msg["content"]:
                    if content_item.get("type") == "image":
                        last_image_index = i
                        break
        
        # If there are images, use the image-aware prefill format
        if last_image_index >= 0:
            # Build user messages and assistant prefix using image-aware logic
            user_messages = []
            current_text_block = []
            
            # Process messages up to and including the last image
            for i, msg in enumerate(chat_messages):
                # Check if this is an image message
                is_image_message = False
                if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                    for content_item in msg["content"]:
                        if content_item.get("type") == "image":
                            is_image_message = True
                            break
                
                if is_image_message and i <= last_image_index:
                    # First, add any accumulated text block
                    if current_text_block:
                        text_content = "\n\n".join(current_text_block)
                        user_messages.append({"role": "user", "content": text_content})
                        current_text_block = []
                    
                    # Then add the image message as-is
                    user_messages.append(msg)
                elif i <= last_image_index:
                    # Regular text message before the last image - format and accumulate
                    content = msg.get('content', '')
                    if msg["role"] == "user":
                        # User messages might already have character names from format_messages_for_character
                        # Only add "H:" if there's no existing prefix
                        if isinstance(content, str) and not re.match(r'^[^:]+:', content):
                            content = f"H: {content}"
                        current_text_block.append(content)
                    else:  # assistant
                        # Assistant messages are already properly formatted
                        # Just use the content as-is
                        current_text_block.append(content)
            
            # Add any remaining text block before the last image
            if current_text_block and last_image_index >= 0:
                text_content = "\n\n".join(current_text_block)
                user_messages.append({"role": "user", "content": text_content})
                current_text_block = []
            
            # Build assistant prefix from everything after the last image
            assistant_parts = []
            for i, msg in enumerate(chat_messages):
                if i > last_image_index:
                    content = msg.get('content', '')
                    if isinstance(content, list):
                        # Extract text from list content
                        text_parts = []
                        for item in content:
                            if item.get("type") == "text":
                                text_parts.append(item.get("text", ""))
                        content = " ".join(text_parts)
                    
                    if msg["role"] == "user":
                        # User messages might already have character names from format_messages_for_character
                        # Only add "H:" if there's no existing prefix
                        if isinstance(content, str) and not re.match(r'^[^:]+:', content):
                            content = f"H: {content}"
                        assistant_parts.append(content)
                    else:  # assistant
                        # Assistant messages are already properly formatted
                        # Just use the content as-is
                        assistant_parts.append(content)
            
            # Create assistant prefix
            if assistant_parts:
                assistant_prefix = "\n\n".join(assistant_parts) + f"\n\n{character_prefill_name}:"
            else:
                assistant_prefix = f"{character_prefill_name}:"
            
            # If no user messages were created, use the default
            if not user_messages:
                user_messages = [{"role": "user", "content": self.config.conversation.prefill_user_message}]
            
            # Build final messages
            prefill_messages = [{"role": "system", "content": system_prompt}] + user_messages + [{"role": "assistant", "content": assistant_prefix}]
        else:
            # No images - use the original text-only approach
            conversation_parts = []
            human_name = "H"
            
            for msg in chat_messages:
                content = msg["content"]
                
                if msg["role"] == "user":
                    if isinstance(content, str) and not re.match(r'^[^:]+:', content):
                        content = f"{human_name}: {content}"
                    conversation_parts.append(content)
                else:  # assistant
                    # Assistant messages are already properly formatted
                    # Just use the content as-is
                    conversation_parts.append(content)
            
            # Create prefill format
            if conversation_parts:
                assistant_prefix = "\n\n".join(conversation_parts) + f"\n\n{character_prefill_name}:"
            else:
                assistant_prefix = f"{character_prefill_name}:"
            
            prefill_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self.config.conversation.prefill_user_message},
                {"role": "assistant", "content": assistant_prefix}
            ]
        
        # Add metadata to indicate this is prefill format
        prefill_messages[-1]["_is_prefill"] = True
        prefill_messages[-1]["_prefill_name"] = character_prefill_name
        
        return prefill_messages
    
    def _get_conversation_history_for_character(self):
        """Get conversation history formatted for character LLM.
        Preserves full content including images.
        """
        history = []
        for turn in self.state.conversation_history:
            # Include completed, interrupted, processing, and pending turns
            # (All statuses are valid for context - we want the full conversation)
            if turn.status in ["completed", "interrupted", "processing", "pending"]:
                entry = {
                    "role": turn.role,
                    "content": turn.content  # Keep as-is (string or list)
                }
                if turn.character:
                    entry["character"] = turn.character
                if turn.speaker_name:
                    entry["speaker_name"] = turn.speaker_name
                history.append(entry)
        
        return history
    
    
    async def _conversation_manager(self):
        """Simplified conversation manager - processing is now handled by _process_pending_utterances."""
        print("🔄 Conversation manager started (simplified)")
        
        while self.state.is_active:
            try:
                await asyncio.sleep(0.1)  # Less frequent checking since processing is event-driven
                
                # Just show debug info occasionally
                if self.config.development.enable_debug_mode:
                    pending_count = sum(1 for turn in self.state.conversation_history 
                                      if turn.role == "user" and turn.status == "pending")
                    if pending_count > 0:
                        print(f"🐛 Debug: {pending_count} pending utterances, "
                              f"Processing: {self.state.is_processing_llm}, "
                              f"Speaking: {self.state.is_speaking}")
                
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
    
    async def _process_with_llm(self, user_input: str, reference_turn: ConversationTurn, generation: int = None):
        """Process user input with LLM and speak response."""
        if self.state.is_processing_llm:
            print(f"⚠️ Already processing LLM, skipping: {user_input}")
            return
        
        # Check generation if provided
        if generation is not None and generation != self.state.current_generation:
            print(f"🚫 LLM processing cancelled - stale generation {generation} (current: {self.state.current_generation})")
            return
            
        # Mark all processing user turns as completed NOW since we're committing to process them
        for turn in self.state.conversation_history:
            if turn.role == "user" and turn.status == "processing":
                turn.status = "completed"
        self._save_conversation_state()
        
        # Always use character processing
        await self._process_with_character_llm(user_input, reference_turn, generation)
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
    
    def _send_osc_speaking_start(self, character_name: str):
        """Send OSC message when character starts speaking."""
        if not self.config.osc:
            print("🔇 OSC: Config not available")
            return
            
        if not self.config.osc.enabled:
            print("🔇 OSC: Disabled in config")
            return
            
        if not self.osc_client:
            print("❌ OSC: Client not initialized")
            return
            
        try:
            address = self.config.osc.speaking_start_address
            print(f"📡 OSC: Sending to {self.config.osc.host}:{self.config.osc.port}")
            print(f"📡 OSC: Address: {address}, Data: '{character_name}'")
            
            self.osc_client.send_message(address, character_name)
            
            print(f"✅ OSC: {character_name} started speaking - message sent successfully")
        except Exception as e:
            print(f"❌ OSC send error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    def _send_osc_speaking_stop(self, character_name: str):
        """Send OSC message when character stops speaking."""
        if not self.config.osc:
            print("🔇 OSC: Config not available")
            return
            
        if not self.config.osc.enabled:
            print("🔇 OSC: Disabled in config")
            return
            
        if not self.osc_client:
            print("❌ OSC: Client not initialized")
            return
            
        try:
            address = self.config.osc.speaking_stop_address
            print(f"📡 OSC: Sending to {self.config.osc.host}:{self.config.osc.port}")
            print(f"📡 OSC: Address: {address}, Data: '{character_name}'")
            
            self.osc_client.send_message(address, character_name)
            
            print(f"✅ OSC: {character_name} stopped speaking - message sent successfully")
        except Exception as e:
            print(f"❌ OSC send error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    def _send_osc_color_change(self, character_name: str, color: str):
        """Send OSC message when character changes color."""
        if not self.config.osc:
            print("🔇 OSC: Config not available")
            return
            
        if not self.config.osc.enabled:
            print("🔇 OSC: Disabled in config")
            return
            
        if not self.osc_client:
            print("❌ OSC: Client not initialized")
            return
            
        try:
            address = self.config.osc.color_change_address
            print(f"📡 OSC: Sending to {self.config.osc.host}:{self.config.osc.port}")
            print(f"📡 OSC: Address: {address}, Data: ['{character_name}', '{color}']")
            
            self.osc_client.send_message(address, [character_name, color])
            
            print(f"✅ OSC: {character_name} color change to {color} - message sent successfully")
        except Exception as e:
            print(f"❌ OSC send error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    def _initialize_llm_clients(self, config):
        """Initialize LLM clients based on conversation default and actual character providers."""
        providers_needed = {config.conversation.llm_provider}
        
        # Check what providers are needed by the actual loaded characters
        if hasattr(self, 'character_manager') and self.character_manager:
            for char_name, char_config in self.character_manager.characters.items():
                if hasattr(char_config, 'llm_provider'):
                    providers_needed.add(char_config.llm_provider)
        
        print(f"🔧 LLM providers needed: {providers_needed}")
        
        # Initialize clients for all needed providers
        if "openai" in providers_needed:
            self.openai_client = OpenAI(api_key=config.conversation.openai_api_key)
            print("✅ OpenAI client initialized")
            
        if "anthropic" in providers_needed:
            if not config.conversation.anthropic_api_key:
                raise ValueError("Anthropic API key is required when using Anthropic provider")
            # Use async client for better responsiveness
            self.async_anthropic_client = AsyncAnthropic(api_key=config.conversation.anthropic_api_key)
            print("✅ Anthropic client initialized")
            
        if "bedrock" in providers_needed:
            if not config.conversation.aws_access_key_id or not config.conversation.aws_secret_access_key:
                raise ValueError("AWS credentials are required when using Bedrock provider")
            # Import AnthropicBedrock
            from anthropic import AnthropicBedrock, AsyncAnthropicBedrock
            # Initialize Bedrock clients
            self.bedrock_client = AnthropicBedrock(
                aws_region=config.conversation.aws_region,
                aws_access_key=config.conversation.aws_access_key_id,
                aws_secret_key=config.conversation.aws_secret_access_key
            )
            self.async_bedrock_client = AsyncAnthropicBedrock(
                aws_region=config.conversation.aws_region,
                aws_access_key=config.conversation.aws_access_key_id,
                aws_secret_key=config.conversation.aws_secret_access_key
            )
            print(f"✅ Bedrock client initialized for AWS region: {config.conversation.aws_region}")

    def _send_osc_blank_output(self):
        """Send OSC message to blank the output (empty string for laser rendering)."""
        if not self.config.osc:
            print("🔇 OSC: Config not available")
            return
            
        if not self.config.osc.enabled:
            print("🔇 OSC: Disabled in config")
            return
            
        if not self.osc_client:
            print("❌ OSC: Client not initialized")
            return
            
        try:
            address = self.config.osc.blank_output_address
            print(f"📡 OSC: Sending to {self.config.osc.host}:{self.config.osc.port}")
            print(f"📡 OSC: Address: {address}, Data: ''")
            
            self.osc_client.send_message(address, "")
            
            print(f"✅ OSC: Blank output message sent successfully")
        except Exception as e:
            print(f"❌ OSC send error: {type(e).__name__}: {e}")
            return
    
    async def switch_context(self, context_name: str) -> bool:
        """Switch to a different conversation context."""
        print(f"🔄 UnifiedVoiceConversation.switch_context called with: {context_name}")
        
        if not self.context_manager:
            print("❌ Context manager not available")
            return False
        
        print(f"📋 Current active context: {self.context_manager.active_context_name}")
        
        # Switch context
        if self.context_manager.switch_context(context_name):
            print(f"✅ Context manager switched to: {context_name}")
            
            # Sync history from new context
            self._sync_history_from_context()
            
            # Log the context switch
            switch_turn = ConversationTurn(
                role="system",
                content=f"[Switched to context: {context_name}]",
                timestamp=datetime.now(),
                status="completed"
            )
            self._log_conversation_turn("system", switch_turn.content)
            
            return True
        
        print(f"❌ Context manager failed to switch to: {context_name}")
        return False
    
    async def reset_context(self):
        """Reset current context to original history."""
        if not self.context_manager:
            print("❌ Context manager not available")
            return
        
        # Reset the context
        self.context_manager.reset_active_context()
        
        # Sync history from reset context
        self._sync_history_from_context()
        
        # Log the reset
        reset_turn = ConversationTurn(
            role="system",
            content="[Context reset to original history]",
            timestamp=datetime.now(),
            status="completed"
        )
        self._log_conversation_turn("system", reset_turn.content)
    
    def get_context_list(self) -> List[Dict[str, Any]]:
        """Get list of available contexts."""
        if not self.context_manager:
            return []
        return self.context_manager.get_context_list()
    
    async def cleanup(self):
        """Clean up all resources."""
        # First stop the conversation to prevent new operations
        try:
            await self.stop_conversation()
        except Exception as e:
            print(f"Error stopping conversation: {e}")
        
        # Stop thinking sound early to prevent conflicts
        try:
            await self.thinking_sound.interrupt()  # Force stop any playing sound
            await asyncio.sleep(0.1)  # Give it time to stop
        except Exception as e:
            print(f"Error stopping thinking sound: {e}")
        
        # Clean up STT
        try:
            await self.stt.cleanup()
        except Exception as e:
            print(f"Error cleaning up STT: {e}")
            
        # Clean up camera
        try:
            if self.camera:
                self.camera.stop()
                print("📷 Camera stopped")
        except Exception as e:
            print(f"Error cleaning up camera: {e}")
        
        try:
            await self._interrupt_llm_output()
        except Exception as e:
            print(f"Error interrupting llm output: {e}")
        
        # Clean up TTS
        try:
            await self.tts.cleanup()
        except Exception as e:
            print(f"Error cleaning up TTS: {e}")
            
        try:
            await self._interrupt_llm_output()
        except Exception as e:
            print(f"Error interrupting llm output: {e}")
        
        # Clean up UI server
        try:
            await self.ui_server.stop()
        except Exception as e:
            print(f"Error cleaning up UI server: {e}")
            
        # Finally clean up thinking sound resources
        try:
            await self.thinking_sound.cleanup()
        except Exception as e:
            print(f"Error cleaning up thinking sound: {e}")
        
        try:
            await self._interrupt_llm_output()
        except Exception as e:
            print(f"Error interrupting llm output: {e}")
        
        # Clean up audio connection
        try:
            stop_stream()
        except Exception as e:
            print(f"Error cleaning up audio stream: {e}")
        

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
