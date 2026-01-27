#!/usr/bin/env python3
"""
Configuration Loader for Voice AI System
Loads YAML configuration and creates appropriate dataclass configurations.
"""

import yaml
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path
from copy import deepcopy

# Import our configuration dataclasses
from async_stt_module import STTConfig
from async_tts_module import TTSConfig

@dataclass
class ConversationConfig:
    """Configuration for the unified conversation system."""
    # API Keys
    deepgram_api_key: str
    elevenlabs_api_key: str
    openai_api_key: str
    anthropic_api_key: str = ""
    # AWS credentials for Bedrock
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-west-2"
    
    # Voice settings
    voice_id: str = "T2KZm9rWPG5TgXTyjt7E"
    
    # Conversation timing
    pause_threshold: float = 2.0
    min_words_for_submission: int = 3
    max_wait_time: float = 10.0
    
    # Interruption settings
    interruptions_enabled: bool = False  # Default to false for safety
    interruption_confidence: float = 0.8
    
    # STT settings
    stt_provider: str = "deepgram"  # "deepgram" or "elevenlabs"
    stt_model: str = "nova-3"
    stt_language: str = "en-US"
    interim_results: bool = True
    
    # TTS settings
    tts_model: str = "eleven_multilingual_v1"
    tts_speed: float = 1.0
    tts_stability: float = 0.5
    tts_similarity_boost: float = 0.8
    
    # LLM settings
    llm_provider: str = "openai"  # openai, anthropic, or bedrock
    llm_model: str = "chatgpt-4o-latest"
    conversation_mode: str = "chat"  # chat or prefill
    max_tokens: int = 300
    system_prompt: str = "You are a helpful AI assistant in a voice conversation. Give natural, conversational responses that work well when spoken aloud. Keep responses concise but engaging."
    
    # Prefill mode settings
    prefill_user_message: str = '<cmd>cat untitled.txt</cmd>'
    prefill_participants: List[str] = None
    prefill_system_prompt: str = 'The assistant is in CLI simulation mode, and responds to the user\'s CLI commands only with outputs of the commands.'
    
    # History file settings
    history_file: Optional[str] = None

    # Known speakers whitelist for parsing external contexts
    # Add speaker names here that should always be recognized even if they appear only once
    known_speakers: Optional[List[str]] = None

    # Tools configuration
    tools_config: Optional[Dict[str, Any]] = None
    
    # Character configuration
    characters_config: Optional[Dict[str, Any]] = None
    director_config: Optional[Dict[str, Any]] = None
    director_enabled: bool = False  # Default to disabled director (legacy)
    director_mode: str = "off"  # "off", "same_model", or "director"
    default_character: Optional[str] = None  # Default character for same_model mode (uses first non-user if not set)

    # STT control settings
    stt_start_enabled: bool = False  # Whether STT starts automatically (default: off)
    mute_while_speaking: bool = True  # Mute input while AI is speaking (default: on)

    # Echo cancellation settings
    enable_echo_cancellation: bool = False
    aec_frame_size: int = 256  # Must be power of 2
    aec_filter_length: int = 2048
    aec_delay_ms: int = 200  # Reference delay in milliseconds (increased for bursty TTS)

    # Audio archive settings
    audio_archive_enabled: bool = True  # Save all audio (user and AI) by default
    audio_archive_dir: str = "./audio_archive"  # Directory for audio files

    def __post_init__(self):
        if self.prefill_participants is None:
            self.prefill_participants = ['H', 'Claude']

@dataclass
class SpeakerProfile:
    """Configuration for a known speaker."""
    name: str
    description: str = ""
    reference_audio: Optional[str] = None  # Path to reference audio file (30+ seconds)

@dataclass
class SpeakerRecognitionConfig:
    """Configuration for speaker recognition settings."""
    confidence_threshold: float = 0.7
    learning_mode: bool = True
    max_speakers: int = 4
    voice_fingerprint_length: int = 128

@dataclass
class SpeakersConfig:
    """Configuration for speaker identification and voice fingerprinting."""
    profiles: Dict[str, SpeakerProfile] = field(default_factory=dict)
    recognition: SpeakerRecognitionConfig = field(default_factory=SpeakerRecognitionConfig)

@dataclass
class AudioConfig:
    """Audio device configuration."""
    input_device_name: Optional[str] = None
    output_device_name: Optional[str] = None
    stream_sample_rate: Optional[int] = None
    stream_channels: Optional[int] = None
    stream_buffer_size: Optional[int] = None
    stream_enable_aec: Optional[bool] = None
    aec_filter_length: Optional[int] = None

@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    show_interim_results: bool = True
    show_tts_chunks: bool = False
    show_audio_debug: bool = False

@dataclass
class CameraConfig:
    """Camera capture configuration."""
    enabled: bool = False
    device_id: int = 0
    resolution: List[int] = field(default_factory=lambda: [640, 480])
    capture_on_speech: bool = True
    save_captures: bool = False
    capture_dir: str = "camera_captures"
    jpeg_quality: int = 85

@dataclass
class EchoFilterConfig:
    """Echo filter configuration for preventing TTS feedback."""
    enabled: bool = True
    similarity_threshold: float = 0.75
    time_window: float = 15.0
    min_length: int = 3

@dataclass
class DevelopmentConfig:
    """Development and testing configuration."""
    enable_debug_mode: bool = False
    test_mode: bool = False
    mock_apis: bool = False

@dataclass
class OSCConfig:
    """OSC (Open Sound Control) configuration."""
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 7000
    speaking_start_address: str = "/character/speaking/start"
    speaking_stop_address: str = "/character/speaking/stop"
    color_change_address: str = "/character/color/change"
    blank_output_address: str = "/character/blank"


@dataclass
class ContextConfig:
    """Configuration for a conversation context."""
    name: str
    history_file: str
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    character_histories: Optional[Dict[str, str]] = None
    system_active_message: Optional[str] = None  # Message to inject after reset point when context becomes active
    context_type: str = "file"  # "file" or "discord"
    channel_id: Optional[str] = None  # For Discord contexts


@dataclass
class ContextsConfig:
    """Configuration for conversation contexts."""
    enabled: bool = True
    contexts: List[ContextConfig] = field(default_factory=list)
    state_dir: str = "./context_states"
    auto_save_enabled: bool = True
    auto_save_interval: int = 30  # seconds


class ContextMode:
    """Mode for context management - determines where inference happens."""
    LOCAL = "local"  # Local LLM inference, Melodeus manages context
    DISCORD = "discord"  # ChapterX handles inference, Discord is context source


@dataclass
class RelayBotConfig:
    """Configuration for a bot in the relay system."""
    name: str
    user_id: str  # Discord user ID for @mention
    voice_id: Optional[str] = None  # ElevenLabs voice ID for this bot
    enabled: bool = True
    aliases: List[str] = field(default_factory=list)


@dataclass
class RelaySpeakerConfig:
    """Mapping from speaker name to Discord user info for webhook display."""
    speaker_name: str  # Name from STT (e.g., "antra", "User 1")
    discord_username: str  # Username to display in Discord
    discord_user_id: Optional[str] = None  # Discord user ID (for avatar lookup)
    avatar_url: Optional[str] = None  # Direct avatar URL
    aliases: List[str] = field(default_factory=list)  # Alternative speaker names


@dataclass
class RelayWebhookConfig:
    """Configuration for Discord webhook posting."""
    bot_token: str  # Discord bot token for creating webhooks dynamically
    username: str = "Melodeus"  # Display name in Discord
    avatar_url: Optional[str] = None
    include_speaker_name: bool = True  # Prefix messages with speaker name
    mention_mode: str = "default"  # "default", "explicit", "round_robin", "last_speaker"


@dataclass
class RelayConfig:
    """Configuration for Discord relay mode (ChapterX integration)."""
    enabled: bool = False
    url: str = ""  # WebSocket URL for relay server (e.g., "ws://localhost:8800")
    client_id: str = "melodeus-1"  # Unique identifier for this client
    token: str = ""  # Authentication token for relay
    channels: List[str] = field(default_factory=list)  # Discord channel IDs to subscribe to
    default_bot: Optional[str] = None  # Default bot to mention
    bots: Dict[str, RelayBotConfig] = field(default_factory=dict)  # Bot configurations
    speakers: Dict[str, RelaySpeakerConfig] = field(default_factory=dict)  # Speaker -> Discord user mappings
    webhook: Optional[RelayWebhookConfig] = None  # Webhook config for posting STT
    reconnect_interval: float = 5.0  # Seconds between reconnection attempts
    auto_switch_on_connect: bool = False  # Auto-switch to DISCORD mode when relay connects


@dataclass
class VoiceAIConfig:
    """Complete voice AI system configuration."""
    conversation: ConversationConfig
    stt: STTConfig
    tts: TTSConfig
    audio: AudioConfig
    logging: LoggingConfig
    development: DevelopmentConfig
    speakers: SpeakersConfig = field(default_factory=SpeakersConfig)
    camera: Optional[CameraConfig] = None
    echo_filter: Optional[EchoFilterConfig] = None
    osc: Optional[OSCConfig] = None
    contexts: Optional[ContextsConfig] = None
    relay: Optional[RelayConfig] = None  # Discord relay mode configuration
    _raw_config: Dict[str, Any] = field(default_factory=dict)  # Store raw YAML for custom sections

def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries, with override values taking precedence.
    
    Args:
        base: Base dictionary
        override: Override dictionary with values to merge in
        
    Returns:
        Merged dictionary
    """
    result = deepcopy(base)
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    
    return result

class ConfigLoader:
    """Loads and validates configuration from YAML files."""
    
    DEFAULT_CONFIG_PATHS = [
        "config.yaml",
        "config/config.yaml",
        os.path.expanduser("~/.voiceai/config.yaml"),
        "/etc/voiceai/config.yaml"
    ]
    
    PRESET_DIRS = [
        "presets",
        "config/presets",
        os.path.expanduser("~/.voiceai/presets"),
        "/etc/voiceai/presets"
    ]
    
    @classmethod
    def load(cls, config_path: Optional[str] = None, preset: Optional[str] = None) -> VoiceAIConfig:
        """
        Load configuration from YAML file with optional preset overrides.
        
        Args:
            config_path: Path to config file. If None, searches default locations.
            preset: Name of preset to apply (without .yaml extension). 
                   Can also be set via VOICE_AI_PRESET environment variable.
            
        Returns:
            VoiceAIConfig: Complete configuration object
            
        Raises:
            FileNotFoundError: If no config file is found
            ValueError: If configuration is invalid
        """
        # Determine preset to use
        if preset is None:
            preset = os.environ.get('VOICE_AI_PRESET')
        
        # Find config file
        if config_path:
            config_file = Path(config_path)
            if not config_file.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
        else:
            config_file = cls._find_config_file()
            if not config_file:
                raise FileNotFoundError(
                    f"No config file found in default locations: {cls.DEFAULT_CONFIG_PATHS}"
                )
        
        # Load base YAML
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")

        # Apply preset overrides if specified
        if preset:
            preset_data = cls._load_preset(preset)
            config_data = deep_merge(config_data, preset_data)
        # Validate and create configurations
        return cls._create_config(config_data)
    
    @classmethod
    def _find_config_file(cls) -> Optional[Path]:
        """Find the first available config file from default paths."""
        for path in cls.DEFAULT_CONFIG_PATHS:
            config_file = Path(path)
            if config_file.exists():
                return config_file
        return None
    
    @classmethod
    def _find_preset_file(cls, preset_name: str) -> Path:
        """Find preset file in default preset directories."""
        preset_filename = f"{preset_name}.yaml"
        
        for preset_dir in cls.PRESET_DIRS:
            preset_path = Path(preset_dir) / preset_filename
            if preset_path.exists():
                return preset_path
        
        raise FileNotFoundError(
            f"Preset '{preset_name}' not found in any preset directory: {cls.PRESET_DIRS}"
        )
    
    @classmethod
    def _load_preset(cls, preset_name: str) -> Dict[str, Any]:
        """Load preset configuration from YAML file."""
        preset_file = cls._find_preset_file(preset_name)
        
        try:
            with open(preset_file, 'r') as f:
                preset_data = yaml.safe_load(f) or {}
            return preset_data
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in preset file '{preset_file}': {e}")
    
    @classmethod
    def _create_config(cls, config_data: Dict[str, Any]) -> VoiceAIConfig:
        """Create configuration objects from loaded YAML data."""
        
        # Validate required API keys
        api_keys = config_data.get('api_keys', {})
        required_keys = ['deepgram', 'elevenlabs', 'openai']
        missing_keys = [key for key in required_keys if not api_keys.get(key)]
        
        if missing_keys:
            raise ValueError(f"Missing required API keys: {missing_keys}")
        
        # Check for Anthropic key if Anthropic provider is selected
        conversation_config_data = config_data.get('conversation', {})
        if conversation_config_data.get('llm_provider') == 'anthropic' and not api_keys.get('anthropic'):
            raise ValueError("Anthropic API key is required when using Anthropic as LLM provider")
        
        # Check for AWS credentials if Bedrock provider is selected
        if conversation_config_data.get('llm_provider') == 'bedrock':
            if not api_keys.get('aws_access_key_id') or not api_keys.get('aws_secret_access_key'):
                raise ValueError("AWS credentials (aws_access_key_id and aws_secret_access_key) are required when using Bedrock as LLM provider")
        
        # Extract configuration sections
        voice_config = config_data.get('voice', {})
        stt_config_data = config_data.get('stt', {})
        tts_config_data = config_data.get('tts', {})
        audio_config_data = config_data.get('audio', {})
        logging_config_data = config_data.get('logging', {})
        dev_config_data = config_data.get('development', {})
        tools_config = config_data.get('tools', {})
        characters_config = config_data.get('characters', {})
        camera_config_data = config_data.get('camera', {})
        director_config = config_data.get('director', {})
        echo_filter_data = config_data.get('echo_filter', {})
        
        # Parse keywords if provided
        keywords = None
        if 'keywords' in stt_config_data:
            keywords = []
            for kw in stt_config_data['keywords']:
                if isinstance(kw, dict):
                    keywords.append((kw['word'], kw.get('weight', 5.0)))
                elif isinstance(kw, str):
                    keywords.append((kw, 5.0))  # Default weight
                elif isinstance(kw, (list, tuple)) and len(kw) >= 2:
                    keywords.append((kw[0], float(kw[1])))
        
        # Create STT configuration
        # Use the correct API key based on provider
        stt_provider = stt_config_data.get('provider', 'deepgram')
        stt_api_key = api_keys['elevenlabs'] if stt_provider == 'elevenlabs' else api_keys['deepgram']

        # Parse gate config if provided
        gate_config = stt_config_data.get('gate', None)
        
        # Parse batch diarization config if provided
        batch_diarization = stt_config_data.get('batch_diarization', None)
        
        stt_config = STTConfig(
            api_key=stt_api_key,
            model=stt_config_data.get('model', 'nova-3'),
            language=stt_config_data.get('language', 'en-US'),
            sample_rate=stt_config_data.get('sample_rate', 16000),
            chunk_size=stt_config_data.get('chunk_size', 8000),
            channels=1,  # Always mono for voice
            smart_format=True,  # Always enabled
            interim_results=stt_config_data.get('interim_results', True),
            punctuate=stt_config_data.get('punctuate', True),
            diarize=stt_config_data.get('diarize', True),
            utterance_end_ms=stt_config_data.get('utterance_end_ms', 1000),
            vad_events=stt_config_data.get('vad_events', True),
            # Audio input device - check both stt and audio sections for backward compatibility
            input_device_name=stt_config_data.get('input_device_name') or audio_config_data.get('input_device_name'),
            enable_speaker_id=stt_config_data.get('enable_speaker_id', False),
            speaker_profiles_path=stt_config_data.get('speaker_profiles_path'),
            keywords=keywords,
            debug_speaker_data=stt_config_data.get('debug_speaker_data', False),
            save_user_audio=stt_config_data.get('save_user_audio', False),
            gate_config=gate_config,
            batch_diarization=batch_diarization
        )
        
        # Create TTS configuration
        # Get audio archive directory if enabled
        audio_archive_dir = None
        if conversation_config_data.get('audio_archive_enabled', True):
            audio_archive_dir = conversation_config_data.get('audio_archive_dir', './audio_archive')

        tts_config = TTSConfig(
            api_key=api_keys['elevenlabs'],
            voice_id=voice_config.get('id', 'T2KZm9rWPG5TgXTyjt7E'),
            model_id=tts_config_data.get('model_id', 'eleven_multilingual_v2'),
            output_format=tts_config_data.get('output_format', 'pcm_22050'),
            sample_rate=tts_config_data.get('sample_rate', 22050),
            speed=tts_config_data.get('speed', 1.0),
            stability=tts_config_data.get('stability', 0.5),
            similarity_boost=tts_config_data.get('similarity_boost', 0.8),
            chunk_size=tts_config_data.get('chunk_size', 1024),
            buffer_size=tts_config_data.get('buffer_size', 2048),
            # Multi-voice support
            emotive_voice_id=voice_config.get('emotive_id'),
            emotive_speed=tts_config_data.get('emotive_speed', 1.0),
            emotive_stability=tts_config_data.get('emotive_stability', 0.5),
            emotive_similarity_boost=tts_config_data.get('emotive_similarity_boost', 0.8),
            # Audio output device
            output_device_name=tts_config_data.get('output_device_name'),
            # Audio archiving
            audio_archive_dir=audio_archive_dir
        )
        
        # Create conversation configuration
        conversation_config = ConversationConfig(
            deepgram_api_key=api_keys['deepgram'],
            elevenlabs_api_key=api_keys['elevenlabs'],
            openai_api_key=api_keys['openai'],
            anthropic_api_key=api_keys.get('anthropic', ''),
            # AWS credentials for Bedrock
            aws_access_key_id=api_keys.get('aws_access_key_id', ''),
            aws_secret_access_key=api_keys.get('aws_secret_access_key', ''),
            aws_region=api_keys.get('aws_region', 'us-west-2'),
            voice_id=voice_config.get('id', 'T2KZm9rWPG5TgXTyjt7E'),
            pause_threshold=conversation_config_data.get('pause_threshold', 2.0),
            min_words_for_submission=conversation_config_data.get('min_words_for_submission', 3),
            max_wait_time=conversation_config_data.get('max_wait_time', 10.0),
            interruptions_enabled=conversation_config_data.get('interruptions_enabled', False),
            interruption_confidence=conversation_config_data.get('interruption_confidence', 0.8),
            stt_provider=stt_config_data.get('provider', 'deepgram'),
            stt_model=stt_config_data.get('model', 'nova-3'),
            stt_language=stt_config_data.get('language', 'en-US'),
            interim_results=stt_config_data.get('interim_results', True),
            tts_model=tts_config_data.get('model_id', 'eleven_multilingual_v2'),
            tts_speed=tts_config_data.get('speed', 1.0),
            tts_stability=tts_config_data.get('stability', 0.5),
            tts_similarity_boost=tts_config_data.get('similarity_boost', 0.8),
            llm_provider=conversation_config_data.get('llm_provider', 'openai'),
            llm_model=conversation_config_data.get('llm_model', 'chatgpt-4o-latest'),
            conversation_mode=conversation_config_data.get('conversation_mode', 'chat'),
            max_tokens=conversation_config_data.get('max_tokens', 300),
            system_prompt=conversation_config_data.get('system_prompt', 
                "You are a helpful AI assistant in a voice conversation. Give natural, conversational responses that work well when spoken aloud. Keep responses concise but engaging."),
            prefill_user_message=conversation_config_data.get('prefill_user_message', '<cmd>cat untitled.txt</cmd>'),
            prefill_participants=conversation_config_data.get('prefill_participants', ['H', 'Claude']),
            prefill_system_prompt=conversation_config_data.get('prefill_system_prompt', 
                'The assistant is in CLI simulation mode, and responds to the user\'s CLI commands only with outputs of the commands.'),
            history_file=conversation_config_data.get('history_file'),
            known_speakers=conversation_config_data.get('known_speakers'),
            tools_config=tools_config,
            characters_config=characters_config,
            director_config=director_config,
            director_enabled=conversation_config_data.get('director_enabled', False),
            director_mode=conversation_config_data.get('director_mode', 
                "director" if conversation_config_data.get('director_enabled', False) else "off"),
            enable_echo_cancellation=conversation_config_data.get('enable_echo_cancellation', False),
            aec_frame_size=conversation_config_data.get('aec_frame_size', 256),
            aec_filter_length=conversation_config_data.get('aec_filter_length', 2048),
            aec_delay_ms=conversation_config_data.get('aec_delay_ms', 200),
            default_character=conversation_config_data.get('default_character'),
            stt_start_enabled=conversation_config_data.get('stt_start_enabled', False),
            mute_while_speaking=conversation_config_data.get('mute_while_speaking', True),
            audio_archive_enabled=conversation_config_data.get('audio_archive_enabled', True),
            audio_archive_dir=conversation_config_data.get('audio_archive_dir', './audio_archive'),
        )
        
        # Create other configurations
        audio_config = AudioConfig(
            input_device_name=audio_config_data.get('input_device_name'),
            output_device_name=audio_config_data.get('output_device_name'),
            stream_sample_rate=audio_config_data.get('stream_sample_rate'),
            stream_channels=audio_config_data.get('stream_channels'),
            stream_buffer_size=audio_config_data.get('stream_buffer_size'),
            stream_enable_aec=audio_config_data.get('stream_enable_aec'),
            aec_filter_length=audio_config_data.get('aec_filter_length'),
        )
        
        logging_config = LoggingConfig(
            level=logging_config_data.get('level', 'INFO'),
            show_interim_results=logging_config_data.get('show_interim_results', True),
            show_tts_chunks=logging_config_data.get('show_tts_chunks', False),
            show_audio_debug=logging_config_data.get('show_audio_debug', False)
        )
        
        development_config = DevelopmentConfig(
            enable_debug_mode=dev_config_data.get('enable_debug_mode', False),
            test_mode=dev_config_data.get('test_mode', False),
            mock_apis=dev_config_data.get('mock_apis', False)
        )
        
        # Create camera config if provided
        camera_config = None
        if camera_config_data:
            camera_config = CameraConfig(
                enabled=camera_config_data.get('enabled', False),
                device_id=camera_config_data.get('device_id', 0),
                resolution=camera_config_data.get('resolution', [640, 480]),
                capture_on_speech=camera_config_data.get('capture_on_speech', True),
                save_captures=camera_config_data.get('save_captures', False),
                capture_dir=camera_config_data.get('capture_dir', 'camera_captures'),
                jpeg_quality=camera_config_data.get('jpeg_quality', 85)
            )
        
        # Create echo filter config
        echo_filter_config = None
        if echo_filter_data or echo_filter_data is None:  # Create with defaults if not explicitly disabled
            echo_filter_config = EchoFilterConfig(
                enabled=echo_filter_data.get('enabled', True),
                similarity_threshold=echo_filter_data.get('similarity_threshold', 0.75),
                time_window=echo_filter_data.get('time_window', 15.0),
                min_length=echo_filter_data.get('min_length', 3)
            )
        
        # Create speakers config
        speakers_data = config_data.get('speakers', {})
        
        # Parse speaker profiles
        speaker_profiles = {}
        profiles_data = speakers_data.get('profiles', {})
        for profile_id, profile_data in profiles_data.items():
            speaker_profiles[profile_id] = SpeakerProfile(
                name=profile_data.get('name', profile_id),
                description=profile_data.get('description', ''),
                reference_audio=profile_data.get('reference_audio')
            )
        
        # Parse recognition settings
        recognition_data = speakers_data.get('recognition', {})
        recognition_config = SpeakerRecognitionConfig(
            confidence_threshold=recognition_data.get('confidence_threshold', 0.7),
            learning_mode=recognition_data.get('learning_mode', True),
            max_speakers=recognition_data.get('max_speakers', 4),
            voice_fingerprint_length=recognition_data.get('voice_fingerprint_length', 128)
        )
        
        speakers_config = SpeakersConfig(
            profiles=speaker_profiles,
            recognition=recognition_config
        )
        
        # Create OSC configuration
        osc_data = config_data.get('osc', {})
        osc_config = None
        if osc_data.get('enabled', False):
            osc_config = OSCConfig(
                enabled=True,
                host=osc_data.get('host', '127.0.0.1'),
                port=osc_data.get('port', 7000),
                speaking_start_address=osc_data.get('speaking_start_address', '/character/speaking/start'),
                speaking_stop_address=osc_data.get('speaking_stop_address', '/character/speaking/stop'),
                color_change_address=osc_data.get('color_change_address', '/character/color/change'),
                blank_output_address=osc_data.get('blank_output_address', '/character/blank')
            )
        
        # Create contexts configuration
        contexts_data = config_data.get('contexts', {})
        contexts_config = None
        if contexts_data.get('enabled', True):
            context_list = []
            for ctx_data in contexts_data.get('contexts', []):
                context = ContextConfig(
                    name=ctx_data['name'],
                    history_file=ctx_data['history_file'],
                    description=ctx_data.get('description'),
                    metadata=ctx_data.get('metadata', {}),
                    character_histories=ctx_data.get('character_histories'),
                    system_active_message=ctx_data.get('system_active_message')
                )
                context_list.append(context)
            
            contexts_config = ContextsConfig(
                enabled=True,
                contexts=context_list,
                state_dir=contexts_data.get('state_dir', './context_states'),
                auto_save_enabled=contexts_data.get('auto_save_enabled', True),
                auto_save_interval=contexts_data.get('auto_save_interval', 30)
            )

        # Create relay configuration for Discord/ChapterX integration
        relay_data = config_data.get('relay', {})
        relay_config = None
        if relay_data.get('enabled', False):
            # Parse bot configurations
            relay_bots = {}
            bots_data = relay_data.get('bots', {})
            for bot_name, bot_data in bots_data.items():
                if isinstance(bot_data, dict):
                    relay_bots[bot_name] = RelayBotConfig(
                        name=bot_name,
                        user_id=bot_data.get('user_id', ''),
                        voice_id=bot_data.get('voice_id'),
                        enabled=bot_data.get('enabled', True),
                        aliases=bot_data.get('aliases', [])
                    )
                elif isinstance(bot_data, str):
                    # Simple format: bot_name: user_id
                    relay_bots[bot_name] = RelayBotConfig(
                        name=bot_name,
                        user_id=bot_data
                    )

            # Parse speaker mappings for webhook display
            relay_speakers = {}
            speakers_data = relay_data.get('speakers', {})
            for speaker_name, speaker_data in speakers_data.items():
                if isinstance(speaker_data, dict):
                    relay_speakers[speaker_name] = RelaySpeakerConfig(
                        speaker_name=speaker_name,
                        discord_username=speaker_data.get('discord_username', speaker_name),
                        discord_user_id=speaker_data.get('discord_user_id'),
                        avatar_url=speaker_data.get('avatar_url'),
                        aliases=speaker_data.get('aliases', [])
                    )

            # Parse webhook configuration
            webhook_data = relay_data.get('webhook', {})
            webhook_config = None
            if webhook_data.get('bot_token'):
                webhook_config = RelayWebhookConfig(
                    bot_token=webhook_data['bot_token'],
                    username=webhook_data.get('username', 'Melodeus'),
                    avatar_url=webhook_data.get('avatar_url'),
                    include_speaker_name=webhook_data.get('include_speaker_name', True),
                    mention_mode=webhook_data.get('mention_mode', 'default')
                )

            relay_config = RelayConfig(
                enabled=True,
                url=relay_data.get('url', ''),
                client_id=relay_data.get('client_id', 'melodeus-1'),
                token=relay_data.get('token', ''),
                channels=relay_data.get('channels', []),
                default_bot=relay_data.get('default_bot'),
                bots=relay_bots,
                speakers=relay_speakers,
                webhook=webhook_config,
                reconnect_interval=relay_data.get('reconnect_interval', 5.0),
                auto_switch_on_connect=relay_data.get('auto_switch_on_connect', False)
            )

        config = VoiceAIConfig(
            conversation=conversation_config,
            stt=stt_config,
            tts=tts_config,
            audio=audio_config,
            logging=logging_config,
            development=development_config,
            speakers=speakers_config,
            camera=camera_config,
            echo_filter=echo_filter_config,
            osc=osc_config,
            contexts=contexts_config,
            relay=relay_config,
            _raw_config=config_data  # Store raw config for custom sections like flic
        )

        try:
            from mel_aec_audio import configure_audio_stream_from_config

            configure_audio_stream_from_config(config)
        except Exception as exc:
            print(f"⚠️ Unable to configure mel-aec stream from config: {exc}")

        return config
    
    @classmethod
    def create_example_config(cls, output_path: str = "config.yaml"):
        """Create an example configuration file."""
        example_config = {
            'api_keys': {
                'deepgram': 'your_deepgram_api_key_here',
                'elevenlabs': 'your_elevenlabs_api_key_here',
                'openai': 'your_openai_api_key_here'
            },
            'voice': {
                'id': 'T2KZm9rWPG5TgXTyjt7E'
            },
            'stt': {
                'model': 'nova-3',
                'language': 'en-US',
                'interim_results': True
            },
            'tts': {
                'model_id': 'eleven_multilingual_v2',
                'speed': 1.0,
                'stability': 0.5
            },
            'conversation': {
                'pause_threshold': 2.0,
                'min_words_for_submission': 3,
                'llm_model': 'chatgpt-4o-latest'
            }
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(example_config, f, default_flow_style=False, indent=2)
        
        print(f"✅ Example config created: {output_path}")

# Convenience functions
def load_config(config_path: Optional[str] = None, preset: Optional[str] = None) -> VoiceAIConfig:
    """Load configuration from YAML file with optional preset overrides."""
    return ConfigLoader.load(config_path, preset)

def create_example_config(output_path: str = "config.yaml"):
    """Create an example configuration file."""
    ConfigLoader.create_example_config(output_path)

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "create-example":
        output_path = sys.argv[2] if len(sys.argv) > 2 else "config.yaml"
        create_example_config(output_path)
    else:
        try:
            config = load_config()
            print("✅ Configuration loaded successfully!")
            print(f"   - Deepgram API key: {'✓' if config.conversation.deepgram_api_key else '✗'}")
            print(f"   - ElevenLabs API key: {'✓' if config.conversation.elevenlabs_api_key else '✗'}")
            print(f"   - OpenAI API key: {'✓' if config.conversation.openai_api_key else '✗'}")
            print(f"   - Voice ID: {config.conversation.voice_id}")
            print(f"   - STT Model: {config.stt.model}")
            print(f"   - TTS Model: {config.tts.model_id}")
            print(f"   - LLM Model: {config.conversation.llm_model}")
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            print("\n💡 To create an example config file, run:")
            print("   python config_loader.py create-example") 
