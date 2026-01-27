#!/usr/bin/env python3
"""
Relay Mode Manager for Melodeus

Manages the Discord relay mode where:
- STT transcripts are posted to Discord via webhook
- ChapterX bots run inference and stream responses via relay
- TTS receives chunks from relay instead of local LLM

This provides a clean interface for switching between LOCAL and DISCORD context modes.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable, Awaitable, Dict, Any, TYPE_CHECKING, AsyncGenerator
from enum import Enum

from relay_tts_client import RelayTTSClient, RelayConfig, BlockType, ActivationEvent, ActivationEndReason
from discord_webhook import DiscordWebhookPoster, DiscordWebhookConfig, BotMapping, SpeakerMapping, MentionMode

if TYPE_CHECKING:
    from config_loader import VoiceAIConfig, RelayConfig as RelayConfigType
    from async_tts_module import AsyncTTSStreamer

logger = logging.getLogger(__name__)


class ContextMode(Enum):
    """Context management mode."""
    LOCAL = "local"  # Local LLM inference, Melodeus manages context
    DISCORD = "discord"  # ChapterX handles inference, Discord is context source


@dataclass
class RelayModeState:
    """State for relay mode."""
    mode: ContextMode = ContextMode.LOCAL
    relay_connected: bool = False
    current_bot_speaking: Optional[str] = None
    voiced_text: str = ""  # Text that has been sent to TTS
    activation_in_progress: bool = False  # Bot is thinking/processing
    current_activation_bot: Optional[str] = None  # Bot ID of current activation


class RelayModeManager:
    """
    Manages Discord relay mode for Melodeus.

    Provides a clean interface for:
    - Switching between LOCAL and DISCORD context modes
    - Posting STT to Discord in DISCORD mode
    - Receiving TTS chunks from relay
    - Handling interruptions

    Usage:
        manager = RelayModeManager(config)
        await manager.start()

        # Switch to Discord mode
        await manager.set_mode(ContextMode.DISCORD)

        # In STT callback, check mode and handle accordingly
        if manager.is_discord_mode:
            await manager.post_utterance(text, speaker_name)
        else:
            # Normal local LLM processing
            ...

        # Get TTS stream from relay
        async for text in manager.get_relay_tts_stream():
            # Feed to TTS
            ...
    """

    def __init__(self, config: 'VoiceAIConfig'):
        self.config = config
        self.state = RelayModeState()

        # Components (initialized in start())
        self._relay_client: Optional[RelayTTSClient] = None
        self._webhook_poster: Optional[DiscordWebhookPoster] = None

        # Callbacks
        self._on_mode_change: Optional[Callable[[ContextMode], Awaitable[None]]] = None
        self._on_relay_connection_change: Optional[Callable[[bool], Awaitable[None]]] = None
        self._on_tts_stream_start: Optional[Callable[[str, str], Awaitable[None]]] = None  # bot_id, channel_id
        self._on_tts_stream_end: Optional[Callable[[str, str, str], Awaitable[None]]] = None  # bot_id, channel_id, content
        self._on_activation_start: Optional[Callable[[str, str], Awaitable[None]]] = None  # bot_id, channel_id
        self._on_activation_end: Optional[Callable[[str, str, str], Awaitable[None]]] = None  # bot_id, channel_id, reason

        # TTS streaming state
        self._current_stream_task: Optional[asyncio.Task] = None
        self._stream_cancelled = False

    @property
    def is_discord_mode(self) -> bool:
        """Check if currently in Discord context mode."""
        return self.state.mode == ContextMode.DISCORD

    @property
    def is_local_mode(self) -> bool:
        """Check if currently in local context mode."""
        return self.state.mode == ContextMode.LOCAL

    @property
    def is_relay_connected(self) -> bool:
        """Check if relay WebSocket is connected."""
        return self._relay_client is not None and self._relay_client.is_connected

    @property
    def current_mode(self) -> ContextMode:
        """Get the current context mode."""
        return self.state.mode

    @property
    def is_activation_in_progress(self) -> bool:
        """Check if a bot is currently processing/thinking."""
        return self.state.activation_in_progress

    @property
    def current_activation_bot(self) -> Optional[str]:
        """Get the bot ID of the current activation, if any."""
        return self.state.current_activation_bot

    def set_on_mode_change(self, callback: Callable[[ContextMode], Awaitable[None]]):
        """Set callback for mode changes."""
        self._on_mode_change = callback

    def set_on_relay_connection_change(self, callback: Callable[[bool], Awaitable[None]]):
        """Set callback for relay connection state changes."""
        self._on_relay_connection_change = callback

    def set_on_tts_stream_start(self, callback: Callable[[str, str], Awaitable[None]]):
        """Set callback for when TTS streaming starts from a bot."""
        self._on_tts_stream_start = callback

    def set_on_tts_stream_end(self, callback: Callable[[str, str, str], Awaitable[None]]):
        """Set callback for when TTS streaming ends."""
        self._on_tts_stream_end = callback

    def set_on_activation_start(self, callback: Callable[[str, str], Awaitable[None]]):
        """Set callback for when bot activation starts (thinking). Args: bot_id, channel_id."""
        self._on_activation_start = callback

    def set_on_activation_end(self, callback: Callable[[str, str, str], Awaitable[None]]):
        """Set callback for when bot activation ends. Args: bot_id, channel_id, reason."""
        self._on_activation_end = callback

    async def start(self) -> bool:
        """
        Start the relay mode manager.

        Initializes relay client and webhook poster if configured.
        Does not connect automatically - call connect_relay() or set_mode(DISCORD).

        Returns:
            True if initialization succeeded.
        """
        relay_config = self.config.relay
        if not relay_config or not relay_config.enabled:
            logger.info("Relay mode not configured or disabled")
            return True

        # Create relay client
        self._relay_client = self._create_relay_client(relay_config)

        # Create webhook poster
        if relay_config.webhook:
            self._webhook_poster = self._create_webhook_poster(relay_config)
            await self._webhook_poster.start()

        logger.info("Relay mode manager initialized")
        return True

    async def stop(self):
        """Stop the relay mode manager and clean up."""
        if self._current_stream_task:
            self._current_stream_task.cancel()
            try:
                await self._current_stream_task
            except asyncio.CancelledError:
                pass

        if self._relay_client:
            await self._relay_client.stop()

        if self._webhook_poster:
            await self._webhook_poster.stop()

        logger.info("Relay mode manager stopped")

    async def connect_relay(self) -> bool:
        """
        Connect to the TTS relay server.

        Returns:
            True if connection succeeded.
        """
        if not self._relay_client:
            logger.error("Relay client not initialized")
            return False

        success = await self._relay_client.start()

        if success:
            self.state.relay_connected = True
            logger.info("Connected to TTS relay")

            if self._on_relay_connection_change:
                await self._on_relay_connection_change(True)

            # Auto-switch to Discord mode if configured
            relay_config = self.config.relay
            if relay_config and relay_config.auto_switch_on_connect:
                await self.set_mode(ContextMode.DISCORD)

        return success

    async def disconnect_relay(self):
        """Disconnect from the TTS relay server."""
        if self._relay_client:
            await self._relay_client.stop()
            self.state.relay_connected = False

            if self._on_relay_connection_change:
                await self._on_relay_connection_change(False)

            logger.info("Disconnected from TTS relay")

    async def set_mode(self, mode: ContextMode) -> bool:
        """
        Switch context mode.

        Args:
            mode: The new context mode.

        Returns:
            True if mode switch succeeded.
        """
        if mode == self.state.mode:
            return True

        old_mode = self.state.mode

        # Validate mode switch
        if mode == ContextMode.DISCORD:
            if not self._relay_client:
                logger.error("Cannot switch to DISCORD mode: relay not configured")
                return False

            # Connect relay if not already connected
            if not self.is_relay_connected:
                if not await self.connect_relay():
                    logger.error("Cannot switch to DISCORD mode: relay connection failed")
                    return False

        self.state.mode = mode
        logger.info(f"Context mode changed: {old_mode.value} -> {mode.value}")

        if self._on_mode_change:
            await self._on_mode_change(mode)

        return True

    async def toggle_mode(self) -> ContextMode:
        """
        Toggle between LOCAL and DISCORD modes.

        Returns:
            The new mode.
        """
        new_mode = ContextMode.LOCAL if self.is_discord_mode else ContextMode.DISCORD
        await self.set_mode(new_mode)
        return self.state.mode

    async def post_utterance(
        self,
        text: str,
        channel_id: Optional[str] = None,
        speaker_name: Optional[str] = None,
        target_bot: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Post an STT utterance to Discord (DISCORD mode only).

        Args:
            text: The transcribed text.
            channel_id: Discord channel ID to post to. Uses first subscribed channel if None.
            speaker_name: Name of the speaker.
            target_bot: Specific bot to mention (uses config default if None).

        Returns:
            Dict with message_id, webhook_id, webhook_token if successful, None otherwise.
        """
        if not self.is_discord_mode:
            logger.warning("post_utterance called but not in DISCORD mode")
            return None

        if not self._webhook_poster:
            logger.error("Webhook poster not configured")
            return None

        # Use default channel from config if not specified
        relay_config = self.config.relay
        if channel_id is None and relay_config and relay_config.channels:
            channel_id = relay_config.channels[0]

        if not channel_id:
            logger.error("No channel_id specified and no default channels configured")
            return None

        # Note: target_bot is only used if explicitly provided
        # We don't auto-default to default_bot - mentions should be intentional

        result = await self._webhook_poster.post_transcript(
            text=text,
            channel_id=channel_id,
            speaker_name=speaker_name,
            target_bot=target_bot
        )

        if result:
            logger.debug(f"Posted to Discord channel {channel_id} (msg_id={result.get('message_id')}): {text[:50]}...")
        else:
            logger.error(f"Failed to post to Discord: {text[:50]}...")

        return result

    async def get_tts_stream(self):
        """
        Get an async generator of TTS text chunks from the relay.

        This is meant to be used as the text source for AsyncTTSStreamer.speak_text().

        Yields:
            Text chunks as they arrive from the relay.
        """
        if not self._relay_client or not self.is_relay_connected:
            logger.error("Cannot get TTS stream: relay not connected")
            return

        self._stream_cancelled = False
        self.state.voiced_text = ""

        async for chunk in self._relay_client.stream_chunks():
            if self._stream_cancelled:
                break

            # Track what we're yielding for interruption reporting
            self.state.voiced_text += chunk
            self._relay_client.mark_text_as_voiced(chunk)

            yield chunk

    async def wait_for_tts_stream(self):
        """
        Wait for a new TTS generation to start from the relay.

        Returns an async generator of text chunks, or None if not connected.
        This should be called after posting an utterance to Discord.

        Returns:
            Async generator of text chunks, or None.
        """
        if not self._relay_client or not self.is_relay_connected:
            return None

        return await self._relay_client.wait_for_generation()

    async def send_interruption(self, reason: str = "user_speech") -> bool:
        """
        Send an interruption event to the relay.

        Call this when the user speaks and TTS should be interrupted.
        The relay will forward this to ChapterX to truncate the Discord message.

        Args:
            reason: Reason for interruption.

        Returns:
            True if interruption was sent.
        """
        if not self._relay_client:
            return False

        self._stream_cancelled = True

        success = await self._relay_client.send_interruption(reason)

        if success:
            logger.info(f"Sent interruption: {reason}")

        return success

    def get_voice_id_for_bot(self, bot_id: str) -> Optional[str]:
        """
        Get the ElevenLabs voice ID for a bot.

        Args:
            bot_id: The bot identifier.

        Returns:
            Voice ID or None if not configured.
        """
        relay_config = self.config.relay
        if not relay_config:
            return None

        # Case-insensitive lookup since relay sends bot_id in lowercase
        bot_id_lower = bot_id.lower()
        for name, bot_config in relay_config.bots.items():
            if name.lower() == bot_id_lower:
                return bot_config.voice_id

        return None

    def get_tts_stream(self) -> Optional['AsyncGenerator[str, None]']:
        """
        Get the TTS chunk stream from the relay client.

        Returns:
            Async generator yielding text chunks, or None if not connected.
        """
        if not self._relay_client or not self._relay_client.is_connected:
            return None
        # Note: Queue is cleared in _handle_block_start when a new block begins,
        # not here, to avoid race conditions with chunks that may have already arrived
        return self._relay_client.stream_chunks()

    def _create_relay_client(self, relay_config: 'RelayConfigType') -> RelayTTSClient:
        """Create and configure the relay TTS client."""
        # Build voice routing from bot configs
        # Use lowercase keys since relay sends bot_id in lowercase
        voice_routing = {}
        for name, bot in relay_config.bots.items():
            voice_routing[name.lower()] = {
                "voice_id": bot.voice_id,
                "enabled": bot.enabled
            }

        config = RelayConfig(
            url=relay_config.url,
            client_id=relay_config.client_id,
            token=relay_config.token,
            channels=relay_config.channels,
            reconnect_interval=relay_config.reconnect_interval,
            voice_routing=voice_routing
        )

        client = RelayTTSClient(config)

        # Set up callbacks
        client.set_on_connection_change(self._handle_relay_connection_change)
        client.set_on_block_start(self._handle_block_start)
        client.set_on_block_complete(self._handle_block_complete)
        client.set_on_activation_start(self._handle_activation_start)
        client.set_on_activation_end(self._handle_activation_end)

        return client

    def _create_webhook_poster(self, relay_config: 'RelayConfigType') -> DiscordWebhookPoster:
        """Create and configure the Discord webhook poster."""
        webhook_config = relay_config.webhook

        # Build bot mappings
        bot_mappings = {}
        for name, bot in relay_config.bots.items():
            bot_mappings[name] = BotMapping(
                name=name,
                user_id=bot.user_id,
                aliases=bot.aliases,
                enabled=bot.enabled
            )

        # Build speaker mappings for webhook display
        speaker_mappings = {}
        for name, speaker in relay_config.speakers.items():
            speaker_mappings[name] = SpeakerMapping(
                speaker_name=speaker.speaker_name,
                discord_username=speaker.discord_username,
                discord_user_id=speaker.discord_user_id,
                avatar_url=speaker.avatar_url,
                aliases=speaker.aliases
            )

        config = DiscordWebhookConfig(
            bot_token=webhook_config.bot_token,
            bots=bot_mappings,
            speakers=speaker_mappings,
            default_bot=relay_config.default_bot,
            mention_mode=MentionMode(webhook_config.mention_mode),
            include_speaker_name=webhook_config.include_speaker_name,
            webhook_username=webhook_config.username,
            webhook_avatar_url=webhook_config.avatar_url
        )

        return DiscordWebhookPoster(config)

    async def _handle_relay_connection_change(self, connected: bool):
        """Handle relay connection state changes."""
        self.state.relay_connected = connected

        if not connected and self.is_discord_mode:
            # Lost connection while in Discord mode - could auto-switch to local
            logger.warning("Lost relay connection while in DISCORD mode")

        if self._on_relay_connection_change:
            await self._on_relay_connection_change(connected)

    async def _handle_block_start(self, bot_id: str, channel_id: str, block_type: BlockType):
        """Handle start of a new content block from relay."""
        if block_type == BlockType.TEXT:
            self.state.current_bot_speaking = bot_id
            self.state.voiced_text = ""

            if self._on_tts_stream_start:
                await self._on_tts_stream_start(bot_id, channel_id)

    async def _handle_block_complete(self, bot_id: str, channel_id: str, content: str, block_type: BlockType):
        """Handle completion of a content block from relay."""
        if block_type == BlockType.TEXT:
            self.state.current_bot_speaking = None

            if self._on_tts_stream_end:
                await self._on_tts_stream_end(bot_id, channel_id, content)

            # Update webhook poster's last speaker for mention mode
            if self._webhook_poster:
                self._webhook_poster.set_last_speaker(bot_id)

    async def _handle_activation_start(self, event: ActivationEvent):
        """Handle activation start from relay - bot has started thinking."""
        self.state.activation_in_progress = True
        self.state.current_activation_bot = event.bot_id

        logger.info(f"Bot activation started: {event.bot_id}")

        if self._on_activation_start:
            await self._on_activation_start(event.bot_id, event.channel_id)

    async def _handle_activation_end(self, event: ActivationEvent):
        """Handle activation end from relay - bot has finished thinking."""
        self.state.activation_in_progress = False
        self.state.current_activation_bot = None

        reason = event.reason.value if event.reason else "unknown"
        logger.info(f"Bot activation ended: {event.bot_id} ({reason})")

        if self._on_activation_end:
            await self._on_activation_end(event.bot_id, event.channel_id, reason)


# Convenience function
def create_relay_manager(config: 'VoiceAIConfig') -> Optional[RelayModeManager]:
    """
    Create a RelayModeManager if relay is configured.

    Args:
        config: The voice AI configuration.

    Returns:
        RelayModeManager instance or None if relay not configured.
    """
    if config.relay and config.relay.enabled:
        return RelayModeManager(config)
    return None
