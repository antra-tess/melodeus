#!/usr/bin/env python3
"""
Relay TTS Client for Melodeus

Connects to the TTS relay server to receive streaming text from ChapterX Discord bots
and send interruption events back. This enables Discord-based context management
with Melodeus handling only voice I/O.

Protocol: See /Users/olena/connectome-local/tts-relay/SPEC.md
"""

import asyncio
import json
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable, Awaitable, AsyncGenerator, List
from enum import Enum

import websockets
from websockets.client import WebSocketClientProtocol

logger = logging.getLogger(__name__)


class RelayMessageType(Enum):
    """Message types in the relay protocol."""
    # Auth
    AUTH = "auth"
    AUTH_OK = "auth_ok"
    AUTH_FAILED = "auth_failed"

    # Subscription
    SUBSCRIBE = "subscribe"
    SUBSCRIBED = "subscribed"

    # Content streaming (from bots via relay)
    CHUNK = "chunk"
    BLOCK_START = "block_start"
    BLOCK_COMPLETE = "block_complete"

    # Activation lifecycle (from bots via relay)
    ACTIVATION_START = "activation_start"
    ACTIVATION_END = "activation_end"

    # Interruption (to bots via relay)
    INTERRUPTION = "interruption"


class ActivationEndReason(Enum):
    """Reasons for activation ending."""
    COMPLETE = "complete"  # Finished normally
    ABORT = "abort"        # Interrupted by user
    ERROR = "error"        # Failed with error


class BlockType(Enum):
    """Types of content blocks from the LLM."""
    TEXT = "text"
    THINKING = "thinking"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


@dataclass
class RelayConfig:
    """Configuration for the relay TTS client."""
    url: str  # WebSocket URL, e.g., "ws://localhost:8800"
    client_id: str  # Unique identifier for this client
    token: str  # Authentication token
    channels: List[str] = field(default_factory=list)  # Discord channel IDs to subscribe to
    reconnect_interval: float = 5.0  # Seconds between reconnection attempts
    voice_routing: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # Bot ID -> voice config


@dataclass
class StreamingMessage:
    """A message being streamed from a bot."""
    bot_id: str
    channel_id: str
    user_id: str
    username: str
    block_index: int
    block_type: BlockType
    voiced_text: str = ""  # Accumulates what we've sent to TTS
    total_text: str = ""  # Accumulates all text received (including unvoiced)
    start_time: float = field(default_factory=time.time)


@dataclass
class ChunkEvent:
    """A chunk received from the relay."""
    bot_id: str
    channel_id: str
    user_id: str
    username: str
    text: str
    block_index: int
    block_type: BlockType
    visible: bool
    timestamp: int


@dataclass
class ActivationEvent:
    """An activation lifecycle event from the relay."""
    bot_id: str
    channel_id: str
    user_id: str
    username: str
    timestamp: int
    reason: Optional[ActivationEndReason] = None  # Only set for activation_end


class RelayTTSClient:
    """
    WebSocket client for the TTS relay server.

    Receives streaming text from ChapterX Discord bots and provides it as an
    async generator compatible with Melodeus TTS. Tracks what has been voiced
    for accurate interruption reporting.
    """

    def __init__(self, config: RelayConfig):
        self.config = config
        self._ws: Optional[WebSocketClientProtocol] = None
        self._connected = False
        self._authenticated = False
        self._subscribed_channels: List[str] = []

        # Current streaming state
        self._current_message: Optional[StreamingMessage] = None
        self._chunk_queue: asyncio.Queue[Optional[ChunkEvent]] = asyncio.Queue()

        # Callbacks
        self._on_block_start: Optional[Callable[[str, str, BlockType], Awaitable[None]]] = None
        self._on_block_complete: Optional[Callable[[str, str, str, BlockType], Awaitable[None]]] = None
        self._on_connection_change: Optional[Callable[[bool], Awaitable[None]]] = None
        self._on_activation_start: Optional[Callable[[ActivationEvent], Awaitable[None]]] = None
        self._on_activation_end: Optional[Callable[[ActivationEvent], Awaitable[None]]] = None

        # Activation tracking
        self._activation_in_progress = False
        self._current_activation: Optional[ActivationEvent] = None

        # Background tasks
        self._receive_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None
        self._should_run = False

    @property
    def is_connected(self) -> bool:
        """Check if connected and authenticated."""
        return self._connected and self._authenticated

    @property
    def current_bot_id(self) -> Optional[str]:
        """Get the bot ID of the current streaming message."""
        return self._current_message.bot_id if self._current_message else None

    @property
    def current_channel_id(self) -> Optional[str]:
        """Get the channel ID of the current streaming message."""
        return self._current_message.channel_id if self._current_message else None

    @property
    def voiced_text(self) -> str:
        """Get the text that has been voiced so far."""
        return self._current_message.voiced_text if self._current_message else ""

    @property
    def is_activation_in_progress(self) -> bool:
        """Check if an activation is currently in progress (bot is thinking/responding)."""
        return self._activation_in_progress

    @property
    def current_activation(self) -> Optional[ActivationEvent]:
        """Get the current activation event if one is in progress."""
        return self._current_activation

    def set_on_activation_start(self, callback: Callable[[ActivationEvent], Awaitable[None]]):
        """Set callback for when bot activation starts. Args: ActivationEvent."""
        self._on_activation_start = callback

    def set_on_activation_end(self, callback: Callable[[ActivationEvent], Awaitable[None]]):
        """Set callback for when bot activation ends. Args: ActivationEvent (with reason)."""
        self._on_activation_end = callback

    def set_on_block_start(self, callback: Callable[[str, str, BlockType], Awaitable[None]]):
        """Set callback for when a new block starts. Args: bot_id, channel_id, block_type."""
        self._on_block_start = callback

    def set_on_block_complete(self, callback: Callable[[str, str, str, BlockType], Awaitable[None]]):
        """Set callback for when a block completes. Args: bot_id, channel_id, content, block_type."""
        self._on_block_complete = callback

    def set_on_connection_change(self, callback: Callable[[bool], Awaitable[None]]):
        """Set callback for connection state changes. Args: is_connected."""
        self._on_connection_change = callback

    def clear_chunk_queue(self):
        """Clear any stale chunks from the queue."""
        cleared = 0
        while not self._chunk_queue.empty():
            try:
                self._chunk_queue.get_nowait()
                cleared += 1
            except asyncio.QueueEmpty:
                break
        if cleared > 0:
            logger.info(f"DEBUG: Cleared {cleared} stale items from chunk queue")

    async def connect(self) -> bool:
        """
        Connect to the relay server and authenticate.

        Returns:
            True if connection and authentication succeeded.
        """
        try:
            ws_url = f"{self.config.url}/tts"
            logger.info(f"Connecting to relay at {ws_url}")

            self._ws = await websockets.connect(
                ws_url,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=5
            )
            self._connected = True

            # Authenticate
            await self._send({
                "type": RelayMessageType.AUTH.value,
                "clientId": self.config.client_id,
                "token": self.config.token
            })

            # Wait for auth response
            response = await asyncio.wait_for(self._ws.recv(), timeout=10.0)
            msg = json.loads(response)

            if msg.get("type") == RelayMessageType.AUTH_OK.value:
                self._authenticated = True
                logger.info("Relay authentication successful")

                if self._on_connection_change:
                    await self._on_connection_change(True)

                return True
            else:
                logger.error(f"Relay authentication failed: {msg}")
                await self._cleanup_connection()
                return False

        except Exception as e:
            logger.error(f"Failed to connect to relay: {e}")
            await self._cleanup_connection()
            return False

    async def subscribe(self, channels: Optional[List[str]] = None) -> bool:
        """
        Subscribe to Discord channels.

        Args:
            channels: List of channel IDs. If None, uses channels from config.

        Returns:
            True if subscription succeeded.
        """
        if not self.is_connected:
            logger.error("Cannot subscribe: not connected")
            return False

        channels_to_sub = channels or self.config.channels
        if not channels_to_sub:
            logger.warning("No channels to subscribe to")
            return True

        try:
            await self._send({
                "type": RelayMessageType.SUBSCRIBE.value,
                "channels": channels_to_sub
            })

            # Wait for subscription confirmation
            response = await asyncio.wait_for(self._ws.recv(), timeout=10.0)
            msg = json.loads(response)

            if msg.get("type") == RelayMessageType.SUBSCRIBED.value:
                self._subscribed_channels = msg.get("channels", channels_to_sub)
                logger.info(f"Subscribed to channels: {self._subscribed_channels}")
                return True
            else:
                logger.error(f"Subscription failed: {msg}")
                return False

        except Exception as e:
            logger.error(f"Failed to subscribe: {e}")
            return False

    async def start(self) -> bool:
        """
        Start the relay client (connect, authenticate, subscribe, start receiving).

        Returns:
            True if started successfully.
        """
        self._should_run = True

        if not await self.connect():
            return False

        if not await self.subscribe():
            return False

        # Start receive loop
        self._receive_task = asyncio.create_task(self._receive_loop())

        return True

    async def stop(self):
        """Stop the relay client and disconnect."""
        self._should_run = False

        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        if self._reconnect_task:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
            self._reconnect_task = None

        await self._cleanup_connection()

    async def send_interruption(self, reason: str = "user_speech") -> bool:
        """
        Send an interruption event to the relay.

        This notifies the bot that TTS was interrupted and provides the text
        that was actually spoken, so the bot can edit its Discord message.

        Args:
            reason: Reason for interruption ("user_speech", "manual", "timeout")

        Returns:
            True if interruption was sent.
        """
        if not self.is_connected:
            logger.warning("Cannot send interruption: not connected")
            return False

        if not self._current_message:
            logger.warning("Cannot send interruption: no current message")
            return False

        try:
            await self._send({
                "type": RelayMessageType.INTERRUPTION.value,
                "botId": self._current_message.bot_id,
                "channelId": self._current_message.channel_id,
                "spokenText": self._current_message.voiced_text,
                "reason": reason,
                "timestamp": int(time.time() * 1000)
            })

            logger.info(f"Sent interruption to {self._current_message.bot_id}: "
                       f"'{self._current_message.voiced_text[:50]}...' ({reason})")

            # Clear current message state
            self._current_message = None

            # Signal end of stream to any waiting consumers
            await self._chunk_queue.put(None)

            return True

        except Exception as e:
            logger.error(f"Failed to send interruption: {e}")
            return False

    def mark_text_as_voiced(self, text: str):
        """
        Mark text as having been voiced by TTS.

        Call this as text is actually spoken to keep accurate track of
        what was voiced for interruption reporting.

        Args:
            text: The text that was just voiced.
        """
        if self._current_message:
            self._current_message.voiced_text += text

    def get_voice_config(self, bot_id: str) -> Optional[Dict[str, Any]]:
        """
        Get voice configuration for a bot.

        Args:
            bot_id: The bot identifier.

        Returns:
            Voice config dict with voice_id, enabled, etc. or None if not configured.
        """
        return self.config.voice_routing.get(bot_id)

    def should_voice_bot(self, bot_id: str) -> bool:
        """
        Check if a bot should be voiced.

        Args:
            bot_id: The bot identifier.

        Returns:
            True if this bot should be voiced based on config.
        """
        config = self.get_voice_config(bot_id)
        if config is None:
            return False
        return config.get("enabled", True)

    async def stream_chunks(self) -> AsyncGenerator[str, None]:
        """
        Async generator that yields text chunks for TTS.

        This is compatible with the existing Melodeus TTS interface.
        Only yields chunks from visible text blocks that should be voiced.

        Yields:
            Text chunks as they arrive from the relay.
        """
        logger.info("DEBUG: stream_chunks() started")
        while True:
            try:
                logger.info("DEBUG: stream_chunks waiting for chunk...")
                chunk = await self._chunk_queue.get()
                logger.info(f"DEBUG: stream_chunks got chunk: {chunk}")

                if chunk is None:
                    # End of stream (block complete or interruption)
                    logger.info("DEBUG: stream_chunks got None, ending")
                    break

                # Only yield visible text from bots we should voice
                if chunk.visible and chunk.block_type == BlockType.TEXT:
                    if self.should_voice_bot(chunk.bot_id):
                        logger.info(f"DEBUG: stream_chunks yielding: '{chunk.text[:30]}...'")
                        yield chunk.text
                    else:
                        logger.info(f"DEBUG: stream_chunks skipping (bot not voiced): {chunk.bot_id}")
                else:
                    logger.info(f"DEBUG: stream_chunks skipping (not visible/text): visible={chunk.visible} type={chunk.block_type}")

            except asyncio.CancelledError:
                logger.info("DEBUG: stream_chunks cancelled")
                break
        logger.info("DEBUG: stream_chunks() finished")

    async def wait_for_generation(self) -> Optional[AsyncGenerator[str, None]]:
        """
        Wait for a new generation to start and return a stream of chunks.

        Returns:
            An async generator of text chunks, or None if disconnected.
        """
        if not self.is_connected:
            return None

        # Clear any stale chunks
        while not self._chunk_queue.empty():
            try:
                self._chunk_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Wait for a block_start event (handled in _receive_loop)
        # The stream_chunks() generator will then yield the content
        return self.stream_chunks()

    async def _send(self, message: Dict[str, Any]):
        """Send a JSON message to the relay."""
        if self._ws:
            await self._ws.send(json.dumps(message))

    async def _receive_loop(self):
        """Background task that receives and processes messages from the relay."""
        while self._should_run and self._ws:
            try:
                message = await self._ws.recv()
                msg = json.loads(message)
                await self._handle_message(msg)

            except websockets.ConnectionClosed:
                logger.warning("Relay connection closed")
                await self._handle_disconnect()
                break

            except asyncio.CancelledError:
                break

            except Exception as e:
                logger.error(f"Error in receive loop: {e}")
                continue

    async def _handle_message(self, msg: Dict[str, Any]):
        """Handle a message from the relay."""
        msg_type = msg.get("type")

        if msg_type == RelayMessageType.CHUNK.value:
            await self._handle_chunk(msg)

        elif msg_type == RelayMessageType.BLOCK_START.value:
            await self._handle_block_start(msg)

        elif msg_type == RelayMessageType.BLOCK_COMPLETE.value:
            await self._handle_block_complete(msg)

        elif msg_type == RelayMessageType.ACTIVATION_START.value:
            await self._handle_activation_start(msg)

        elif msg_type == RelayMessageType.ACTIVATION_END.value:
            await self._handle_activation_end(msg)

        else:
            logger.debug(f"Unhandled message type: {msg_type}")

    async def _handle_chunk(self, msg: Dict[str, Any]):
        """Handle a chunk message."""
        try:
            block_type = BlockType(msg.get("blockType", "text"))
        except ValueError:
            block_type = BlockType.TEXT

        chunk = ChunkEvent(
            bot_id=msg.get("botId", ""),
            channel_id=msg.get("channelId", ""),
            user_id=msg.get("userId", ""),
            username=msg.get("username", ""),
            text=msg.get("text", ""),
            block_index=msg.get("blockIndex", 0),
            block_type=block_type,
            visible=msg.get("visible", True),
            timestamp=msg.get("timestamp", int(time.time() * 1000))
        )

        # Update current message tracking
        if self._current_message and self._current_message.bot_id == chunk.bot_id:
            self._current_message.total_text += chunk.text

        # Queue chunk for streaming
        logger.info(f"DEBUG: Queueing chunk: '{chunk.text[:30]}...' visible={chunk.visible} type={chunk.block_type}")
        await self._chunk_queue.put(chunk)

    async def _handle_block_start(self, msg: Dict[str, Any]):
        """Handle a block_start message."""
        # Clear any stale chunks from previous streams BEFORE new chunks arrive
        self.clear_chunk_queue()

        try:
            block_type = BlockType(msg.get("blockType", "text"))
        except ValueError:
            block_type = BlockType.TEXT

        bot_id = msg.get("botId", "")
        channel_id = msg.get("channelId", "")

        # Start tracking new message
        self._current_message = StreamingMessage(
            bot_id=bot_id,
            channel_id=channel_id,
            user_id=msg.get("userId", ""),
            username=msg.get("username", ""),
            block_index=msg.get("blockIndex", 0),
            block_type=block_type
        )

        logger.debug(f"Block start: {bot_id} ({block_type.value})")

        if self._on_block_start:
            await self._on_block_start(bot_id, channel_id, block_type)

    async def _handle_block_complete(self, msg: Dict[str, Any]):
        """Handle a block_complete message."""
        try:
            block_type = BlockType(msg.get("blockType", "text"))
        except ValueError:
            block_type = BlockType.TEXT

        bot_id = msg.get("botId", "")
        channel_id = msg.get("channelId", "")
        content = msg.get("content", "")

        logger.debug(f"Block complete: {bot_id} ({block_type.value})")

        # Signal end of stream
        await self._chunk_queue.put(None)

        if self._on_block_complete:
            await self._on_block_complete(bot_id, channel_id, content, block_type)

        # Clear current message if this was the one being tracked
        if self._current_message and self._current_message.bot_id == bot_id:
            self._current_message = None

    async def _handle_activation_start(self, msg: Dict[str, Any]):
        """Handle an activation_start message - bot has started processing."""
        event = ActivationEvent(
            bot_id=msg.get("botId", ""),
            channel_id=msg.get("channelId", ""),
            user_id=msg.get("userId", ""),
            username=msg.get("username", ""),
            timestamp=msg.get("timestamp", int(time.time() * 1000))
        )

        self._activation_in_progress = True
        self._current_activation = event

        logger.info(f"Activation started: {event.bot_id} in {event.channel_id}")

        if self._on_activation_start:
            await self._on_activation_start(event)

    async def _handle_activation_end(self, msg: Dict[str, Any]):
        """Handle an activation_end message - bot has finished processing."""
        try:
            reason = ActivationEndReason(msg.get("reason", "complete"))
        except ValueError:
            reason = ActivationEndReason.COMPLETE

        event = ActivationEvent(
            bot_id=msg.get("botId", ""),
            channel_id=msg.get("channelId", ""),
            user_id=msg.get("userId", ""),
            username=msg.get("username", ""),
            timestamp=msg.get("timestamp", int(time.time() * 1000)),
            reason=reason
        )

        self._activation_in_progress = False
        self._current_activation = None

        logger.info(f"Activation ended: {event.bot_id} ({reason.value})")

        if self._on_activation_end:
            await self._on_activation_end(event)

    async def _handle_disconnect(self):
        """Handle disconnection from the relay."""
        was_connected = self._connected
        await self._cleanup_connection()

        if was_connected and self._on_connection_change:
            await self._on_connection_change(False)

        # Start reconnection if we should still be running
        if self._should_run:
            self._reconnect_task = asyncio.create_task(self._reconnect_loop())

    async def _reconnect_loop(self):
        """Background task that attempts to reconnect."""
        while self._should_run:
            logger.info(f"Attempting to reconnect in {self.config.reconnect_interval}s...")
            await asyncio.sleep(self.config.reconnect_interval)

            if await self.connect() and await self.subscribe():
                logger.info("Reconnected to relay")
                self._receive_task = asyncio.create_task(self._receive_loop())
                break

    async def _cleanup_connection(self):
        """Clean up connection state."""
        self._connected = False
        self._authenticated = False
        self._subscribed_channels = []

        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None


# Convenience function for creating a configured client
def create_relay_client(
    url: str,
    client_id: str,
    token: str,
    channels: List[str],
    voice_routing: Optional[Dict[str, Dict[str, Any]]] = None
) -> RelayTTSClient:
    """
    Create a configured RelayTTSClient.

    Args:
        url: WebSocket URL for the relay server.
        client_id: Unique identifier for this client.
        token: Authentication token.
        channels: Discord channel IDs to subscribe to.
        voice_routing: Optional bot voice configuration.

    Returns:
        Configured RelayTTSClient instance.
    """
    config = RelayConfig(
        url=url,
        client_id=client_id,
        token=token,
        channels=channels,
        voice_routing=voice_routing or {}
    )
    return RelayTTSClient(config)
