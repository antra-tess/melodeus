#!/usr/bin/env python3
"""
Discord Webhook Poster for Melodeus

Posts STT transcripts to Discord via webhook, triggering ChapterX bot responses.
Supports @mentions for targeting specific bots and speaker attribution.

Dynamically creates webhooks per channel using Discord bot token.
Also provides methods to fetch channel history for context switching.
"""

import asyncio
import aiohttp
import logging
import re
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

DISCORD_API_BASE = "https://discord.com/api/v10"


class MentionMode(Enum):
    """How to determine which bot to mention."""
    DEFAULT = "default"  # Always mention the default bot
    EXPLICIT = "explicit"  # User must say bot name to mention it
    ROUND_ROBIN = "round_robin"  # Alternate between bots
    LAST_SPEAKER = "last_speaker"  # Mention whoever spoke last


@dataclass
class BotMapping:
    """Mapping from bot name to Discord user ID."""
    name: str  # Friendly name (e.g., "claude", "aria")
    user_id: str  # Discord user ID for @mention
    aliases: List[str] = field(default_factory=list)  # Alternative names to match
    enabled: bool = True


@dataclass
class SpeakerMapping:
    """Mapping from speaker name to Discord user info for webhook display."""
    speaker_name: str  # Name from STT (e.g., "antra", "User 1")
    discord_username: str  # Username to display in Discord
    discord_user_id: Optional[str] = None  # Discord user ID (for avatar lookup)
    avatar_url: Optional[str] = None  # Direct avatar URL (overrides user_id lookup)
    aliases: List[str] = field(default_factory=list)  # Alternative speaker names to match

    def get_avatar_url(self) -> Optional[str]:
        """Get avatar URL, constructing from user_id if needed."""
        if self.avatar_url:
            return self.avatar_url
        # Note: To get avatar from user_id, you need the avatar hash from Discord API
        # For now, return None if no direct avatar_url is set
        return None


@dataclass
class DiscordWebhookConfig:
    """Configuration for Discord webhook posting."""
    bot_token: str  # Discord bot token for API access
    bots: Dict[str, BotMapping] = field(default_factory=dict)  # name -> BotMapping
    speakers: Dict[str, SpeakerMapping] = field(default_factory=dict)  # speaker_name -> SpeakerMapping
    default_bot: Optional[str] = None  # Default bot to mention if none specified
    mention_mode: MentionMode = MentionMode.DEFAULT
    include_speaker_name: bool = True  # Prefix messages with speaker name (if no speaker mapping)
    webhook_username: Optional[str] = "Melodeus"  # Default username to display in Discord
    webhook_avatar_url: Optional[str] = None  # Default avatar URL for webhook
    webhook_url: Optional[str] = None  # Optional static webhook URL (if not using dynamic)


@dataclass
class CachedWebhook:
    """Cached webhook info for a channel."""
    id: str
    token: str
    url: str


class DiscordWebhookPoster:
    """
    Posts STT transcripts to Discord via webhook.

    This triggers ChapterX bots to respond by @mentioning them in messages.
    The bots stream their responses back via the TTS relay.

    Dynamically creates webhooks per channel using the bot token.
    """

    def __init__(self, config: DiscordWebhookConfig):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_bot_mentioned: Optional[str] = None
        self._round_robin_index: int = 0

        # Cache of channel_id -> webhook info
        self._webhook_cache: Dict[str, CachedWebhook] = {}

        # Cache of user_id -> {username, avatar_url}
        self._user_cache: Dict[str, Dict[str, str]] = {}

        # Build name -> bot lookup including aliases
        self._name_lookup: Dict[str, str] = {}
        for name, bot in config.bots.items():
            self._name_lookup[name.lower()] = name
            for alias in bot.aliases:
                self._name_lookup[alias.lower()] = name

        # Build speaker name -> SpeakerMapping lookup including aliases
        self._speaker_lookup: Dict[str, SpeakerMapping] = {}
        for name, speaker in config.speakers.items():
            self._speaker_lookup[name.lower()] = speaker
            self._speaker_lookup[speaker.speaker_name.lower()] = speaker
            for alias in speaker.aliases:
                self._speaker_lookup[alias.lower()] = speaker

    def get_speaker_mapping(self, speaker_name: Optional[str]) -> Optional[SpeakerMapping]:
        """Look up speaker mapping by name (case-insensitive, includes aliases)."""
        if not speaker_name:
            return None
        return self._speaker_lookup.get(speaker_name.lower())

    def get_cached_user_info(self, user_id: str) -> Optional[Dict[str, str]]:
        """Get cached user info (username, avatar_url) by user ID."""
        return self._user_cache.get(user_id)

    async def _fetch_user_info(self, user_id: str) -> Optional[Dict[str, str]]:
        """Fetch user info from Discord API and cache it."""
        if not self._session:
            return None

        try:
            url = f"{DISCORD_API_BASE}/users/{user_id}"
            async with self._session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    username = data.get("username", "")
                    display_name = data.get("global_name") or username
                    avatar_hash = data.get("avatar")

                    avatar_url = None
                    if avatar_hash:
                        # Construct avatar URL from hash
                        ext = "gif" if avatar_hash.startswith("a_") else "png"
                        avatar_url = f"https://cdn.discordapp.com/avatars/{user_id}/{avatar_hash}.{ext}"

                    user_info = {
                        "username": username,  # Discord username (e.g., "antra_tessera")
                        "display_name": display_name,  # Display name (e.g., "antra")
                        "avatar_url": avatar_url
                    }
                    self._user_cache[user_id] = user_info
                    logger.info(f"Fetched Discord user info for {user_id}: @{username} ({display_name})")
                    return user_info
                else:
                    logger.warning(f"Failed to fetch user {user_id}: {response.status}")
        except Exception as e:
            logger.error(f"Error fetching user {user_id}: {e}")
        return None

    async def _fetch_all_speaker_users(self):
        """Fetch Discord user info for all speakers with user IDs and add to lookup."""
        for name, speaker in self.config.speakers.items():
            if speaker.discord_user_id:
                user_info = await self._fetch_user_info(speaker.discord_user_id)
                if user_info:
                    # Add both username and display_name as lookup aliases
                    username = user_info.get("username", "").lower()
                    display_name = user_info.get("display_name", "").lower()
                    if username and username not in self._speaker_lookup:
                        self._speaker_lookup[username] = speaker
                    if display_name and display_name not in self._speaker_lookup:
                        self._speaker_lookup[display_name] = speaker

    async def start(self):
        """Start the webhook poster (creates HTTP session, fetches user info)."""
        if self._session is None:
            self._session = aiohttp.ClientSession(headers={
                "Authorization": f"Bot {self.config.bot_token}",
                "Content-Type": "application/json"
            })

        # Fetch user info for all configured speakers
        await self._fetch_all_speaker_users()

        logger.info("Discord webhook poster started")

    async def stop(self):
        """Stop the webhook poster (closes HTTP session)."""
        if self._session:
            await self._session.close()
            self._session = None
        logger.info("Discord webhook poster stopped")

    def get_our_webhook_ids(self) -> set:
        """Get the set of webhook IDs that belong to us (Melodeus)."""
        return {wh.id for wh in self._webhook_cache.values()}

    async def _get_or_create_webhook(self, channel_id: str) -> Optional[CachedWebhook]:
        """
        Get or create a webhook for a channel.

        Args:
            channel_id: Discord channel ID

        Returns:
            CachedWebhook or None if creation failed
        """
        # Check cache first
        if channel_id in self._webhook_cache:
            return self._webhook_cache[channel_id]

        if not self._session:
            await self.start()

        try:
            # First, check if we already have a Melodeus webhook in this channel
            existing = await self._find_existing_webhook(channel_id)
            if existing:
                self._webhook_cache[channel_id] = existing
                logger.info(f"Found existing webhook for channel {channel_id}")
                return existing

            # Create a new webhook
            webhook = await self._create_webhook(channel_id)
            if webhook:
                self._webhook_cache[channel_id] = webhook
                logger.info(f"Created new webhook for channel {channel_id}")
                return webhook

        except Exception as e:
            logger.error(f"Failed to get/create webhook for channel {channel_id}: {e}")

        return None

    async def _find_existing_webhook(self, channel_id: str) -> Optional[CachedWebhook]:
        """Find an existing Melodeus webhook in a channel."""
        try:
            url = f"{DISCORD_API_BASE}/channels/{channel_id}/webhooks"
            async with self._session.get(url) as response:
                if response.status == 200:
                    webhooks = await response.json()
                    for wh in webhooks:
                        # Look for our webhook by name
                        if wh.get("name") == (self.config.webhook_username or "Melodeus"):
                            return CachedWebhook(
                                id=wh["id"],
                                token=wh["token"],
                                url=f"https://discord.com/api/webhooks/{wh['id']}/{wh['token']}"
                            )
                elif response.status == 403:
                    logger.warning(f"No permission to view webhooks in channel {channel_id}")
                else:
                    error = await response.text()
                    logger.warning(f"Failed to list webhooks: {response.status} {error}")
        except Exception as e:
            logger.error(f"Error finding webhook: {e}")

        return None

    async def _create_webhook(self, channel_id: str) -> Optional[CachedWebhook]:
        """Create a new webhook in a channel."""
        try:
            url = f"{DISCORD_API_BASE}/channels/{channel_id}/webhooks"
            payload = {
                "name": self.config.webhook_username or "Melodeus"
            }

            async with self._session.post(url, json=payload) as response:
                if response.status in (200, 201):
                    wh = await response.json()
                    return CachedWebhook(
                        id=wh["id"],
                        token=wh["token"],
                        url=f"https://discord.com/api/webhooks/{wh['id']}/{wh['token']}"
                    )
                elif response.status == 403:
                    logger.error(f"No permission to create webhook in channel {channel_id}")
                else:
                    error = await response.text()
                    logger.error(f"Failed to create webhook: {response.status} {error}")
        except Exception as e:
            logger.error(f"Error creating webhook: {e}")

        return None

    async def post_transcript(
        self,
        text: str,
        channel_id: str,
        speaker_name: Optional[str] = None,
        target_bot: Optional[str] = None,
        force_mention: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Post an STT transcript to Discord.

        Args:
            text: The transcribed text to post.
            channel_id: Discord channel ID to post to.
            speaker_name: Name of the speaker (from STT diarization).
            target_bot: Specific bot to mention. If None, determined by mention_mode.
            force_mention: If True, always include a mention even if text seems like
                          a continuation.

        Returns:
            Dict with message_id, webhook_id, webhook_token if successful, None otherwise.
        """
        if not self._session:
            await self.start()

        # Get or create webhook for this channel
        webhook = await self._get_or_create_webhook(channel_id)
        if not webhook:
            logger.error(f"No webhook available for channel {channel_id}")
            return None

        # Determine which bot to mention
        mention_str = ""
        bot_to_mention = self._determine_bot_to_mention(text, target_bot)

        if bot_to_mention:
            bot_mapping = self.config.bots.get(bot_to_mention)
            if bot_mapping and bot_mapping.enabled:
                mention_str = f"<@{bot_mapping.user_id}> "
                self._last_bot_mentioned = bot_to_mention
                logger.debug(f"Mentioning bot: {bot_to_mention}")

        # Look up speaker mapping for webhook display
        speaker_mapping = self.get_speaker_mapping(speaker_name)

        # Format the message - don't include speaker prefix if we have a mapping
        # (the webhook username will show who it's from)
        if speaker_mapping:
            message = f"{mention_str}{text}"
        else:
            message = self._format_message(text, speaker_name, mention_str)

        # Get speaker-specific webhook display settings
        webhook_username = None
        webhook_avatar = None
        if speaker_mapping:
            # Try to get dynamic user info from Discord API cache
            if speaker_mapping.discord_user_id:
                user_info = self.get_cached_user_info(speaker_mapping.discord_user_id)
                if user_info:
                    webhook_username = user_info.get("username") or speaker_mapping.discord_username
                    webhook_avatar = user_info.get("avatar_url") or speaker_mapping.get_avatar_url()
                else:
                    webhook_username = speaker_mapping.discord_username
                    webhook_avatar = speaker_mapping.get_avatar_url()
            else:
                webhook_username = speaker_mapping.discord_username
                webhook_avatar = speaker_mapping.get_avatar_url()

        # Post to webhook
        return await self._post_to_webhook(
            message, webhook.url,
            username_override=webhook_username,
            avatar_override=webhook_avatar
        )

    async def post_raw(self, content: str, channel_id: str) -> Optional[Dict[str, Any]]:
        """
        Post raw content to Discord without any formatting.

        Args:
            content: Raw message content.
            channel_id: Discord channel ID to post to.

        Returns:
            Dict with message_id, webhook_id, webhook_token if successful, None otherwise.
        """
        if not self._session:
            await self.start()

        webhook = await self._get_or_create_webhook(channel_id)
        if not webhook:
            return None

        return await self._post_to_webhook(content, webhook.url)

    async def post_mention(self, user_id: str, channel_id: str) -> Optional[Dict[str, Any]]:
        """
        Post a mention to Discord to trigger a bot.

        Args:
            user_id: Discord user ID to mention.
            channel_id: Discord channel ID.

        Returns:
            Dict with message_id, webhook_id, webhook_token if successful, None otherwise.
        """
        if not self._session:
            await self.start()

        if not channel_id:
            logger.error("No channel_id specified for post_mention")
            return None

        webhook = await self._get_or_create_webhook(channel_id)
        if not webhook:
            return None

        # Just post the mention
        content = f"<@{user_id}>"
        return await self._post_to_webhook(content, webhook.url)

    def _determine_bot_to_mention(
        self,
        text: str,
        target_bot: Optional[str]
    ) -> Optional[str]:
        """
        Determine which bot to mention based on config and text.

        Args:
            text: The transcript text (may contain bot name).
            target_bot: Explicitly specified target bot.

        Returns:
            Bot name to mention, or None.
        """
        # Explicit target takes priority
        if target_bot:
            return target_bot if target_bot in self.config.bots else None

        mode = self.config.mention_mode

        if mode == MentionMode.EXPLICIT:
            # Look for bot name in text
            return self._find_bot_in_text(text)

        elif mode == MentionMode.DEFAULT:
            return self.config.default_bot

        elif mode == MentionMode.LAST_SPEAKER:
            return self._last_bot_mentioned or self.config.default_bot

        elif mode == MentionMode.ROUND_ROBIN:
            enabled_bots = [
                name for name, bot in self.config.bots.items()
                if bot.enabled
            ]
            if enabled_bots:
                bot = enabled_bots[self._round_robin_index % len(enabled_bots)]
                self._round_robin_index += 1
                return bot

        return self.config.default_bot

    def _find_bot_in_text(self, text: str) -> Optional[str]:
        """
        Find a bot name mentioned in the text.

        Args:
            text: Text to search.

        Returns:
            Bot name if found, None otherwise.
        """
        text_lower = text.lower()

        # Check for exact word matches
        words = re.findall(r'\b\w+\b', text_lower)
        for word in words:
            if word in self._name_lookup:
                return self._name_lookup[word]

        # Check for partial matches (e.g., "hey claude" or "claude,")
        for name in self._name_lookup:
            if name in text_lower:
                return self._name_lookup[name]

        return None

    def _format_message(
        self,
        text: str,
        speaker_name: Optional[str],
        mention_str: str
    ) -> str:
        """
        Format the message for Discord.

        Args:
            text: The transcript text.
            speaker_name: Name of the speaker.
            mention_str: The mention string (e.g., "<@123456> ").

        Returns:
            Formatted message.
        """
        # Build message parts
        parts = []

        # Add speaker attribution if configured
        if self.config.include_speaker_name and speaker_name:
            parts.append(f"**{speaker_name}:**")

        # Add mention and text
        parts.append(f"{mention_str}{text}")

        return " ".join(parts)

    async def _post_to_webhook(
        self,
        content: str,
        webhook_url: str,
        username_override: Optional[str] = None,
        avatar_override: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Post content to the Discord webhook.

        Args:
            content: Message content.
            webhook_url: The webhook URL to post to.
            username_override: Override username for this message (speaker-specific).
            avatar_override: Override avatar URL for this message (speaker-specific).

        Returns:
            Dict with message_id, webhook_id, webhook_token if successful, None otherwise.
        """
        if not self._session:
            logger.error("No HTTP session available")
            return None

        payload: Dict[str, Any] = {
            "content": content
        }

        # Use override if provided, otherwise fall back to config defaults
        username = username_override or self.config.webhook_username
        avatar_url = avatar_override or self.config.webhook_avatar_url

        if username:
            payload["username"] = username

        if avatar_url:
            payload["avatar_url"] = avatar_url

        # Parse webhook_id and webhook_token from URL
        # Format: https://discord.com/api/webhooks/{webhook_id}/{webhook_token}
        import re
        webhook_match = re.search(r'/webhooks/(\d+)/([^/?]+)', webhook_url)
        webhook_id = webhook_match.group(1) if webhook_match else None
        webhook_token = webhook_match.group(2) if webhook_match else None

        try:
            # Use ?wait=true to get the message object back
            post_url = f"{webhook_url}?wait=true"

            # Use a separate session without auth header for webhook posts
            async with aiohttp.ClientSession() as webhook_session:
                async with webhook_session.post(
                    post_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        # Success with wait=true returns the message object
                        message_data = await response.json()
                        message_id = message_data.get("id")
                        logger.debug(f"Posted to Discord (msg_id={message_id}): {content[:50]}...")
                        return {
                            "message_id": message_id,
                            "webhook_id": webhook_id,
                            "webhook_token": webhook_token
                        }
                    elif response.status == 204:
                        # Success without wait (shouldn't happen with wait=true)
                        logger.debug(f"Posted to Discord: {content[:50]}...")
                        return {
                            "message_id": None,
                            "webhook_id": webhook_id,
                            "webhook_token": webhook_token
                        }
                    elif response.status == 429:
                        # Rate limited
                        retry_after = response.headers.get("Retry-After", "5")
                        logger.warning(f"Discord rate limited, retry after {retry_after}s")
                        return None
                    else:
                        error_text = await response.text()
                        logger.error(f"Discord webhook failed ({response.status}): {error_text}")
                        return None

        except asyncio.TimeoutError:
            logger.error("Discord webhook request timed out")
            return None
        except Exception as e:
            logger.error(f"Discord webhook error: {e}")
            return None

    async def edit_webhook_message(
        self,
        message_id: str,
        webhook_id: str,
        webhook_token: str,
        new_content: str
    ) -> bool:
        """
        Edit a previously posted webhook message.

        Args:
            message_id: The ID of the message to edit.
            webhook_id: The webhook ID.
            webhook_token: The webhook token.
            new_content: The new content for the message.

        Returns:
            True if successful.
        """
        if not message_id or not webhook_id or not webhook_token:
            logger.error("Missing required parameters for editing webhook message")
            return False

        edit_url = f"https://discord.com/api/webhooks/{webhook_id}/{webhook_token}/messages/{message_id}"
        payload = {"content": new_content}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.patch(
                    edit_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        logger.debug(f"Edited Discord message {message_id}")
                        return True
                    elif response.status == 429:
                        retry_after = response.headers.get("Retry-After", "5")
                        logger.warning(f"Discord rate limited, retry after {retry_after}s")
                        return False
                    else:
                        error_text = await response.text()
                        logger.error(f"Discord edit failed ({response.status}): {error_text}")
                        return False

        except asyncio.TimeoutError:
            logger.error("Discord edit request timed out")
            return False
        except Exception as e:
            logger.error(f"Discord edit error: {e}")
            return False

    def set_last_speaker(self, bot_name: str):
        """
        Set the last bot that spoke (for LAST_SPEAKER mode).

        Call this when receiving TTS from a bot to track who spoke last.

        Args:
            bot_name: Name of the bot that just spoke.
        """
        if bot_name in self.config.bots:
            self._last_bot_mentioned = bot_name

    async def fetch_channel_info(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch channel information from Discord API.

        Args:
            channel_id: Discord channel ID

        Returns:
            Channel info dict with 'name', 'id', 'guild_id' etc., or None if failed.
        """
        if not self._session:
            await self.start()

        try:
            url = f"{DISCORD_API_BASE}/channels/{channel_id}"
            async with self._session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "id": data.get("id"),
                        "name": data.get("name", f"Channel {channel_id}"),
                        "guild_id": data.get("guild_id"),
                        "type": data.get("type")
                    }
                else:
                    logger.warning(f"Failed to fetch channel info for {channel_id}: {response.status}")
        except Exception as e:
            logger.error(f"Error fetching channel info: {e}")
        return None

    async def fetch_channel_messages(
        self,
        channel_id: str,
        limit: int = 100,
        before: Optional[str] = None,
        after: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch recent messages from a Discord channel.

        Args:
            channel_id: Discord channel ID
            limit: Maximum number of messages to fetch (max 100)
            before: Message ID to fetch messages before (for pagination)
            after: Message ID to fetch messages after (for polling new messages)

        Returns:
            List of message dicts, most recent first.
        """
        if not self._session:
            await self.start()

        try:
            url = f"{DISCORD_API_BASE}/channels/{channel_id}/messages"
            params = {"limit": min(limit, 100)}
            if before:
                params["before"] = before
            if after:
                params["after"] = after

            async with self._session.get(url, params=params) as response:
                if response.status == 200:
                    messages = await response.json()
                    logger.info(f"Fetched {len(messages)} messages from channel {channel_id}")
                    return messages
                elif response.status == 403:
                    logger.warning(f"No permission to read messages in channel {channel_id}")
                else:
                    error = await response.text()
                    logger.warning(f"Failed to fetch messages: {response.status} {error}")
        except Exception as e:
            logger.error(f"Error fetching channel messages: {e}")
        return []


def discord_message_to_turn(msg: Dict[str, Any], bot_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Convert a Discord message to a conversation turn format.

    Args:
        msg: Discord message dict from API
        bot_ids: List of bot user IDs to identify as 'assistant' role

    Returns:
        Dict with 'role', 'content', 'speaker_name', 'timestamp', 'metadata'
    """
    author = msg.get("author", {})
    author_id = author.get("id", "")
    is_bot = author.get("bot", False)

    # Determine role based on whether author is a bot
    if bot_ids and author_id in bot_ids:
        role = "assistant"
    elif is_bot:
        role = "assistant"
    else:
        role = "user"

    # Get display name
    speaker_name = author.get("global_name") or author.get("username", "Unknown")

    # Parse timestamp
    timestamp_str = msg.get("timestamp", "")
    try:
        # Discord uses ISO 8601 format
        timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        timestamp = datetime.now()

    # Process content - convert mentions from <@user_id> to @username
    content = msg.get("content", "")
    mentions = msg.get("mentions", [])
    for mention in mentions:
        mention_id = mention.get("id", "")
        mention_name = mention.get("global_name") or mention.get("username", "")
        if mention_id and mention_name:
            # Replace <@user_id> and <@!user_id> (nickname mention) with @username
            content = content.replace(f"<@{mention_id}>", f"@{mention_name}")
            content = content.replace(f"<@!{mention_id}>", f"@{mention_name}")

    return {
        "role": role,
        "content": content,
        "speaker_name": speaker_name,
        "timestamp": timestamp,
        "metadata": {
            "discord_message_id": msg.get("id"),
            "discord_author_id": author_id,
            "is_bot": is_bot,
            "attachments": len(msg.get("attachments", [])) > 0,
            "embeds": len(msg.get("embeds", [])) > 0
        }
    }


# Convenience function for creating a configured poster
def create_discord_webhook(
    bot_token: str,
    bots: Dict[str, str],  # Simple name -> user_id mapping
    default_bot: Optional[str] = None,
    mention_mode: str = "default",
    webhook_username: str = "Melodeus"
) -> DiscordWebhookPoster:
    """
    Create a configured DiscordWebhookPoster.

    Args:
        bot_token: Discord bot token for API access.
        bots: Mapping of bot names to Discord user IDs.
        default_bot: Default bot to mention.
        mention_mode: One of "default", "explicit", "round_robin", "last_speaker".
        webhook_username: Username to display in Discord.

    Returns:
        Configured DiscordWebhookPoster instance.
    """
    bot_mappings = {
        name: BotMapping(name=name, user_id=user_id)
        for name, user_id in bots.items()
    }

    config = DiscordWebhookConfig(
        bot_token=bot_token,
        bots=bot_mappings,
        default_bot=default_bot,
        mention_mode=MentionMode(mention_mode),
        webhook_username=webhook_username
    )

    return DiscordWebhookPoster(config)
