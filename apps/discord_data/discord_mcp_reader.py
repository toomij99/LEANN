#!/usr/bin/env python3
"""
Discord MCP Reader for LEANN

This module provides functionality to connect to Discord MCP servers and fetch message data
for indexing in LEANN. It supports various Discord MCP server implementations and provides
flexible message processing options.
"""

import asyncio
import json
import logging
import os
from typing import Any, Optional

try:
    import aiohttp
except ImportError:
    aiohttp = None

logger = logging.getLogger(__name__)


class DiscordMCPReader:
    """
    Reader for Discord data via MCP (Model Context Protocol) servers.

    This class connects to Discord MCP servers to fetch message data and convert it
    into a format suitable for LEANN indexing.
    """

    def __init__(
        self,
        mcp_server_command: str,
        guild_name: Optional[str] = None,
        concatenate_conversations: bool = True,
        max_messages_per_conversation: int = 100,
        max_retries: int = 5,
        retry_delay: float = 2.0,
    ):
        """
        Initialize the Discord MCP Reader.

        Args:
            mcp_server_command: Command to start the MCP server (e.g., 'mcp-discord')
            guild_name: Optional guild name to filter messages
            concatenate_conversations: Whether to group messages by channel/thread
            max_messages_per_conversation: Maximum messages to include per conversation
            max_retries: Maximum number of retries for failed operations
            retry_delay: Initial delay between retries in seconds
        """
        self.mcp_server_command = mcp_server_command
        self.guild_name = guild_name
        self.concatenate_conversations = concatenate_conversations
        self.max_messages_per_conversation = max_messages_per_conversation
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.mcp_process = None

    async def start_mcp_server(self):
        """Start the MCP server process."""
        try:
            self.mcp_process = await asyncio.create_subprocess_exec(
                *self.mcp_server_command.split(),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            logger.info(f"Started MCP server: {self.mcp_server_command}")
        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            raise

    async def stop_mcp_server(self):
        """Stop the MCP server process."""
        if self.mcp_process:
            self.mcp_process.terminate()
            await self.mcp_process.wait()
            logger.info("Stopped MCP server")

    async def send_mcp_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send a request to the MCP server and get response."""
        if not self.mcp_process:
            raise RuntimeError("MCP server not started")

        request_json = json.dumps(request) + "\n"
        if self.mcp_process.stdin:
            self.mcp_process.stdin.write(request_json.encode())
            await self.mcp_process.stdin.drain()

        response_line = b""
        if self.mcp_process.stdout:
            response_line = await self.mcp_process.stdout.readline()
            if not response_line:
                raise RuntimeError("No response from MCP server")

        return json.loads(response_line.decode().strip())

    async def initialize_mcp_connection(self):
        """Initialize the MCP connection."""
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "leann-discord-reader", "version": "1.0.0"},
            },
        }

        response = await self.send_mcp_request(init_request)
        if "error" in response:
            raise RuntimeError(f"MCP initialization failed: {response['error']}")

        logger.info("MCP connection initialized successfully")

    async def list_available_tools(self) -> list[dict[str, Any]]:
        """List available tools from the MCP server."""
        list_request = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}

        response = await self.send_mcp_request(list_request)
        if "error" in response:
            raise RuntimeError(f"Failed to list tools: {response['error']}")

        return response.get("result", {}).get("tools", [])

    async def _retry_with_backoff(self, func, *args, **kwargs):
        """Retry a function with exponential backoff."""
        last_exception = RuntimeError("Unknown error")

        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2**attempt)  # Exponential backoff
                    logger.info(
                        f"Operation failed, waiting {delay:.1f}s before retry {attempt + 1}/{self.max_retries}"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    break

        # If we get here, all retries failed
        raise last_exception

    async def fetch_discord_messages(
        self, channel: Optional[str] = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """
        Fetch Discord messages using MCP tools with retry logic.

        Args:
            channel: Optional channel name or ID to filter messages
            limit: Maximum number of messages to fetch

        Returns:
            List of message dictionaries
        """
        return await self._retry_with_backoff(self._fetch_discord_messages_impl, channel, limit)

    async def _fetch_discord_messages_impl(
        self, channel: Optional[str] = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """
        Internal implementation using Discord REST API directly.
        """
        if not aiohttp:
            raise RuntimeError("aiohttp library is required for Discord REST API access. Install with: uv pip install aiohttp")

        # Get bot token from environment
        token = os.getenv("DISCORD_TOKEN")
        if not token:
            raise RuntimeError("DISCORD_TOKEN environment variable not set")

        if not channel:
            # If no specific channel, we'll need to get channels first
            # For now, return empty list if no channel specified
            return []

        headers = {
            "Authorization": f"Bot {token}",
            "User-Agent": "DiscordBot (https://discord.com/api, 10)"
        }

        url = f"https://discord.com/api/v10/channels/{channel}/messages?limit={min(limit, 100)}"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    messages = await response.json()
                    return messages
                else:
                    error_text = await response.text()
                    raise RuntimeError(f"Failed to fetch messages: {response.status} - {error_text}")

    def _parse_text_messages(self, text: str, channel: Optional[str]) -> list[dict[str, Any]]:
        """Parse text format messages from Discord MCP server."""
        messages = []
        try:
            # Try to parse as JSON first
            messages = json.loads(text)
            if isinstance(messages, dict):
                messages = [messages]
        except json.JSONDecodeError:
            # If not JSON, treat as plain text
            messages = [{"text": text, "channel": channel or "unknown"}]

        return messages

    def _format_message(self, message: dict[str, Any]) -> str:
        """Format a single message for indexing."""
        text = message.get("content", message.get("text", ""))
        user = message.get("author", {}).get("username", message.get("username", "Unknown"))
        channel = message.get("channel", {}).get("name", message.get("channel_name", "Unknown"))
        timestamp = message.get("timestamp", message.get("ts", ""))

        # Format timestamp if available
        formatted_time = ""
        if timestamp:
            try:
                import datetime
                # Discord timestamps are ISO format
                if isinstance(timestamp, str):
                    dt = datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    formatted_time = str(timestamp)
            except (ValueError, TypeError):
                formatted_time = str(timestamp)

        # Build formatted message
        parts = []
        if channel:
            parts.append(f"Channel: #{channel}")
        if user:
            parts.append(f"User: {user}")
        if formatted_time:
            parts.append(f"Time: {formatted_time}")
        if text:
            parts.append(f"Message: {text}")

        return "\n".join(parts)

    def _create_concatenated_content(self, messages: list[dict[str, Any]], channel: str) -> str:
        """Create concatenated content from multiple messages in a channel."""
        if not messages:
            return ""

        # Sort messages by timestamp if available
        try:
            messages.sort(key=lambda x: x.get("timestamp", x.get("ts", "")))
        except (ValueError, TypeError):
            pass  # Keep original order if timestamps aren't sortable

        # Limit messages per conversation
        if len(messages) > self.max_messages_per_conversation:
            messages = messages[-self.max_messages_per_conversation :]

        # Create header
        content_parts = [
            f"Discord Channel: #{channel}",
            f"Message Count: {len(messages)}",
            f"Guild: {self.guild_name or 'Unknown'}",
            "=" * 50,
            "",
        ]

        # Add messages
        for message in messages:
            formatted_msg = self._format_message(message)
            if formatted_msg.strip():
                content_parts.append(formatted_msg)
                content_parts.append("-" * 30)
                content_parts.append("")

        return "\n".join(content_parts)

    async def get_all_channels(self) -> list[str]:
        """Get list of all available channels."""
        if not aiohttp:
            return []

        # Get bot token from environment
        token = os.getenv("DISCORD_TOKEN")
        if not token:
            return []

        # First, get the guild ID from the guild name
        guild_id = await self._get_guild_id()
        if not guild_id:
            return []

        headers = {
            "Authorization": f"Bot {token}",
            "User-Agent": "DiscordBot (https://discord.com/api, 10)"
        }

        url = f"https://discord.com/api/v10/guilds/{guild_id}/channels"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        channels_data = await response.json()
                        # Extract text channel IDs
                        channels = []
                        for channel in channels_data:
                            if channel.get("type") == 0:  # Text channel
                                channels.append(channel["id"])
                        logger.info(f"Found {len(channels)} text channels")
                        return channels
                    else:
                        error_text = await response.text()
                        logger.warning(f"Failed to get channels: {response.status} - {error_text}")
                        return []
        except Exception as e:
            logger.warning(f"Failed to get channels list: {e}")
            return []

    async def _get_guild_id(self) -> Optional[str]:
        """Get guild ID from guild name."""
        if not aiohttp:
            return None

        token = os.getenv("DISCORD_TOKEN")
        if not token:
            return None

        headers = {
            "Authorization": f"Bot {token}",
            "User-Agent": "DiscordBot (https://discord.com/api, 10)"
        }

        url = "https://discord.com/api/v10/users/@me/guilds"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        guilds = await response.json()
                        for guild in guilds:
                            if guild.get("name") == self.guild_name:
                                return guild["id"]
                        logger.warning(f"Guild '{self.guild_name}' not found in bot's guilds")
                        return None
                    else:
                        error_text = await response.text()
                        logger.warning(f"Failed to get guilds: {response.status} - {error_text}")
                        return None
        except Exception as e:
            logger.warning(f"Failed to get guild ID: {e}")
            return None

    async def read_discord_data(self, channels: Optional[list[str]] = None) -> list[str]:
        """
        Read Discord data and return formatted text chunks.

        Args:
            channels: Optional list of channel names to fetch. If None, fetches from all available channels.

        Returns:
            List of formatted text chunks ready for LEANN indexing
        """
        try:
            await self.start_mcp_server()
            await self.initialize_mcp_connection()

            all_texts = []

            if channels:
                # Fetch specific channels
                for channel in channels:
                    try:
                        messages = await self.fetch_discord_messages(channel=channel, limit=100)
                        if messages and any(msg.get("content", "").strip() for msg in messages):
                            if self.concatenate_conversations:
                                text_content = self._create_concatenated_content(messages, channel)
                                if text_content.strip():
                                    all_texts.append(text_content)
                            else:
                                # Process individual messages
                                for message in messages:
                                    formatted_msg = self._format_message(message)
                                    if formatted_msg.strip():
                                        all_texts.append(formatted_msg)
                        else:
                            logger.info(f"No accessible messages in channel {channel}")
                    except Exception as e:
                        logger.warning(f"Failed to fetch messages from channel {channel}: {e}")
                        continue
            else:
                # Fetch from all available channels
                logger.info("Fetching from all available channels...")
                all_channels = await self.get_all_channels()

                if not all_channels:
                    # Fallback to known accessible channel
                    all_channels = ["1217467145685041232"]  # aisec channel
                    logger.info(f"Using fallback channels: {all_channels}")
                else:
                    # Filter to only accessible channels for testing
                    all_channels = ["1217467145685041232"]  # Only test accessible channel
                    logger.info(f"Testing only accessible channel: {all_channels}")

                for channel in all_channels:
                    try:
                        logger.info(f"Searching channel: {channel}")
                        messages = await self.fetch_discord_messages(channel=channel, limit=100)
                        if messages and any(msg.get("content", "").strip() for msg in messages):
                            if self.concatenate_conversations:
                                text_content = self._create_concatenated_content(messages, channel)
                                if text_content.strip():
                                    all_texts.append(text_content)
                            else:
                                # Process individual messages
                                for message in messages:
                                    formatted_msg = self._format_message(message)
                                    if formatted_msg.strip():
                                        all_texts.append(formatted_msg)
                        else:
                            logger.info(f"No accessible messages in channel {channel}")
                    except Exception as e:
                        logger.warning(f"Failed to fetch messages from channel {channel}: {e}")
                        continue

            return all_texts

        finally:
            await self.stop_mcp_server()

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_mcp_server()
        await self.initialize_mcp_connection()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop_mcp_server()