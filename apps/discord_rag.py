#!/usr/bin/env python3
"""
Discord RAG Application with MCP Support

This application enables RAG (Retrieval-Augmented Generation) on Discord messages
by connecting to Discord MCP servers to fetch live data and index it in LEANN.

Usage:
    python -m apps.discord_rag --mcp-server "mcp-discord" --guild-name "my-guild"
"""

import argparse
import asyncio

from apps.base_rag_example import BaseRAGExample
from apps.discord_data.discord_mcp_reader import DiscordMCPReader


class DiscordMCPRAG(BaseRAGExample):
    """
    RAG application for Discord messages via MCP servers.

    This class provides a complete RAG pipeline for Discord data, including
    MCP server connection, data fetching, indexing, and interactive chat.
    """

    def __init__(self):
        super().__init__(
            name="Discord MCP RAG",
            description="RAG application for Discord messages via MCP servers",
            default_index_name="discord_messages",
        )

    def _add_specific_arguments(self, parser: argparse.ArgumentParser):
        """Add Discord MCP-specific arguments."""
        parser.add_argument(
            "--mcp-server",
            type=str,
            required=True,
            help="Command to start the Discord MCP server (e.g., 'mcp-discord')",
        )

        parser.add_argument(
            "--guild-name",
            type=str,
            help="Discord guild/server name for better organization and filtering",
        )

        parser.add_argument(
            "--channels",
            nargs="+",
            help="Specific Discord channels to index (e.g., general random). If not specified, fetches from all available channels",
        )

        parser.add_argument(
            "--concatenate-conversations",
            action="store_true",
            default=True,
            help="Group messages by channel/thread for better context (default: True)",
        )

        parser.add_argument(
            "--no-concatenate-conversations",
            action="store_true",
            help="Process individual messages instead of grouping by channel",
        )

        parser.add_argument(
            "--max-messages-per-channel",
            type=int,
            default=100,
            help="Maximum number of messages to include per channel (default: 100)",
        )

        parser.add_argument(
            "--test-connection",
            action="store_true",
            help="Test MCP server connection and list available tools without indexing",
        )

        parser.add_argument(
            "--max-retries",
            type=int,
            default=5,
            help="Maximum number of retries for failed operations (default: 5)",
        )

        parser.add_argument(
            "--retry-delay",
            type=float,
            default=2.0,
            help="Initial delay between retries in seconds (default: 2.0)",
        )

    async def test_mcp_connection(self, args) -> bool:
        """Test the MCP server connection and display available tools."""
        print(f"Testing connection to MCP server: {args.mcp_server}")

        try:
            reader = DiscordMCPReader(
                mcp_server_command=args.mcp_server,
                guild_name=args.guild_name,
                concatenate_conversations=not args.no_concatenate_conversations,
                max_messages_per_conversation=args.max_messages_per_channel,
                max_retries=args.max_retries,
                retry_delay=args.retry_delay,
            )

            async with reader:
                tools = await reader.list_available_tools()

                print("Successfully connected to MCP server!")
                print(f"Available tools ({len(tools)}):")

                for i, tool in enumerate(tools, 1):
                    name = tool.get("name", "Unknown")
                    description = tool.get("description", "No description available")
                    print(f"\n{i}. {name}")
                    print(
                        f"   Description: {description[:100]}{'...' if len(description) > 100 else ''}"
                    )

                    # Show input schema if available
                    schema = tool.get("inputSchema", {})
                    if schema.get("properties"):
                        props = list(schema["properties"].keys())[:3]  # Show first 3 properties
                        print(
                            f"   Parameters: {', '.join(props)}{'...' if len(schema['properties']) > 3 else ''}"
                        )

                return True

        except Exception as e:
            print(f"Failed to connect to MCP server: {e}")
            print("\nTroubleshooting tips:")
            print("1. Make sure the MCP server is installed and accessible")
            print("2. Check if the server command is correct")
            print("3. Ensure you have proper authentication/credentials configured")
            print("4. Try running the MCP server command directly to test it")
            return False

    async def load_data(self, args) -> list[str]:
        """Load Discord messages via MCP server."""
        print(f"Connecting to Discord MCP server: {args.mcp_server}")

        if args.guild_name:
            print(f"Guild: {args.guild_name}")

        # Filter out empty strings from channels
        channels = [ch for ch in args.channels if ch.strip()] if args.channels else None

        if channels:
            print(f"Channels: {', '.join(channels)}")
        else:
            print("Fetching from all available channels")

        concatenate = not args.no_concatenate_conversations
        print(
            f"Processing mode: {'Concatenated conversations' if concatenate else 'Individual messages'}"
        )

        try:
            reader = DiscordMCPReader(
                mcp_server_command=args.mcp_server,
                guild_name=args.guild_name,
                concatenate_conversations=concatenate,
                max_messages_per_conversation=args.max_messages_per_channel,
                max_retries=args.max_retries,
                retry_delay=args.retry_delay,
            )

            texts = await reader.read_discord_data(channels=channels)

            if not texts:
                print("No messages found! This could mean:")
                print("- The MCP server couldn't fetch messages")
                print("- The specified channels don't exist or are empty")
                print("- Authentication issues with the Discord bot/token")
                return []

            print(f"Successfully loaded {len(texts)} text chunks from Discord")

            # Show sample of what was loaded
            if texts:
                sample_text = texts[0][:200] + "..." if len(texts[0]) > 200 else texts[0]
                print("\nSample content:")
                print("-" * 40)
                print(sample_text)
                print("-" * 40)

            return texts

        except Exception as e:
            print(f"Error loading Discord data: {e}")
            print("\nThis might be due to:")
            print("- MCP server connection issues")
            print("- Authentication problems")
            print("- Network connectivity issues")
            print("- Incorrect channel names")
            raise

    async def run(self):
        """Main entry point with MCP connection testing."""
        args = self.parser.parse_args()

        # Test connection if requested
        if args.test_connection:
            success = await self.test_mcp_connection(args)
            if not success:
                return
            print(
                "MCP server is working! You can now run without --test-connection to start indexing."
            )
            return

        # Run the standard RAG pipeline
        await super().run()


async def main():
    """Main entry point for the Discord MCP RAG application."""
    app = DiscordMCPRAG()
    await app.run()


if __name__ == "__main__":
    asyncio.run(main())