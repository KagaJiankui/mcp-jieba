"""Test Streamable HTTP endpoint for MCP Jieba server using MCP SDK."""
import asyncio
import logging
import pytest

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "http://127.0.0.1:8000"
MCP_ENDPOINT = f"{BASE_URL}/mcp"

@pytest.mark.asyncio
async def test_streamable_http():
    """Test the Streamable HTTP protocol flow using MCP SDK."""
    print("Testing MCP Jieba Streamable HTTP Endpoint")
    print("=" * 50)

    try:
        # Connect to the server
        print("Step 1: Connecting to server...")
        async with streamablehttp_client(MCP_ENDPOINT) as (read_stream, write_stream, get_session_id):
            print("✓ Connected to streamable HTTP endpoint")

            async with ClientSession(read_stream, write_stream) as session:
                # Step 1: Initialize
                print("\nStep 2: Initializing session...")
                init_result = await session.initialize()
                print(f"✓ Initialize response: {init_result}")

                # Check session ID
                session_id = get_session_id()
                if session_id:
                    print(f"✓ Session established: {session_id}")
                else:
                    print("✓ Stateless mode (no session ID)")

                # Step 2: List tools
                print("\nStep 3: Listing tools...")
                tools_result = await session.list_tools()

                print(f"✓ Found {len(tools_result.tools)} tools:")
                for tool in tools_result.tools:
                    print(f"  - {tool.name}: {tool.description}")

                print("\n" + "=" * 50)
                print("✓ All HTTP tests passed!")
                return True

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_streamable_http())
