"""Test Streamable HTTP endpoint for MCP Jieba server."""
import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"
MCP_ENDPOINT = f"{BASE_URL}/mcp"

def test_streamable_http():
    """Test the Streamable HTTP protocol flow."""

    # Step 1: Send initialize request
    print("Step 1: Sending initialize request...")
    initialize_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        }
    }

    response = requests.post(
        MCP_ENDPOINT,
        json=initialize_request,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }
    )

    if response.status_code != 200:
        print(f"❌ Initialize failed: {response.status_code}")
        print(f"Response: {response.text}")
        return False

    # Check if session ID is returned (stateful mode)
    session_id = response.headers.get("Mcp-Session-Id")
    if session_id:
        print(f"✓ Session established: {session_id}")
    else:
        print("✓ Stateless mode (no session ID)")

    result = response.json()
    print(f"✓ Initialize response: {json.dumps(result, indent=2, ensure_ascii=False)}")

    # Step 2: List tools
    print("\nStep 2: Listing tools...")
    list_tools_request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {}
    }

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"
    }
    if session_id:
        headers["Mcp-Session-Id"] = session_id

    response = requests.post(
        MCP_ENDPOINT,
        json=list_tools_request,
        headers=headers
    )

    if response.status_code != 200:
        print(f"❌ List tools failed: {response.status_code}")
        print(f"Response: {response.text}")
        return False

    result = response.json()
    print(f"✓ Tools list response: {json.dumps(result, indent=2, ensure_ascii=False)}")

    # Verify tools are present
    if "result" in result and "tools" in result["result"]:
        tools = result["result"]["tools"]
        print(f"\n✓ Found {len(tools)} tools:")
        for tool in tools:
            print(f"  - {tool['name']}: {tool.get('description', 'No description')}")
        return True
    else:
        print("❌ No tools found in response")
        return False

if __name__ == "__main__":
    print("Testing MCP Jieba Streamable HTTP Endpoint")
    print("=" * 50)

    # Wait a moment to ensure server is ready
    print("Waiting for server to be ready...")
    time.sleep(2)

    try:
        success = test_streamable_http()
        if success:
            print("\n" + "=" * 50)
            print("✓ All HTTP tests passed!")
        else:
            print("\n" + "=" * 50)
            print("❌ HTTP tests failed")
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to server. Is it running on {MCP_ENDPOINT}?")
    # except Exception as e:
    #     print(f"❌ Error during testing: {e}")
