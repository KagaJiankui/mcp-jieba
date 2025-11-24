"""Test all transport modes for MCP Jieba server."""
import subprocess
import time
import requests
import json
import sys

def test_stdio():
    """Test STDIO transport."""
    print("\n" + "="*60)
    print("Testing STDIO Transport")
    print("="*60)

    test_input = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test", "version": "1.0"}}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    ]

    input_data = "\n".join(json.dumps(req) for req in test_input)

    try:
        result = subprocess.run(
            ["python", "-m", "mcp_jieba"],
            input=input_data,
            capture_output=True,
            text=True,
            env={"PYTHONPATH": "src"},
            timeout=5
        )

        # Parse responses
        lines = [line for line in result.stdout.strip().split('\n') if line]

        if len(lines) >= 2:
            init_response = json.loads(lines[0])
            tools_response = json.loads(lines[1])

            print(f"✓ Initialize: {init_response.get('result', {}).get('serverInfo', {}).get('name', 'Unknown')}")

            tools = tools_response.get('result', {}).get('tools', [])
            print(f"✓ Found {len(tools)} tools:")
            for tool in tools:
                print(f"  - {tool['name']}")

            return True
        else:
            print(f"❌ Unexpected output: {result.stdout}")
            return False

    except Exception as e:
        print(f"❌ STDIO test failed: {e}")
        return False

def test_http():
    """Test HTTP transport."""
    print("\n" + "="*60)
    print("Testing Streamable HTTP Transport")
    print("="*60)

    # Start server in background
    print("Starting HTTP server...")
    server = subprocess.Popen(
        ["python", "-m", "mcp_jieba", "--transport", "http"],
        env={"PYTHONPATH": "src"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    time.sleep(3)  # Wait for server to start

    try:
        # Test initialize
        init_response = requests.post(
            "http://127.0.0.1:8000/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1.0"}
                }
            },
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            },
            timeout=5
        )

        if init_response.status_code == 200:
            result = init_response.json()
            server_name = result.get('result', {}).get('serverInfo', {}).get('name', 'Unknown')
            print(f"✓ Initialize: {server_name}")
        else:
            print(f"❌ Initialize failed: {init_response.status_code}")
            return False

        # Test tools/list
        session_id = init_response.headers.get("Mcp-Session-Id")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }
        if session_id:
            headers["Mcp-Session-Id"] = session_id
            print(f"✓ Session ID: {session_id}")

        tools_response = requests.post(
            "http://127.0.0.1:8000/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {}
            },
            headers=headers,
            timeout=5
        )

        if tools_response.status_code == 200:
            result = tools_response.json()
            tools = result.get('result', {}).get('tools', [])
            print(f"✓ Found {len(tools)} tools:")
            for tool in tools:
                print(f"  - {tool['name']}")
            return True
        else:
            print(f"❌ Tools/list failed: {tools_response.status_code}")
            return False

    except Exception as e:
        print(f"❌ HTTP test failed: {e}")
        return False
    finally:
        server.terminate()
        server.wait(timeout=2)
        print("✓ Server stopped")

if __name__ == "__main__":
    print("MCP Jieba Server - Complete Test Suite")
    print("="*60)

    stdio_ok = test_stdio()
    http_ok = test_http()

    print("\n" + "="*60)
    print("Test Results:")
    print(f"  STDIO: {'✓ PASS' if stdio_ok else '✗ FAIL'}")
    print(f"  HTTP:  {'✓ PASS' if http_ok else '✗ FAIL'}")
    print("="*60)

    if stdio_ok and http_ok:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed")
        sys.exit(1)
