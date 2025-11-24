import asyncio
import os
import sys
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configuration
PYTHON_EXE = sys.executable
SERVER_SCRIPT = os.path.join("src", "mcp_jieba", "server.py")
LOREM_FILE = os.path.join("tests", "lorem.txt")
OUTPUT_FILE = os.path.join("tests", "stdio_results.txt")

async def main():
    print(f"Reading {LOREM_FILE}...")
    with open(LOREM_FILE, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    results = []

    # Initialize STDIO Server ONCE
    print("Starting STDIO Server Session...")
    server_params = StdioServerParameters(
        command=PYTHON_EXE,
        args=[SERVER_SCRIPT, "--transport", "stdio"],
        env=os.environ.copy()
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as stdio_session:
            await stdio_session.initialize()

            # Test first 5 lines
            for i, line in enumerate(lines):
                print(f"Processing line {i+1}/{len(lines)}: {line[:20]}...")

                # 1. Single string - STDIO
                try:
                    # Set timeout to detect blocking
                    result = await asyncio.wait_for(
                        stdio_session.call_tool("tokenize", arguments={"text": line}),
                        timeout=10.0
                    )
                    stdio_single_json = json.loads(result.content[0].text)
                except Exception as e:
                    print(f"Error on line {i+1}: {e}")
                    stdio_single_json = f"Error: {str(e)}"

                result_entry = {
                    "line_id": i,
                    "text": line,
                    "stdio_single": stdio_single_json,
                }
                results.append(result_entry)

    print(f"Writing results to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Done.")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    asyncio.run(main())
