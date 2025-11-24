import sys
import traceback
from typing import List, Union
from mcp.server.fastmcp import FastMCP
from mcp_jieba.engine import JiebaEngine

# Initialize the FastMCP server
mcp = FastMCP("jieba-rs")

@mcp.tool()
def tokenize(text: Union[str, List[str]], mode: str = "exact") -> dict:
    """
    Tokenize text using jieba-rs.

    Args:
        text: The text to tokenize. Can be a single string or a list of strings.
        mode: The tokenization mode. "exact" (default) or "search".

    Returns:
        A JSON object representing the tokenization results.
        Format: {"0": ["token1", "token2"], "1": [...]}
    """
    try:
        engine = JiebaEngine()
        results = engine.process(text, mode)
        return results
    except Exception as e:
        # Capture the exception and format the error
        exc_type, exc_value, exc_traceback = sys.exc_info()

        # Get the stack trace
        tb = traceback.extract_tb(exc_traceback)

        # Find the "bottom" of the stack (the actual error location)
        if tb:
            last_frame = tb[-1]
            filename = last_frame.filename
            lineno = last_frame.lineno
            funcname = last_frame.name
            error_location = f"File '{filename}', line {lineno}, in {funcname}"
        else:
            error_location = "Unknown location"

        type_name = exc_type.__name__ if exc_type else "UnknownError"
        error_msg = f"Error Type: {type_name}\nLocation: {error_location}\nMessage: {str(e)}"

        raise RuntimeError(error_msg)

@mcp.tool()
def tag(text: Union[str, List[str]]) -> dict:
    """
    Perform POS tagging on text using jieba-rs.

    Args:
        text: The text to tag. Can be a single string or a list of strings.

    Returns:
        A JSON object representing the tagging results.
        Format: {"0": [{"word": "word", "flag": "pos"}], "1": [...]}
    """
    try:
        engine = JiebaEngine()
        results = engine.tag(text)
        return results
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb = traceback.extract_tb(exc_traceback)
        if tb:
            last_frame = tb[-1]
            filename = last_frame.filename
            lineno = last_frame.lineno
            funcname = last_frame.name
            error_location = f"File '{filename}', line {lineno}, in {funcname}"
        else:
            error_location = "Unknown location"
        type_name = exc_type.__name__ if exc_type else "UnknownError"
        error_msg = f"Error Type: {type_name}\nLocation: {error_location}\nMessage: {str(e)}"
        raise RuntimeError(error_msg)

@mcp.tool()
def extract_keywords(text: List[str], top_k: int = 5) -> dict:
    """
    Extract keywords from a batch of texts using BM25 algorithm.
    The input batch is treated as the corpus for IDF calculation.

    Args:
        text: A list of strings (documents).
        top_k: Number of keywords to extract per document (default 5).

    Returns:
        A JSON object representing the keywords.
        Format: {"0": ["keyword1", "keyword2"], "1": [...]}
    """
    try:
        engine = JiebaEngine()
        # Ensure input is a list for corpus processing
        if isinstance(text, str):
            text = [text]

        results = engine.extract_keywords_bm25(text, top_k)
        return results
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb = traceback.extract_tb(exc_traceback)
        if tb:
            last_frame = tb[-1]
            filename = last_frame.filename
            lineno = last_frame.lineno
            funcname = last_frame.name
            error_location = f"File '{filename}', line {lineno}, in {funcname}"
        else:
            error_location = "Unknown location"
        type_name = exc_type.__name__ if exc_type else "UnknownError"
        error_msg = f"Error Type: {type_name}\nLocation: {error_location}\nMessage: {str(e)}"
        raise RuntimeError(error_msg)

def main():
    """Main entry point for the server."""
    import argparse

    parser = argparse.ArgumentParser(description="MCP Jieba Server")
    parser.add_argument("--transport", default="stdio",
                        choices=["stdio", "http"],
                        help="Transport protocol (stdio or http)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Host to bind to (HTTP only)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to bind to (HTTP only)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set logging level")

    args = parser.parse_args()

    # Configure server settings before running
    if args.transport == "http":
        # Create new server instance with HTTP-specific settings
        # Use streamable-http transport with stateless mode for better compatibility
        server = FastMCP(
            "jieba-rs",
            host=args.host,
            port=args.port,
            log_level=args.log_level,
            stateless_http=True  # Stateless mode for maximum compatibility
        )
        # Re-register tools on new instance
        server.add_tool(tokenize)
        server.add_tool(tag)
        server.add_tool(extract_keywords)
        # Use streamable-http transport (mounts at /mcp by default)
        server.run(transport="streamable-http")
    else:
        # Use default instance with custom log level
        if args.log_level != "INFO":
            import os
            os.environ["FASTMCP_LOG_LEVEL"] = args.log_level
        mcp.run(transport="stdio")

if __name__ == "__main__":
    main()