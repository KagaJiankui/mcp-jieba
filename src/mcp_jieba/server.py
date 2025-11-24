import sys
import traceback
import os
from typing import List, Union, Callable
from functools import wraps
from mcp.server.fastmcp import FastMCP
from mcp_jieba.engine import JiebaEngine

# Initialize the FastMCP server
mcp = FastMCP("jieba-rs")

# Initialize JiebaEngine once globally
_engine = JiebaEngine()

# Exception handling decorator
def handle_exceptions(func: Callable) -> Callable:
    """Decorator to handle exceptions uniformly across all tools."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
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
    return wrapper

@mcp.tool()
@handle_exceptions
def tokenize(text: Union[str, List[str]], mode: str = "exact") -> dict:
    """
    Tokenize text(s) with jieba segmentation (exact or search mode).

    Args:
        text: `Union[str, List[str]]` A single string or a list of strings to tokenize.
        mode:  `str` Tokenization mode - "exact" for precise segmentation (default) or "search" for search engine mode.

    Returns:
        A dictionary where keys are indices (as strings) and values are lists of tokens.
        Example: {"0": ["token1", "token2", ...], "1": [...]}
    """
    results = _engine.process(text, mode)
    return results

@mcp.tool()
@handle_exceptions
def tag(text: Union[str, List[str]]) -> dict:
    """
    Perform POS tagging on text(s) with jieba.

    Args:
        text: `Union[str, List[str]]` A single string or a list of strings to tag.

    Returns:
        A dictionary where keys are indices (as strings) and values are lists of word-flag pairs.
        Example: {"0": [{"word1": "flag1"}, {"word2": "flag2"}, ...], "1": [...]}
    """
    results = _engine.tag(text)
    return results

@mcp.tool()
@handle_exceptions
def extract_keywords(text: Union[str, List[str]], top_k: int = 3) -> dict:
    """
    Extract keywords from text(s) using BM25-adpt algorithm with numpy.
    Each input string is treated as an independent corpus, split into sentences for analysis.

    Args:
        text: `Union[str, List[str]]` A single string or a list of strings to extract keywords from.
        top_k:  `int` Number of top keywords to extract per document (default 3).

    Returns:
        A dictionary where keys are indices (as strings) and values are lists of keywords.
        Example: {"0": ["keyword1", "keyword2", "keyword3", ...], "1": [...]}
    """
    results = _engine.extract_keywords_bm25(text, top_k)
    return results

def main():
    """Main entry point for the server."""
    import argparse

    parser = argparse.ArgumentParser(description="MCP Jieba Server")
    parser.add_argument("--transport", default="http",
                        choices=["stdio", "http"],
                        help="Transport protocol (stdio or http)")
    parser.add_argument("--host", default=None,
                        help="Host to bind to (HTTP only)")
    parser.add_argument("--port", type=int, default=None,
                        help="Port to bind to (HTTP only)")
    parser.add_argument("--log-level", default="CRITICAL",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set logging level")

    args = parser.parse_args()

    # Check for environment variable configuration
    bind_env = os.environ.get("MCP_JIEBA_BIND")
    if bind_env:
        try:
            env_host, env_port = bind_env.rsplit(":", 1)
            env_port = int(env_port)
            # Environment variables override defaults but not command line args
            if args.host is None:
                args.host = env_host
            if args.port is None:
                args.port = env_port
        except ValueError:
            print(f"Warning: Invalid MCP_JIEBA_BIND format: {bind_env}. Expected format: host:port")

    # Set defaults if not provided by args or env
    if args.host is None:
        args.host = "127.0.0.1"
    if args.port is None:
        args.port = 8000

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
            os.environ["FASTMCP_LOG_LEVEL"] = args.log_level
        mcp.run(transport="stdio")

if __name__ == "__main__":
    main()