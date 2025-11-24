import sys
import traceback
from typing import List, Union, Optional
from mcp.server.fastmcp import FastMCP
from .engine import JiebaEngine

# Initialize the FastMCP server
mcp = FastMCP("jieba-rs")

@mcp.tool()
def tokenize(text: Union[str, List[str]], mode: str = "exact", user_dict: Optional[List[str]] = None) -> dict:
    """
    Tokenize text using jieba-rs.

    Args:
        text: The text to tokenize. Can be a single string or a list of strings.
        mode: The tokenization mode. "exact" (default) or "search".
        user_dict: Optional list of user dictionary terms to improve segmentation.

    Returns:
        A JSON object representing the tokenization results.
        Format: {"0": ["token1", "token2"], "1": [...]}
    """
    try:
        engine = JiebaEngine()
        # Ensure user_dict is passed correctly (it can be None)
        results = engine.process(text, mode, user_dict)
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

if __name__ == "__main__":
    # Default to STDIO mode when run directly
    mcp.run()
