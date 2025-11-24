import sys
import traceback
import threading
from typing import List, Union, Callable, Optional
from functools import wraps
from mcp.server.fastmcp import FastMCP
from mcp_jieba.engine import JiebaEngine

# Initialize the FastMCP server
mcp = FastMCP("mcp-jieba", dependencies=["rjieba", "numpy","mcp_jieba","threading","mcp"])

# 使用线程安全的懒加载
_engine: Optional[JiebaEngine] = None
_engine_lock = threading.Lock()

def get_engine() -> JiebaEngine:
    """线程安全的懒加载 JiebaEngine，确保只初始化一次"""
    global _engine
    if _engine is None:
        with _engine_lock:
            # 双重检查，避免锁竞争后的重复初始化
            if _engine is None:
                _engine = JiebaEngine()
    return _engine

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
        text: `Union[str, List[str]]` A single string or a list of strings `["str1", "str2", ...]` to tokenize.
        mode:  `str` Tokenization mode - "exact" for precise segmentation (default) or "search" for search engine mode.

    Returns:
        A dictionary where keys are indices (as strings) and values are lists of tokens.
        Example: {0: ["token1", "token2", ...], 1: [...]}
    """
    engine = get_engine()  # 懒加载获取实例
    results = engine.process(text, mode)
    return results

@mcp.tool()
@handle_exceptions
def tag(text: Union[str, List[str]]) -> dict:
    """
    Perform POS tagging on text(s) with jieba.

    Args:
        text: `Union[str, List[str]]` A single string or a list of strings `["str1", "str2", ...]` to tag.

    Returns:
        A dictionary where keys are indices (as strings) and values are dicts of word-flag pairs.
        Example: {0: {"word1": "flag1", "word2": "flag2", ...}, 1: {...}}

        The flags follow ICTCLAS POS tagging conventions:
        ```json
        {"a": "形容词", "b": "区别词", "c": "连词", "d": "副词", "e": "叹词", "g": "语素字", "h": "前接成分", "i": "习用语", "j": "简称", "k": "后接成分", "m": "数词", "n": "普通名词", "nd": "方位名词", "nh": "人名", "ni": "机构名", "nl": "处所名词", "ns": "地名", "nt": "时间词", "nz": "其他专名", "o": "拟声词", "p": "介词", "q": "量词", "r": "代词", "u": "助词", "v": "动词", "wp": "标点符号", "ws": "字符串", "x": "非语素字", "y": "语气词", "z": "状态词"}
        ```
    """
    engine = get_engine()  # 懒加载获取实例
    results = engine.tag(text)
    return results

@mcp.tool()
@handle_exceptions
def extract_keywords(text: Union[str, List[str]], top_k: int = 3) -> dict:
    """
    Extract keywords from text(s) using BM25-adpt algorithm with numpy.
    Each input string is treated as an independent corpus, split into sentences for analysis.

    Args:
        text: `Union[str, List[str]]` A single string or a list of strings `["str1", "str2", ...]` to extract keywords from.
        top_k:  `int` Number of top keywords to extract per document (default 3).

    Returns:
        A dictionary where keys are indices (as strings) and values are lists of keywords.
        Example: {0: ["keyword1", "keyword2", "keyword3", ...], 1: [...]}
    """
    engine = get_engine()  # 懒加载获取实例
    results = engine.extract_keywords_bm25(text, top_k)
    return results

def main():
    """Main entry point for the server."""
    import argparse

    parser = argparse.ArgumentParser(description="MCP Jieba Server")
    parser.add_argument("--transport", default="http",
                        choices=["stdio", "http"],
                        help="Transport protocol (stdio or http)")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Host to bind to (HTTP only)")
    parser.add_argument("--port", type=int, default=3001,
                        help="Port to bind to (HTTP only)")
    args = parser.parse_args()

    # 从环境变量 BIND_ADDR 获取 host 和 port（格式为 host:port）
    import os
    bind_addr = os.environ.get("BIND_ADDR")
    if bind_addr:
        try:
            host, port = bind_addr.split(":")
            args.host = host
            args.port = int(port)
        except ValueError:
            print("Invalid BIND_ADDR format. Expected 'host:port'", file=sys.stderr)
            sys.exit(1)

    if args.transport == "http":
        # HTTP 模式：使用 SSE 传输协议 (MCP 标准 HTTP 实现)
        # 创建新实例以应用 host/port 配置
        server = FastMCP("mcp-jieba", host=args.host, port=args.port)

        # 重新注册工具
        server.add_tool(tokenize)
        server.add_tool(tag)
        server.add_tool(extract_keywords)

        print(f"Starting MCP Jieba server over HTTP at {args.host}:{args.port}...", file=sys.stderr)
        server.run(transport="streamable-http", mount_path="/mcp")
    else:
        # STDIO 模式
        print("Starting MCP Jieba server over STDIO...", file=sys.stderr)
        mcp.run(transport="stdio")

if __name__ == "__main__":
    main()