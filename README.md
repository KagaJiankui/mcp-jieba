# MCP Jieba Server

这是一个基于 [rjieba](https://github.com/messense/rjieba) (Rust implementation of Jieba) 的 Model Context Protocol (MCP) 服务器，提供高性能的中文分词服务。

## 功能特性

- **高性能分词**: 使用 Rust 编写的底层引擎。
- **多模式支持**: 支持精确模式 (`exact`) 和搜索引擎模式 (`search`)。
- **自定义词典**: 支持动态传入术语表。
- **停用词过滤**: 内置常用停用词表，自动过滤停用词和标点。
- **批量处理**: 支持单字符串或字符串数组输入，返回 JSON 格式结果。
- **双模部署**:
  - **STDIO**: 适用于本地开发和 Cursor/VS Code 集成。
  - **SSE**: 适用于远程部署（如 ModelScope）。

## 安装

### 使用 pip / uv / pipx

```bash
# 使用 pip
pip install .

# 使用 uv
uv pip install .
```

## 使用方法

### 1. 本地运行 (STDIO)

直接运行模块即可启动 STDIO 服务器：

```bash
python -m mcp_jieba.server
```

或者在 Cursor/VS Code 的 MCP 配置中添加：

```json
{
  "mcpServers": {
    "jieba": {
      "command": "python",
      "args": ["-m", "mcp_jieba.server"]
    }
  }
}
```

### 2. 远程部署 (SSE)

使用 `uvicorn` 启动 SSE 服务器：

```bash
uvicorn mcp_jieba.server:mcp --host 0.0.0.0 --port 8000
```

SSE 端点地址: `http://localhost:8000/sse`

### ModelScope 部署

在 ModelScope 创建 Space 时，选择 Python 环境，并使用以下启动命令：

```bash
uvicorn mcp_jieba.server:mcp --host 0.0.0.0 --port 8000
```

确保 `requirements.txt` 或 `pyproject.toml` 包含所有依赖。

## 工具说明

### `tokenize`

对文本进行分词。

**参数**:

- `text` (required): 待分词的文本，可以是单个字符串或字符串数组。
- `mode` (optional): 分词模式，可选 `"exact"` (默认) 或 `"search"`。
- `user_dict` (optional): 自定义术语表数组，例如 `["新词1", "新词2"]`。

**返回**:
JSON 对象，键为输入数组的索引（字符串格式），值为分词结果数组。

示例:
输入: `text=["我爱北京天安门"], mode="exact"`
输出: `{"0": ["我", "爱", "北京", "天安门"]}`
