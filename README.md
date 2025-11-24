# MCP Jieba Server

这是一个基于 [rjieba](https://github.com/messense/rjieba) (Rust implementation of Jieba) 的 Model Context Protocol (MCP) 服务器，提供高性能的中文分词服务。

## 功能特性

- **高性能分词**: 使用 Rust 编写的底层引擎。
- **多模式支持**: 支持精确模式 (`exact`) 和搜索引擎模式 (`search`)。
- **词性标注**: 支持 ICTCLAS 兼容的词性标注。
- **关键词提取**: 基于 BM25 算法的关键词提取。
- **批量处理**: 支持单字符串或字符串数组输入，返回 JSON 格式结果。
- **双模部署**:
  - **STDIO**: 适用于本地开发和 Claude Context/Cherry Studio/VS Code 集成。
  - **Streamable-HTTP**: 适用于远程部署（如 ModelScope）。

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

或者在 Claude Context/Cherry Studio/VS Code 的 MCP 配置中添加：

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

### 2. 远程部署 (Streamable-HTTP)

使用命令行参数启动 HTTP 服务器：

```bash
python -m mcp_jieba.server --transport http --host 0.0.0.0 --port 8000
```

SSE 端点地址: `http://localhost:8000/sse`

### ModelScope 部署

在 ModelScope 创建 Space 时，选择 Python 环境，并使用以下启动命令：

```bash
python -m mcp_jieba.server --transport http --host 0.0.0.0 --port 8000
```

当前 `pyproject.toml` 已经包含所有依赖。

## 开发与测试

目前项目的单元测试尚不完善。建议使用 MCP Inspector 进行交互式测试和调试。

```bash
bunx @modelcontextprotocol/inspector python -m mcp_jieba.server
```

## 工具说明

### `tokenize`

对文本进行分词。

|   项目   | 描述                                                                                                                                                         |
| :------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **参数** | <ul><li>`text` (required): 待分词的文本，可以是单个字符串或字符串数组。</li><li>`mode` (optional): 分词模式，可选 `"exact"` (默认) 或 `"search"`。</li></ul> |
| **返回** | JSON 对象，键为输入数组的索引（字符串格式），值为分词结果数组。                                                                                              |

示例:

- 输入: `text=["我爱北京天安门"], mode="exact"`
- 输出: `{"0": ["我", "爱", "北京", "天安门"]}`

### `tag`

对文本进行词性标注，标注类型符合ICTCLAS标准。

|   项目   | 描述                                                                              |
| :------: | --------------------------------------------------------------------------------- |
| **参数** | <ul><li>`text` (required): 待标注的文本，可以是单个字符串或字符串数组。</li></ul> |
| **返回** | JSON 对象，键为输入数组的索引，值为单词-词性对的列表。                            |

示例:

- 输入: `text=["我爱北京天安门"]`
- 输出: `{"0": [{"word": "我", "flag": "r"}, {"word": "爱", "flag": "v"}, ...]}`

### `extract_keywords`

使用向量化的针对关键词[BM25-ADPT算法](https://doi.org/10.1145/2063576.2063871)提取关键词。

|   项目   | 描述                                                                                                                                              |
| :------: | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| **参数** | <ul><li>`text` (required): 待提取的文本，可以是单个字符串或字符串数组。</li><li>`top_k` (optional): 每个文档提取的关键词数量 (默认 3)。</li></ul> |
| **返回** | JSON 对象，键为输入数组的索引，值为关键词列表。                                                                                                   |

示例:

- 输入: `text=["我爱北京天安门"], top_k=2`
- 输出: `{"0": ["天安门", "北京"]}`

------------------

## 鸣谢

- @messense/jieba-rs
- @messense/rjieba-py
- BM25-ADPT <https://doi.org/10.1145/2063576.2063871>
- GitHub Copilot
