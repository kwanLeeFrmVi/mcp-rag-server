# mcp-rag-server

[![npm version](https://img.shields.io/npm/v/mcp-rag-server.svg)](https://www.npmjs.com/package/mcp-rag-server)  [![License](https://img.shields.io/npm/l/mcp-rag-server.svg)](LICENSE)  [![Node.js Version](https://img.shields.io/node/v/mcp-rag-server)](package.json)

A Model Context Protocol (MCP) server that enables Retrieval Augmented Generation (RAG). It indexes your documents and serves relevant context to Large Language Models via the MCP protocol.

## Integration Examples

### Generic MCP Client Configuration

```json
{
  "mcpServers": {
    "rag": {
      "command": "npx",
      "args": ["-y", "mcp-rag-server"],
      "env": {
        "BASE_LLM_API": "http://localhost:11434/v1",
        "EMBEDDING_MODEL": "nomic-embed-text",
        "VECTOR_STORE_PATH": "./vector_store",
        "CHUNK_SIZE": "500"
      }
    }
  }
}
```

### Example Interaction

```shell
# Index documents
>> tool:embedding_documents {"path":"./docs"}

# Check status
>> resource:embedding-status

<< rag://embedding/status
Current Path: ./docs/file1.md
Completed: 10
Failed: 0
Total chunks: 15
Failed Reason:
```

## Table of Contents

- [Integration Examples](#integration-examples)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
  - [MCP Tools](#mcp-tools)
  - [MCP Resources](#mcp-resources)
- [How RAG Works](#how-rag-works)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## Features

- Index documents in `.txt`, `.md`, `.json`, `.jsonl`, and `.csv` formats
- Customizable chunk size for splitting text
- Local vector store powered by SQLite (via LangChain's LibSQLVectorStore)
- Supports multiple embedding providers (OpenAI, Ollama, Granite, Nomic)
- Exposes MCP tools and resources over stdio for seamless integration with MCP clients

## Installation

### From npm

```bash
npm install -g mcp-rag-server
```

### From Source

```bash
git clone https://github.com/kwanLeeFrmVi/mcp-rag-server.git
cd mcp-rag-server
npm install
npm run build
npm start
```

## Quick Start

```bash
export BASE_LLM_API=http://localhost:11434/v1
export EMBEDDING_MODEL=granite-embedding-278m-multilingual-Q6_K-1743674737397:latest
export VECTOR_STORE_PATH=./vector_store
export CHUNK_SIZE=500

# Run (global install)
mcp-rag-server

# Or via npx
npx mcp-rag-server
```
> 💡 **Tip:** We recommend using [Ollama](https://ollama.com) for embedding. Install and pull the `nomic-embed-text` model:
```bash
ollama pull nomic-embed-text
export EMBEDDING_MODEL=nomic-embed-text
```

## Configuration

| Variable            | Description                                      | Default                           |
| ------------------- | ------------------------------------------------ | --------------------------------- |
| `BASE_LLM_API`      | Base URL for embedding API                       | `http://localhost:11434/v1`       |
| `LLM_API_KEY`       | API key for your LLM provider                    | (empty)                           |
| `EMBEDDING_MODEL`   | Embedding model identifier                       | `nomic-embed-text`                |
| `VECTOR_STORE_PATH` | Directory for local vector store                 | `./vector_store`                  |
| `CHUNK_SIZE`        | Characters per text chunk (number)               | `500`                             |
> 💡 **Recommendation:** Use Ollama embedding models like `nomic-embed-text` for best performance.

## Usage

### MCP Tools

Once running, the server exposes these tools via MCP:

- `embedding_documents(path: string)`: Index documents under the given path
- `query_documents(query: string, k?: number)`: Retrieve top `k` chunks (default 15)
- `remove_document(path: string)`: Remove a specific document
- `remove_all_documents(confirm: boolean)`: Clear the entire index (`confirm=true`)
- `list_documents()`: List all indexed document paths

### MCP Resources

Clients can also read resources via URIs:

- `rag://documents` — List all document URIs
- `rag://document/{path}` — Fetch full content of a document
- `rag://query-document/{numberOfChunks}/{query}` — Query documents as a resource
- `rag://embedding/status` — Check current indexing status (completed, failed, total)

## How RAG Works

1. **Indexing**: Reads files, splits text into chunks based on `CHUNK_SIZE`, and queues them for embedding.
2. **Embedding**: Processes each chunk sequentially against the embedding API, storing vectors in SQLite.
3. **Querying**: Embeds the query and retrieves nearest text chunks from the vector store, returning them to the client.

## Development

```bash
npm install
npm run build      # Compile TypeScript
npm start          # Run server
npm run watch      # Watch for changes
```

## Contributing

Contributions are welcome! Please open issues or pull requests on [GitHub](https://github.com/kwanLeeFrmVi/mcp-rag-server).

## License

MIT 2025 Quan Le
