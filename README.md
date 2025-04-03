# mcp-rag-server

To install dependencies:

```bash
bun install
```

To run:

```bash
bun run index.ts
```

This project was created using `bun init` in bun v1.2.8. [Bun](https://bun.sh) is a fast all-in-one JavaScript runtime.

## Technical

using @modelcontextprotocol/sdk

## How RAG Works

This MCP server implements Retrieval Augmented Generation (RAG) to answer questions based on your documents. Here's the process:

1. **User Query:** You ask a question related to the content of your documents.
2. **Document Reading:** The server reads `json` `jsonl` `csv` `.txt ` and `.md` files from a path you provide.
3. **Embedding & Indexing:** It uses an embedding model to tokenize the text and stores these embeddings in a local vector database for efficient searching.
4. **Search:** When you ask a question, the server searches the vector database for document chunks relevant to your query using embeddings.
5. **Chunk Selection:** It selects the most relevant chunks (default is 15) based on similarity scores.
6. **Response Generation:** The retrieved chunks are combined with your original question and sent to a Large Language Model (LLM) via the configured API to generate a comprehensive answer.

```mermaid
flowchart LR
    A[User asks a question] --> B[RAG mcp server receives query]
    B --> C{Read/Index Documents (.txt, .md)}
    C --> D[Search vector database for relevant chunks]
    D -- Embeddings --> E[Select top 15 chunks]
    E --> F[Combine query + chunks for LLM]
    F --> G[Generate Answer via LLM API]
    G --> H[Return answer to user]
```

## Environment Variables

To configure the server, you need to set the following environment variables:

- `BASE_LLM_API`: The base URL for the Large Language Model API endpoint. accept LM studio API, ollama api, openrouter api, gemini openai compatibility api
- `LLM_API_KEY`: Your API key for authenticating with the LLM service.
- `EMBEDDING_MODEL`: Specifies the embedding model used for tokenizing and searching documents (e.g., 'text-embedding-ada-002', 'nomic-embed-text','granite-embedding').

These variables are crucial for connecting to the necessary AI services.

## Usage Examples

(Provide specific command examples here on how to point the server to document paths and ask questions)
