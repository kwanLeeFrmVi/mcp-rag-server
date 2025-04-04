#!/usr/bin/env node
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { readFileSync, readdirSync, statSync } from "fs";
import { join } from "path";
import fetch from "node-fetch";
import type { Document } from "./types.js";
import { LangChainVectorStore } from "./langchainVectorStore.js";

/**
 * Manages RAG (Retrieval-Augmented Generation) operations including document indexing,
 * querying, and vector store management. Handles document chunking, embedding generation,
 * and interaction with the vector store.
 */
class RAGManager {
  private vectorStore: LangChainVectorStore | null = null;
  private embedder: (text: string) => Promise<number[]>;
  private baseLlmApi: string;
  private llmApiKey: string;
  private embeddingModel: string;
  private vectorStorePath: string;
  private chunkSize: number;

  /**
   * Creates a new RAGManager instance.
   * Initializes configuration from environment variables:
   * - BASE_LLM_API: Base URL for LLM API (default: "http://localhost:11434/v1")
   * - LLM_API_KEY: API key for LLM service (default: "")
   * - EMBEDDING_MODEL: Name of embedding model to use
   * - VECTOR_STORE_PATH: Path to store vector index (default: "./vector_store")
   * - CHUNK_SIZE: Size of text chunks in characters (default: 500)
   */
  constructor() {
    // Get environment variables
    this.baseLlmApi = process.env.BASE_LLM_API || "http://localhost:11434/v1";
    this.llmApiKey = process.env.LLM_API_KEY || "";
    this.embeddingModel =
      process.env.EMBEDDING_MODEL ||
      "granite-embedding-278m-multilingual-Q6_K-1743674737397:latest";
    this.vectorStorePath = process.env.VECTOR_STORE_PATH || "./vector_store";
    this.chunkSize = +(process.env.CHUNK_SIZE || 500);

    // Configure embedder based on the embedding model
    if (
      this.embeddingModel.includes("text-embedding") ||
      this.embeddingModel.includes("nomic-embed") ||
      this.embeddingModel.includes("granite-embedding")
    ) {
      this.embedder = async (text: string) => {
        try {
          return await this.fetchEmbedding({
            input: text,
            model: this.embeddingModel,
          });
        } catch (error) {
          console.error("Error generating embeddings:", error);
          throw error;
        }
      };
    } else {
      this.embedder = async (text: string) => {
        try {
          return await this.fetchEmbedding({
            model: this.embeddingModel,
            prompt: text,
          });
        } catch (error) {
          console.error("Error generating embeddings:", error);
          throw error;
        }
      };
    }
  }

  /**
   * Fetches embeddings for the given text payload
   * @param payload - Object containing text and model parameters
   * @returns Promise resolving to embedding vector (array of numbers)
   * @throws Error if API request fails
   */
  private async fetchEmbedding(payload: object): Promise<number[]> {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (this.llmApiKey) {
      headers.Authorization = `Bearer ${this.llmApiKey}`;
    }
    const response = await fetch(`${this.baseLlmApi}/embeddings`, {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      const errorData = await response.text();
      throw new Error(`Embedding API error (${response.status}): ${errorData}`);
    }
    const data = await response.json();
    return data.data ? data.data[0].embedding : data.embedding;
  }

  /**
   * Indexes documents from the specified path into the vector store
   * @param docPath - Path to file or directory containing documents to index
   * @throws Error if path doesn't exist, is invalid, or contains no supported files
   * Supported file types: .json, .jsonl, .txt, .md, .csv
   */
  async indexDocuments(docPath: string): Promise<void> {
    console.error(`Indexing documents from ${docPath}...`);

    const docs: Document[] = [];

    try {
      const stat = statSync(docPath);
      console.error(
        "🚀 ~ RAGManager ~ indexDocuments ~ stat:",
        stat.isDirectory(),
        stat.isFile()
      );

      if (stat.isDirectory()) {
        // Handle directory path
        const files = readdirSync(docPath).filter(
          (file) =>
            file.endsWith(".json") ||
            file.endsWith(".jsonl") ||
            file.endsWith(".txt") ||
            file.endsWith(".md") ||
            file.endsWith(".csv")
        );

        if (files.length === 0) {
          throw new Error(`No supported files found in ${docPath}`);
        }

        for (const file of files) {
          const filePath = join(docPath, file);
          const content = readFileSync(filePath, "utf-8");

          // Split content into chunks (simple implementation - could be improved)
          const chunks = this.chunkText(content, this.chunkSize, 200);

          chunks.forEach((chunk, index) => {
            docs.push({
              path: filePath,
              content: chunk,
              metadata: {
                source: `${file} (chunk ${index + 1}/${chunks.length})`,
              },
            });
          });
        }
      } else if (stat.isFile()) {
        // Handle single file path
        const fileName = docPath.split("/").pop() || docPath;

        // Check if file has a supported extension
        if (
          !(
            fileName.endsWith(".json") ||
            fileName.endsWith(".jsonl") ||
            fileName.endsWith(".txt") ||
            fileName.endsWith(".md") ||
            fileName.endsWith(".csv")
          )
        ) {
          throw new Error(`Unsupported file type: ${fileName}`);
        }

        const content = readFileSync(docPath, "utf-8");

        // Split content into chunks
        const chunks = this.chunkText(content, this.chunkSize, 200);

        chunks.forEach((chunk, index) => {
          docs.push({
            path: docPath,
            content: chunk,
            metadata: {
              source: `${fileName} (chunk ${index + 1}/${chunks.length})`,
            },
          });
        });
      } else {
        throw new Error(`Path is neither a file nor a directory: ${docPath}`);
      }
    } catch (error: unknown) {
      if (
        error instanceof Error &&
        "code" in error &&
        error.code === "ENOENT"
      ) {
        throw new Error(`Path does not exist: ${docPath}`);
      }
      throw error;
    }

    console.error(`Created ${docs.length} document chunks for indexing`);

    // Use FAISS vector store for persistent storage
    if (!this.vectorStore) {
      // Determine embedding dimension based on model
      let embeddingDimension = 1536; // Default for OpenAI models

      if (
        this.embeddingModel.includes("granite-embedding") ||
        this.embeddingModel.includes("nomic-embed-text")
      ) {
        embeddingDimension = 768; // Granite and some Nomic models use 768 dimensions
      }

      console.error(
        `Using embedding dimension: ${embeddingDimension} for model: ${this.embeddingModel}`
      );

      this.vectorStore = new LangChainVectorStore(
        this.embedder,
        this.vectorStorePath,
        embeddingDimension
      );
      await (this.vectorStore as LangChainVectorStore).initialize();
    }

    await this.vectorStore.addDocuments(docs);
    console.error("Indexing complete");
  }

  /**
   * Splits text into chunks with optional overlap
   * @param text - Text to chunk
   * @param chunkSize - Size of each chunk in characters (default: this.chunkSize)
   * @param overlap - Number of overlapping characters between chunks (default: 100)
   * @returns Array of text chunks
   */
  private chunkText(
    text: string,
    chunkSize: number = this.chunkSize,
    overlap: number = 100
  ): string[] {
    const chunks: string[] = [];
    let i = 0;

    while (i < text.length) {
      const chunk = text.slice(i, i + chunkSize);
      chunks.push(chunk);
      i += chunkSize - overlap;
    }

    return chunks;
  }

  /**
   * Queries the vector store for documents relevant to the query
   * @param query - Search query
   * @param k - Number of results to return (default: 15)
   * @returns Formatted string containing relevant document chunks
   * @throws Error if vector store not initialized or no documents indexed
   */
  async queryDocuments(query: string, k = 15): Promise<string> {
    if (!this.vectorStore) {
      // Try to initialize the vector store from disk
      try {
        // Determine embedding dimension based on model
        let embeddingDimension = 1536; // Default for OpenAI models

        if (
          this.embeddingModel.includes("granite-embedding") ||
          this.embeddingModel.includes("nomic-embed-text")
        ) {
          embeddingDimension = 768; // Granite and some Nomic models use 768 dimensions
        }

        console.error(
          `Using embedding dimension: ${embeddingDimension} for model: ${this.embeddingModel}`
        );

        this.vectorStore = new LangChainVectorStore(
          this.embedder,
          this.vectorStorePath,
          embeddingDimension
        );
        await this.vectorStore.initialize();

        // Check if the vector store is empty by attempting a search
        if (this.vectorStore) {
          const testResults = await this.vectorStore.similaritySearch(
            "test",
            1
          );
          if (testResults.length === 0) {
            throw new Error("No documents found in vector store");
          }
        }
      } catch (error) {
        throw new Error(
          "Documents not indexed yet. Please run index_documents first."
        );
      }
    }

    console.error(`Searching for documents relevant to: "${query}"`);
    if (!this.vectorStore) {
      throw new Error("Vector store not initialized");
    }
    const results = await this.vectorStore.similaritySearch(query, k);
    console.error(`Found ${results.length} relevant document chunks`);

    // Format the relevant chunks
    const contextText = results
      .map((res: Document) => `- From ${res.metadata.source}:\n${res.content}`)
      .join("\n\n");

    // Generate answer using LLM
    return contextText;
  }

  /**
   * Remove a document from the vector store by path
   * @param path - Path to the document to remove
   * @throws Error if vector store not initialized
   */
  async removeDocument(path: string): Promise<void> {
    if (!this.vectorStore) {
      throw new Error("Vector store not initialized");
    }

    console.error(`Removing document: ${path}`);
    await this.vectorStore.removeDocument(path);
    console.error(`Document removed: ${path}`);
  }

  /**
   * Remove all documents from the vector store
   * @throws Error if vector store not initialized
   */
  async removeAllDocuments(): Promise<void> {
    if (!this.vectorStore) {
      throw new Error("Vector store not initialized");
    }

    console.error("Removing all documents from the vector store");
    await this.vectorStore.removeAllDocuments();
    console.error("All documents removed");
  }

  /**
   * List all document paths in the vector store
   * @returns Array of document paths
   * @throws Error if vector store not initialized or no documents indexed
   */
  async listDocumentPaths(): Promise<string[]> {
    if (!this.vectorStore) {
      // Try to initialize the vector store from disk
      try {
        // Determine embedding dimension based on model
        let embeddingDimension = 1536; // Default for OpenAI models

        if (
          this.embeddingModel.includes("granite-embedding") ||
          this.embeddingModel.includes("nomic-embed-text")
        ) {
          embeddingDimension = 768; // Granite and some Nomic models use 768 dimensions
        }

        console.error(
          `Using embedding dimension: ${embeddingDimension} for model: ${this.embeddingModel}`
        );

        this.vectorStore = new LangChainVectorStore(
          this.embedder,
          this.vectorStorePath,
          embeddingDimension
        );
        await this.vectorStore.initialize();
      } catch (error) {
        throw new Error(
          "Documents not indexed yet. Please run index_documents first."
        );
      }
    }

    console.error("Listing all document paths in the vector store");
    const paths = await this.vectorStore.listDocumentPaths();
    console.error(`Found ${paths.length} documents in the vector store`);
    return paths;
  }
}

const ragManager = new RAGManager();

const server = new Server(
  {
    name: "rag-server",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "index_documents",
        description: "Add documents from specified path for RAG Indexing",
        inputSchema: {
          type: "object",
          properties: {
            path: {
              type: "string",
              description: "Path containing .txt and .md files to index",
            },
          },
          required: ["path"],
        },
      },
      {
        name: "query_documents",
        description: "Query indexed documents using RAG",
        inputSchema: {
          type: "object",
          properties: {
            query: {
              type: "string",
              description: "The question to search documents for",
            },
            k: {
              type: "number",
              description: "Number of chunks to return (default: 15)",
            },
          },
          required: ["query"],
        },
      },
      {
        name: "remove_document",
        description: "Remove a specific document from the index by file path",
        inputSchema: {
          type: "object",
          properties: {
            path: {
              type: "string",
              description: "Path to the document to remove from the index",
            },
          },
          required: ["path"],
        },
      },
      {
        name: "remove_all_documents",
        description: "Remove all documents from the index",
        inputSchema: {
          type: "object",
          properties: {
            confirm: {
              type: "boolean",
              description:
                "Confirmation to remove all documents (must be true)",
            },
          },
          required: ["confirm"],
        },
      },
      {
        name: "list_documents",
        description: "List all document paths in the index",
        inputSchema: {
          type: "object",
          properties: {},
        },
      },
    ],
  };
});

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  if (!args) {
    console.error(`No arguments provided for tool: ${name}`);
    return {
      content: [
        {
          type: "text",
          text: `Error: No arguments provided for tool: ${name}`,
        },
      ],
    };
  }

  try {
    switch (name) {
      case "index_documents":
        await ragManager.indexDocuments(args.path as string);
        return {
          content: [
            {
              type: "text",
              text: `Successfully indexed documents from ${args.path}`,
            },
          ],
        };
      case "query_documents": {
        const result = await ragManager.queryDocuments(
          args.query as string,
          (args.k as number) || 15
        );
        return {
          content: [
            {
              type: "text",
              text: result,
            },
          ],
        };
      }
      case "remove_document":
        if (!args.path) {
          return {
            content: [
              {
                type: "text",
                text: "Error: No document path provided",
              },
            ],
          };
        }

        try {
          await ragManager.removeDocument(args.path as string);
          return {
            content: [
              {
                type: "text",
                text: `Successfully removed document: ${args.path}`,
              },
            ],
          };
        } catch (error) {
          console.error(`Error removing document: ${error}`);
          return {
            content: [
              {
                type: "text",
                text: `Error removing document: ${error}`,
              },
            ],
          };
        }

      case "remove_all_documents":
        if (!args.confirm) {
          return {
            content: [
              {
                type: "text",
                text: "Error: You must set confirm=true to remove all documents",
              },
            ],
          };
        }

        try {
          await ragManager.removeAllDocuments();
          return {
            content: [
              {
                type: "text",
                text: "Successfully removed all documents from the index",
              },
            ],
          };
        } catch (error) {
          console.error(`Error removing all documents: ${error}`);
          return {
            content: [
              {
                type: "text",
                text: `Error removing all documents: ${error}`,
              },
            ],
          };
        }

      case "list_documents":
        try {
          const paths = await ragManager.listDocumentPaths();

          if (paths.length === 0) {
            return {
              content: [
                {
                  type: "text",
                  text: "No documents found in the index.",
                },
              ],
            };
          }

          return {
            content: [
              {
                type: "text",
                text: `Found ${paths.length} documents in the index:\n\n${paths
                  .map((path) => `- ${path}`)
                  .join("\n")}`,
              },
            ],
          };
        } catch (error) {
          console.error(`Error listing documents: ${error}`);
          return {
            content: [
              {
                type: "text",
                text: `Error listing documents: ${error}`,
              },
            ],
          };
        }
      default:
        console.error(`Unknown tool: ${name}`);
        return {
          content: [
            {
              type: "text",
              text: `Error: Unknown tool: ${name}`,
            },
          ],
        };
    }
  } catch (error) {
    console.error(`Error handling tool ${name}:`, error);
    return {
      content: [
        {
          type: "text",
          text: `Error: ${
            error instanceof Error ? error.message : String(error)
          }`,
        },
      ],
    };
  }
});

async function main() {
  try {
    const transport = new StdioServerTransport();
    await server.connect(transport);
    console.error("RAG MCP Server running on stdio");
  } catch (error) {
    handleFatalError("Error during server initialization:", error);
  }
}

function handleFatalError(message: string, error: unknown): void {
  console.error(
    message,
    error instanceof Error ? error.message : String(error)
  );
  process.exit(1);
}

main();
