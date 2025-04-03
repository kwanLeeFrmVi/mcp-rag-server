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
import { FaissVectorStore } from "./faissVectorStore.js";

class RAGManager {
  private vectorStore: FaissVectorStore | null = null;
  private embedder: (text: string) => Promise<number[]>;
  private baseLlmApi: string;
  private llmApiKey: string;
  private embeddingModel: string;
  private vectorStorePath: string;
  private chunkSize: number;

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

  async indexDocuments(docPath: string): Promise<void> {
    console.error(`Indexing documents from ${docPath}...`);

    const docs: Document[] = [];

    try {
      const stat = statSync(docPath);
      console.info(
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

      this.vectorStore = new FaissVectorStore(
        this.embedder,
        this.vectorStorePath,
        embeddingDimension
      );
      await (this.vectorStore as FaissVectorStore).initialize();
    }

    await this.vectorStore.addDocuments(docs);
    console.error("Indexing complete");
  }

  // Simple text chunking function
  private chunkText(
    text: string,
    chunkSize: number = this.chunkSize,
    overlap: number = 100
  ): string[] {
    const chunks: string[] = [];
    let i = 0;

    while (i < text.length) {
      let chunk = text.slice(i, i + chunkSize);
      chunks.push(chunk);
      i += chunkSize - overlap;
    }

    return chunks;
  }

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

        this.vectorStore = new FaissVectorStore(
          this.embedder,
          this.vectorStorePath,
          embeddingDimension
        );
        await (this.vectorStore as FaissVectorStore).initialize();

        // Check if the store has documents after initialization
        if ((this.vectorStore as FaissVectorStore).size === 0) {
          throw new Error("No documents found in vector store");
        }
      } catch (error) {
        throw new Error(
          "Documents not indexed yet. Please run index_documents first."
        );
      }
    }

    console.error(`Searching for documents relevant to: "${query}"`);
    const results = await this.vectorStore.similaritySearch(query, k);
    console.error(`Found ${results.length} relevant document chunks`);

    // Format the relevant chunks
    const contextText = results
      .map((res: Document) => `- From ${res.metadata.source}:\n${res.content}`)
      .join("\n\n");

    // Generate answer using LLM
    return contextText;
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
        description: "Index documents from specified path for RAG",
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
    ],
  };
});

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  if (!args) {
    throw new Error(`No arguments provided for tool: ${name}`);
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
      case "query_documents":
        const result = await ragManager.queryDocuments(
          args.query as string,
          (args.k as number) || 15
        );
        return {
          content: [
            {
              type: "text",
              text: result, // This is already the LLM-generated answer from generateAnswer
            },
          ],
        };
      default:
        throw new Error(`Unknown tool: ${name}`);
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
    console.error("Starting RAG MCP Server...");
    console.error(`Using LLM API: ${process.env.BASE_LLM_API}`);
    console.error(`Using embedding model: ${process.env.EMBEDDING_MODEL}`);

    const transport = new StdioServerTransport();
    await server.connect(transport);
    console.info("RAG MCP Server running on stdio");
  } catch (error) {
    handleFatalError("Error during server initialization:", error);
  }
}

// Helper function for handling fatal errors
function handleFatalError(message: string, error: unknown): void {
  console.error(message, error);
  process.exit(1);
}

main();
