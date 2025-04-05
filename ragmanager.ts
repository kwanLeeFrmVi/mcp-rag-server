import { statSync, readdirSync, readFileSync } from "fs";
import { join } from "path";
import { LangChainVectorStore } from "./langchainVectorStore.js";
import { Document } from "./types.js";

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
    const data = (await response.json()) as any;
    return data.data ? data.data[0].embedding : data.embedding;
  }

  /**
   * Indexes documents from the specified path into the vector store
   * @param docPath - Path to file or directory containing documents to index
   * @throws Error if path doesn't exist, is invalid, or contains no supported files
   * Supported file types: .json, .jsonl, .txt, .md, .csv
   */
  async indexDocuments(docPath: string): Promise<void> {
    const docs: Document[] = [];

    try {
      const stat = statSync(docPath);

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

      this.vectorStore = new LangChainVectorStore(
        this.embedder,
        this.vectorStorePath,
        embeddingDimension
      );
      await (this.vectorStore as LangChainVectorStore).initialize();
    }

    await this.vectorStore.addDocuments(docs);
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
    overlap: number = 50
  ): string[] {
    const chunks: string[] = [];

    // Analyze content structure to determine optimal chunking
    const isCode = text.includes("{") && text.includes("}");
    const isMarkdown = text.includes("# ") || text.includes("## ");

    // Adjust chunk size based on content type. Average of 300 and 500
    const dynamicChunkSize = isCode
      ? (300 + chunkSize) / 2
      : isMarkdown
      ? (500 + chunkSize) / 2
      : chunkSize;

    // Split by natural boundaries if possible
    if (isCode) {
      // Try to split at function boundaries for code
      const functionMatches = text.matchAll(/function\s+\w+\([^)]*\)\s*{/g);
      let lastEnd = 0;

      for (const match of functionMatches) {
        if (match.index !== undefined && match.index > lastEnd) {
          chunks.push(text.slice(lastEnd, match.index));
          lastEnd = match.index;
        }
      }

      if (lastEnd < text.length) {
        chunks.push(text.slice(lastEnd));
      }

      return chunks.filter((chunk) => chunk.trim().length > 0);
    } else if (isMarkdown) {
      // Split markdown at heading boundaries
      return text
        .split(/\n(?=#+\s)/)
        .filter((chunk) => chunk.trim().length > 0);
    }

    // Fallback to sliding window chunking
    let i = 0;
    while (i < text.length) {
      const chunk = text.slice(i, i + dynamicChunkSize);
      chunks.push(chunk);
      i += dynamicChunkSize - overlap;
    }

    return chunks;
  }

  /**
   * Deduplicates search results based on content
   * @param results - Array of document results
   * @returns Array of deduplicated document results
   */
  private deduplicateResults(results: Document[]): Document[] {
    const seen = new Set<string>();
    const uniqueResults: Document[] = [];

    for (const result of results) {
      // Create a unique key based on content
      const contentKey = result.content.trim();

      // Only add if we haven't seen this content before
      if (!seen.has(contentKey)) {
        seen.add(contentKey);
        uniqueResults.push(result);
      }
    }

    return uniqueResults;
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
      try {
        let embeddingDimension = 1536;
        if (
          this.embeddingModel.includes("granite-embedding") ||
          this.embeddingModel.includes("nomic-embed-text")
        ) {
          embeddingDimension = 768;
        }

        this.vectorStore = new LangChainVectorStore(
          this.embedder,
          this.vectorStorePath,
          embeddingDimension
        );
        await this.vectorStore.initialize();

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

    if (!this.vectorStore) {
      throw new Error("Vector store not initialized");
    }

    const results = await this.vectorStore.similaritySearch(query, k);
    const uniqueResults = this.deduplicateResults(results);

    // Format for LLM consumption
    const llmFormattedResults = uniqueResults.map((res) => {
      const docName = res.metadata.source
        .replace(/\.md$/, "")
        .replace(/\s+/g, "_");
      return `[DOCUMENT:${docName}]
${res.content.trim()}
[/DOCUMENT:${docName}]`;
    });

    return llmFormattedResults.join("\n\n");
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

    await this.vectorStore.removeDocument(path);
  }

  /**
   * Remove all documents from the vector store
   * @throws Error if vector store not initialized
   */
  async removeAllDocuments(): Promise<void> {
    if (!this.vectorStore) {
      throw new Error("Vector store not initialized");
    }

    await this.vectorStore.removeAllDocuments();
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

    const paths = await this.vectorStore.listDocumentPaths();
    return paths;
  }
}

const ragManager = new RAGManager();
export default ragManager;
