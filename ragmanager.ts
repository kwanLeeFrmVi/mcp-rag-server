import { statSync, readdirSync, readFileSync } from "fs";
import { join } from "path";
import { LangChainVectorStore } from "./langchainVectorStore.js";
import { Document, EmbeddingResponse } from "./types.js";

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
  // Queue for managing concurrent embedding requests
  private embeddingQueue: Promise<unknown> = Promise.resolve();

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
    this.embeddingModel = process.env.EMBEDDING_MODEL || "nomic-embed-text";
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
          const embeddingResponse = await this.fetchEmbedding({
            input: text,
            model: this.embeddingModel,
          });
          if (!embeddingResponse) {
            throw new Error("No embedding response received");
          }
          return embeddingResponse;
        } catch (error) {
          console.error(
            "[RAGManager] Error generating embeddings:",
            error instanceof Error ? error.message : String(error)
          );
          throw error;
        }
      };
    } else {
      this.embedder = async (text: string) => {
        try {
          const embeddingResponse = await this.fetchEmbedding({
            model: this.embeddingModel,
            prompt: text,
          });
          if (!embeddingResponse) {
            throw new Error("No embedding response received");
          }
          return embeddingResponse;
        } catch (error) {
          console.error(
            "[RAGManager] Error generating embeddings:",
            error instanceof Error ? error.message : String(error)
          );
          throw error;
        }
      };
    }
  }

  /**
   * Fetches embeddings for the given text payload
   * Uses a queue to ensure only one request is sent to the Ollama server at a time
   * Includes retry logic for connection errors
   * @param payload - Object containing text and model parameters
   * @param retries - Number of retries left (default: 3)
   * @param delay - Delay between retries in ms (default: 1000)
   * @returns Promise resolving to embedding vector (array of numbers)
   * @throws Error if API request fails after all retries
   */
  private async fetchEmbedding(
    payload: object,
    retries: number = 3,
    delay: number = 1000
  ): Promise<number[]> {
    // Create a new task that will be executed when previous tasks are complete
    return new Promise((resolve, reject) => {
      this.embeddingQueue = this.embeddingQueue
        .then(async () => {
          try {
            console.error("[RAGManager] Fetching embeddings...");
            // Actual API call logic
            const headers: Record<string, string> = {
              "Content-Type": "application/json",
            };
            if (this.llmApiKey) {
              headers.Authorization = `Bearer ${this.llmApiKey}`;
            }

            try {
              const response = await fetch(`${this.baseLlmApi}/embeddings`, {
                method: "POST",
                headers,
                body: JSON.stringify(payload),
              });

              if (!response.ok) {
                const errorData = await response.text();
                throw new Error(
                  `Embedding API error (${response.status}): ${errorData}`
                );
              }

              const data = (await response.json()) as EmbeddingResponse;

              // Ensure we have a valid embedding
              if (
                data.data &&
                data.data.length > 0 &&
                data.data[0]?.embedding
              ) {
                const embedding = data.data[0].embedding;
                resolve(embedding);
                return embedding; // Return for the next task in queue
              } else if (data.embedding) {
                resolve(data.embedding);
                return data.embedding; // Return for the next task in queue
              } else {
                throw new Error("No embedding found in response");
              }
            } catch (error) {
              // Check if we should retry
              if (retries > 0 && this.isRetryableError(error)) {
                console.error(
                  `[RAGManager] Retrying embedding request, ${retries} attempts left. Waiting ${delay}ms...`
                );
                // Wait before retrying
                await new Promise((r) => setTimeout(r, delay));
                // Retry with one less retry count and increased delay
                const result = await this.fetchEmbedding(
                  payload,
                  retries - 1,
                  delay * 1.5
                );
                resolve(result);
                return result;
              }

              // No more retries or non-retryable error
              reject(error);
              throw error; // Rethrow to maintain queue error state
            }
          } catch (error) {
            reject(error);
            throw error; // Rethrow to maintain queue error state
          }
        })
        .catch((error) => {
          // This catch is for the queue promise chain
          // Individual request errors are already handled in the try/catch above
          console.error(
            "[RAGManager] Error in embedding queue:",
            error instanceof Error ? error.message : String(error)
          );
          return Promise.resolve(); // Keep the queue going even if one request fails
        });
    });
  }

  /**
   * Determines if an error is retryable
   * @param error - The error to check
   * @returns boolean indicating if the error is retryable
   * @private
   */
  private isRetryableError(error: unknown): boolean {
    // Check for network errors, EOF errors, and server errors (5xx)
    if (error instanceof Error) {
      const errorMessage = error.message.toLowerCase();

      // Network-related errors
      if (
        errorMessage.includes("network") ||
        errorMessage.includes("eof") ||
        errorMessage.includes("timeout") ||
        errorMessage.includes("econnreset") ||
        errorMessage.includes("econnrefused") ||
        errorMessage.includes("socket hang up") ||
        errorMessage.includes("network error")
      ) {
        return true;
      }

      // Server errors (5xx)
      if (
        errorMessage.includes("500") ||
        errorMessage.includes("502") ||
        errorMessage.includes("503") ||
        errorMessage.includes("504")
      ) {
        return true;
      }

      // Rate limiting or overload errors
      if (
        errorMessage.includes("rate limit") ||
        errorMessage.includes("too many requests") ||
        errorMessage.includes("429")
      ) {
        return true;
      }
    }
    return false;
  }

  /**
   * Indexes documents from the specified path into the vector store
   * @param docPath - Path to file or directory containing documents to index
   * @throws Error if path doesn't exist, is invalid, or contains no supported files
   * Supported file types: .json, .jsonl, .txt, .md, .csv
   */
  // Track the root path being processed to only throw errors at the root level
  private currentProcessingPath: string = "";

  /**
   * Processes a file and adds its chunks to the documents array
   * @param filePath - Full path to the file
   * @param docs - Array of documents to add to
   * @private
   */
  private processFile(filePath: string, docs: Document[]): void {
    const fileName = filePath.split("/").pop() || filePath;
    const content = readFileSync(filePath, "utf-8");

    // Split content into chunks
    const chunks = this.chunkText(content, this.chunkSize, 200);

    chunks.forEach((chunk, index) => {
      docs.push({
        path: filePath,
        content: chunk,
        metadata: {
          source: `${fileName} (chunk ${index + 1}/${chunks.length})`,
        },
      });
    });
  }

  /**
   * Recursively processes directories and files
   * @param dirPath - Path to directory
   * @param docs - Array of documents to add to
   * @private
   */
  private processDirectory(dirPath: string, docs: Document[]): void {
    const entries = readdirSync(dirPath, { withFileTypes: true });
    let foundSupportedFiles = false;

    for (const entry of entries) {
      const fullPath = join(dirPath, entry.name);

      if (entry.isDirectory()) {
        // Recursively process subdirectories
        this.processDirectory(fullPath, docs);
      } else if (
        entry.isFile() &&
        (entry.name.endsWith(".json") ||
          entry.name.endsWith(".jsonl") ||
          entry.name.endsWith(".txt") ||
          entry.name.endsWith(".md") ||
          entry.name.endsWith(".csv"))
      ) {
        foundSupportedFiles = true;
        this.processFile(fullPath, docs);
      }
    }

    // Only throw an error if this is the root directory and no files were found
    if (
      dirPath === this.currentProcessingPath &&
      !foundSupportedFiles &&
      docs.length === 0
    ) {
      throw new Error(
        `No supported files found in ${dirPath} or its subdirectories`
      );
    }
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
      this.currentProcessingPath = docPath;

      if (stat.isDirectory()) {
        // Handle directory path recursively
        this.processDirectory(docPath, docs);
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

        this.processFile(docPath, docs);
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
        console.error(
          "[RAGManager] Error initializing vector store:",
          error instanceof Error ? error.message : String(error)
        );
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
        console.error(
          "[RAGManager] Error initializing vector store:",
          error instanceof Error ? error.message : String(error)
        );
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
