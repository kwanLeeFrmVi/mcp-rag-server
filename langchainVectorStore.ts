import { join } from "path";
import { mkdirSync, existsSync } from "fs";
import { LibSQLVectorStore } from "@langchain/community/vectorstores/libsql";
import { Document as LangChainDocument } from "@langchain/core/documents";
import { Embeddings } from "@langchain/core/embeddings";
import { createClient } from "@libsql/client";
import type { Client } from "@libsql/client";
import type { Document, VectorStore } from "./types.js";
import { debounce } from "./utils.js";

/**
 * Custom embeddings adapter for LangChain
 */
class CustomEmbeddings implements Embeddings {
  caller: any = null;
  constructor(private embedFn: (text: string) => Promise<number[]>) {}

  async embedDocuments(texts: string[]): Promise<number[][]> {
    const embeddings = await Promise.all(texts.map(this.embedFn));
    return embeddings;
  }

  async embedQuery(text: string): Promise<number[]> {
    return this.embedFn(text);
  }
}

/**
 * LangChainVectorStore: A vector store implementation using LangChain's LibSQLVectorStore
 */
export class LangChainVectorStore implements VectorStore {
  private vectorStore: LibSQLVectorStore | null = null;
  private documents = new Map<string, Document>();
  private readonly storePath: string;
  private readonly dimension: number;
  private initialized = false;
  private embeddings: CustomEmbeddings;
  private readonly debounceSaveIndex = debounce(() => {
    this.saveIndex().catch((err) => this.log("Error in saveIndex:", err));
  }, 1000);
  private client: Client;

  constructor(
    private embedder: (text: string) => Promise<number[]>,
    storePath: string = "./vector_store",
    dimension: number = 1536
  ) {
    this.storePath = storePath;
    this.dimension = dimension;
    this.embeddings = new CustomEmbeddings(embedder);

    if (!existsSync(storePath)) {
      mkdirSync(storePath, { recursive: true });
    }

    // Create a local LibSQL database
    const dbPath = join(this.storePath, "vector_store.db");
    this.client = createClient({
      url: `file:${dbPath}`,
    });

    this.log(`Vector store initialized with dimension ${dimension}`);
  }

  /**
   * Log messages for the vector store.
   * MCP servers must use console.error exclusively for logging.
   */
  private log(message: string, ...args: any[]): void {
    console.error(`[LangChainVectorStore] ${message}`, ...args);
  }

  /**
   * Initializes the vector store and loads any existing vectors from the database.
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;

    try {
      // First, ensure the table exists
      await this.ensureTableExists();

      // According to the documentation, we should initialize with empty documents first
      const emptyTexts: string[] = [];
      const emptyMetadata = {};

      this.vectorStore = await LibSQLVectorStore.fromTexts(
        emptyTexts,
        emptyMetadata,
        this.embeddings,
        {
          db: this.client,
          table: "langchain_vectors",
          column: "embedding",
        }
      );

      // Load existing documents from the database
      try {
        const result = await this.client.execute({
          sql: "SELECT id, content, metadata FROM langchain_vectors",
          args: [],
        });

        for (const row of result.rows) {
          try {
            const metadata = JSON.parse(row.metadata as string);
            if (metadata.path) {
              this.documents.set(metadata.path, {
                path: metadata.path,
                content: row.content as string,
                metadata: metadata,
              });
            }
          } catch (parseError) {
            this.log("Error parsing metadata:", parseError);
          }
        }

        this.log(`Loaded ${this.documents.size} documents from database`);
      } catch (loadError) {
        this.log("Error loading documents from database:", loadError);
        // Continue with empty store if load fails
      }

      this.initialized = true;
      this.log("Vector store initialized successfully");
    } catch (error) {
      this.log("Error initializing LangChainVectorStore:", error);
      throw error;
    }
  }

  private async ensureTableExists(): Promise<void> {
    try {
      // Create the table if it doesn't exist
      await this.client.execute({
        sql: `
          CREATE TABLE IF NOT EXISTS langchain_vectors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT,
            metadata TEXT,
            embedding F32_BLOB(${this.dimension})
          )
        `,
        args: [],
      });

      // Create the vector index
      await this.client.execute({
        sql: `
          CREATE INDEX IF NOT EXISTS idx_langchain_vectors_embedding 
          ON langchain_vectors(libsql_vector_idx(embedding))
        `,
        args: [],
      });

      this.log("Database table and index created or verified");
    } catch (error) {
      this.log("Error creating table or index:", error);
      throw error;
    }
  }

  /**
   * Adds new documents to the store.
   */
  async addDocuments(docs: Document[]): Promise<void> {
    if (!this.initialized) await this.initialize();
    if (!this.vectorStore) {
      this.log("Vector store not initialized.");
      return;
    }
    if (!docs?.length) {
      this.log("No documents provided to add.");
      return;
    }

    try {
      // Convert our Document type to LangChain Document type
      const langchainDocs = docs.map((doc) => {
        // Store the document in our map for later retrieval
        this.documents.set(doc.path, doc);

        return new LangChainDocument({
          pageContent: doc.content,
          metadata: {
            ...doc.metadata,
            path: doc.path,
          },
        });
      });

      // Add documents to the vector store
      await this.vectorStore.addDocuments(langchainDocs);
      this.log(`Added ${docs.length} documents to vector store`);

      // Save the index after adding documents
      this.debounceSaveIndex();
    } catch (error) {
      this.log("Error adding documents:", error);
    }
  }

  /**
   * Searches for documents similar to the query text.
   */
  async similaritySearch(queryText: string, k: number): Promise<Document[]> {
    this.log("Searching for similar documents:", queryText);
    if (!this.vectorStore) {
      this.log("Search attempted on an uninitialized vector store.");
      return [];
    }

    try {
      // Perform similarity search using LangChain
      const results = await this.vectorStore.similaritySearch(queryText, k);

      // Convert LangChain results to our Document type
      return results.map((result) => {
        const path = result.metadata.path as string;
        // Try to get the original document from our map
        const originalDoc = this.documents.get(path);

        if (originalDoc) {
          return {
            ...originalDoc,
            metadata: {
              ...originalDoc.metadata,
              score: result.metadata.score as number,
            },
          };
        }

        // If not found in our map, create a new document from the result
        return {
          path,
          content: result.pageContent,
          metadata: {
            source: (result.metadata.source as string) || "unknown",
            score: result.metadata.score as number,
          },
        };
      });
    } catch (error) {
      this.log("Error searching for similar documents:", error);
      return [];
    }
  }

  /**
   * Removes a document from the store.
   */
  async removeDocument(docPath: string): Promise<void> {
    if (!this.initialized || !this.vectorStore) {
      this.log("Store not initialized or missing vector store.");
      return;
    }
    if (!this.documents.has(docPath)) {
      this.log(`Document with path "${docPath}" not found.`);
      return;
    }

    try {
      // Delete the document from the database by path in metadata
      await this.client.execute({
        sql: "DELETE FROM langchain_vectors WHERE json_extract(metadata, '$.path') = ?",
        args: [docPath],
      });

      // Remove from our document map
      this.documents.delete(docPath);

      this.log(`Removed document: ${docPath}`);
    } catch (error) {
      this.log(`Error removing document ${docPath}:`, error);
    }
  }

  /**
   * Removes all documents from the store.
   */
  async removeAllDocuments(): Promise<void> {
    if (!this.initialized || !this.vectorStore) {
      this.log("Store not initialized or missing vector store.");
      return;
    }

    try {
      // Delete all documents from the database
      await this.client.execute({
        sql: "DELETE FROM langchain_vectors",
        args: [],
      });

      // Clear our document map
      this.documents.clear();

      this.log("All documents removed successfully");
    } catch (error) {
      this.log("Error removing all documents:", error);
    }
  }

  /**
   * Saves the current state (no-op for LangChain as it saves automatically)
   */
  private saveIndex(): Promise<void> {
    // LibSQLVectorStore saves automatically to the database
    // This is a no-op for compatibility
    this.log("Vector store state is saved automatically in the database");
    return Promise.resolve();
  }

  /**
   * List all document paths in the vector store
   * @returns Array of document paths
   */
  async listDocumentPaths(): Promise<string[]> {
    if (!this.initialized) {
      await this.initialize();
    }
    
    return Array.from(this.documents.keys());
  }
}
