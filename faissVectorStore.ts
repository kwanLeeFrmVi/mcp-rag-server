import { join } from "path";
import { mkdirSync, existsSync } from "fs";
import pkg from "faiss-node";
import Database from "better-sqlite3";
import * as sqliteVec from "sqlite-vec";
import type { Document, VectorStore } from "./types.js";
import { debounce } from "./utils.js";

type Embedding = number[];

/**
 * Extended Faiss Index with a reset method for clearing IDs.
 */
export class FaissIndex extends pkg.Index {
  reset?(): void {
    const total = this.ntotal();
    if (total === 0) return;

    const allIds = Array.from({ length: total }, (_, idx) => idx);
    this.removeIds(allIds);
  }
}

/**
 * Row structure for the vector storage database.
 */
interface VectorRow {
  path: string;
  document_data: string;
  vector: Embedding;
}

/**
 * FaissVectorStore: A FAISS vector store that persists data in an SQLite database using sqlite-vec.
 */
export class FaissVectorStore implements VectorStore {
  private index: FaissIndex | null = null;
  private documents = new Map<string, Document>();
  private vectors: Embedding[] = [];
  private pathToIndexMap = new Map<string, number>();
  private indexToPathMap = new Map<number, string>();

  private readonly storePath: string;
  private readonly dimension: number;
  private initialized = false;
  private readonly db: Database.Database;
  private readonly debounceSaveIndex = debounce(() => this.saveIndex(), 1000);

  constructor(
    private embedder: (text: string) => Promise<Embedding>,
    storePath: string = "./vector_store",
    dimension: number = 1536
  ) {
    this.storePath = storePath;
    this.dimension = dimension;

    if (!existsSync(storePath)) {
      mkdirSync(storePath, { recursive: true });
    }

    const dbPath = join(this.storePath, "vector_store.db");
    this.db = new Database(dbPath);
    
    try {
      // Load the sqlite-vec extension
      sqliteVec.load(this.db);
      
      // Create the vector store table with explicit vector handling
      this.db.exec(`
        -- Create the vector store table if it doesn't exist
        CREATE TABLE IF NOT EXISTS vector_store (
          path TEXT PRIMARY KEY,
          document_data TEXT NOT NULL,
          embedding BLOB NOT NULL
        );
        
        -- Create helper functions if they don't exist
        -- These will be used to verify the extension is working
        CREATE TABLE IF NOT EXISTS _vec_test (v BLOB);
      `);
      
      // Verify the extension is loaded by testing a function
      try {
        // Test with a small vector
        const testVector = [0.1, 0.2, 0.3, 0.4, 0.5];
        const result = this.db.prepare("SELECT vec_from_blob(vec_to_blob(?))").get(testVector);
        this.log("SQLite vector extension loaded successfully");
      } catch (testError) {
        this.log("Vector functions test failed:", testError);
        throw new Error(`SQLite vector functions test failed: ${testError}`);
      }
    } catch (error) {
      this.log("Failed to load SQLite vector extension:", error);
      throw new Error(`SQLite vector extension initialization failed: ${error}`);
    }
  }

  /**
   * Log messages for the vector store.
   * MCP servers must use console.error exclusively for logging.
   */
  private log(message: string, ...args: any[]): void {
    console.error(`[FaissVectorStore] ${message}`, ...args);
  }

  /**
   * Initializes the vector store and loads any existing vectors from the database.
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;

    try {
      // Create a new FAISS index
      this.index = new pkg.IndexFlatL2(this.dimension);
      
      // Verify SQLite vector extension is working
      try {
        // Test the vector functions with a small test vector
        const testVector = [0.1, 0.2, 0.3, 0.4, 0.5];
        this.db.prepare("SELECT vec_from_blob(vec_to_blob(?))").get(testVector);
      } catch (sqliteError) {
        this.log("SQLite vector functions not available, attempting to reload extension:", sqliteError);
        
        // Try to reload the extension with explicit initialization
        sqliteVec.load(this.db);
        
        // Create a test table to verify vector operations
        this.db.exec(`
          -- Create a temporary test table for vector operations
          CREATE TABLE IF NOT EXISTS _vec_test (v BLOB);
          
          -- Insert a test vector to verify functionality
          DELETE FROM _vec_test;
        `);
        
        // Verify it worked by testing vector operations
        try {
          const testVector = [0.1, 0.2, 0.3, 0.4, 0.5];
          this.db.prepare("INSERT INTO _vec_test VALUES (vec_to_blob(?))").run(testVector);
          const result = this.db.prepare("SELECT vec_from_blob(v) FROM _vec_test LIMIT 1").get();
          this.log("SQLite vector extension successfully reloaded");
        } catch (reloadError) {
          throw new Error(`Failed to reload SQLite vector extension: ${reloadError}`);
        }
      }
      
      // Now try to load vectors from the database with explicit vector handling
      const rows = this.db
        .prepare(
          `
          SELECT 
            path, 
            document_data, 
            vec_from_blob(embedding) AS vector
          FROM vector_store
        `
        )
        .all() as VectorRow[];

      if (rows.length > 0) {
        this.log(`Loaded ${rows.length} vectors from database`);
        this.restoreIndexFromRows(rows);
      } else {
        this.log("No existing vectors found in database");
      }
      
      this.initialized = true;
      this.log("Vector store initialized successfully");
    } catch (error) {
      this.log("Error initializing FaissVectorStore:", error);
      this.resetInMemoryStore();
      throw error; // Re-throw to allow caller to handle the error
    }
  }

  /**
   * Adds new documents to the store after embedding and normalization.
   */
  async addDocuments(docs: Document[]): Promise<void> {
    if (!this.initialized) await this.initialize();
    if (!this.index) {
      this.log("FAISS index not initialized.");
      return;
    }
    if (!docs?.length) {
      this.log("No documents provided to add.");
      return;
    }

    const newDocs = docs.filter(
      (doc) => doc?.path && !this.documents.has(doc.path)
    );
    if (!newDocs.length) return;

    const validDocs = await this.prepareDocuments(newDocs);
    if (!validDocs.length) return;

    let currentIndex = this.index.ntotal();
    const vectorsToAdd: Embedding[] = [];

    validDocs.forEach(({ doc, embedding }) => {
      this.documents.set(doc.path, doc);
      this.pathToIndexMap.set(doc.path, currentIndex);
      this.indexToPathMap.set(currentIndex, doc.path);
      vectorsToAdd.push(embedding);
      currentIndex++;
    });

    this.index.add(vectorsToAdd.flat());
    this.vectors.push(...vectorsToAdd);
    this.debounceSaveIndex();
  }

  /**
   * Searches for documents similar to the query text.
   */
  async similaritySearch(queryText: string, k: number): Promise<Document[]> {
    this.log(
      "🚀 ~ FaissVectorStore ~ similaritySearch ~ queryText:",
      queryText
    );
    if (!this.index || this.index.ntotal() === 0) {
      this.log("Search attempted on an empty or uninitialized index.");
      return [];
    }

    let queryVector: Embedding;
    try {
      queryVector = await this.embedder(queryText);
    } catch {
      this.log("Error generating query embedding.");
      return [];
    }

    const normalized = this.validateAndNormalizeEmbedding(queryVector);
    if (!normalized) {
      this.log("Invalid query embedding generated.");
      return [];
    }

    const { distances, labels: indices } = this.index.search(normalized, k);
    const results: Document[] = [];

    for (let i = 0; i < indices.length; i++) {
      const idx = indices[i];
      const distance = distances[i];

      if (idx === undefined || idx < 0 || distance === undefined) {
        this.log(`Skipping invalid search result at position ${i}.`);
        continue;
      }

      const path = this.indexToPathMap.get(idx);
      if (!path) {
        this.log(`No path found for index ${idx}. Skipping result.`);
        continue;
      }

      const doc = this.documents.get(path);
      if (!doc) {
        this.log(`No document data for path ${path}. Skipping result.`);
        continue;
      }

      const score = 1 / (1 + distance);
      results.push({
        ...doc,
        metadata: { ...doc.metadata, score },
      });
    }

    return results;
  }

  /**
   * Returns the number of documents in the store.
   */
  get size(): number {
    return this.index?.ntotal() ?? 0;
  }

  /**
   * Removes a document and rebuilds the index.
   */
  async removeDocument(docPath: string): Promise<void> {
    if (!this.initialized || !this.index) {
      this.log("Store not initialized or missing index.");
      return;
    }
    if (!this.documents.has(docPath)) {
      this.log(`Document with path "${docPath}" not found.`);
      return;
    }

    this.log(`Removing document: ${docPath}. Rebuilding index...`);
    this.documents.delete(docPath);
    const docIndex = this.pathToIndexMap.get(docPath);
    if (docIndex !== undefined) this.indexToPathMap.delete(docIndex);
    this.pathToIndexMap.delete(docPath);

    this.rebuildIndex();
    this.debounceSaveIndex();
  }

  /**
   * Removes multiple documents by their paths.
   */
  async removeDocumentsByPaths(paths: string[]): Promise<boolean> {
    let removedAny = false;
    for (const path of paths) {
      if (this.documents.has(path)) {
        await this.removeDocument(path);
        removedAny = true;
      }
    }
    return removedAny;
  }

  /**
   * Removes all documents and resets the index.
   */
  async removeAllDocuments(): Promise<void> {
    if (!this.initialized || !this.index) {
      this.log("Store not initialized or missing index.");
      return;
    }
    this.log("Removing all documents and resetting index.");

    try {
      // this.index.reset?.();
      this.rebuildIndex();
      this.resetInMemoryStore();
      this.db.prepare("DELETE FROM vector_store").run();
      this.log("All documents removed successfully.");
    } catch {
      this.log("Error removing all documents.");
    }
  }

  // -------------------------------------------
  //            Private Helper Methods
  // -------------------------------------------

  /**
   * Clears in-memory structures.
   */
  private resetInMemoryStore(): void {
    this.documents.clear();
    this.vectors = [];
    this.pathToIndexMap.clear();
    this.indexToPathMap.clear();
    this.initialized = false;
  }

  /**
   * Restores the in-memory index from database rows.
   */
  private restoreIndexFromRows(rows: VectorRow[]): void {
    if (!this.index) return;

    const vectorsToAdd: Embedding[] = [];
    let currentIndex = 0;

    for (const row of rows) {
      try {
        const doc = JSON.parse(row.document_data) as Document;
        const vector = row.vector;

        if (!doc || !doc.path || doc.path !== row.path) {
          this.log(`Inconsistent data for path='${row.path}'. Skipping.`);
          continue;
        }
        if (!vector || vector.length !== this.dimension) {
          this.log(`Invalid vector for path='${row.path}'. Skipping.`);
          continue;
        }

        this.documents.set(doc.path, doc);
        vectorsToAdd.push(vector);
        this.pathToIndexMap.set(doc.path, currentIndex);
        this.indexToPathMap.set(currentIndex, doc.path);
        currentIndex++;
      } catch {
        this.log(`Error parsing document data for path='${row.path}'.`);
      }
    }

    this.vectors = [...vectorsToAdd];
    if (vectorsToAdd.length > 0) {
      this.index.add(vectorsToAdd.flat());
    } else {
      this.log("No valid vectors found in DB restoration.");
    }
  }

  /**
   * Validates and normalizes an embedding vector.
   */
  private validateAndNormalizeEmbedding(
    embedding: Embedding
  ): Embedding | null {
    if (!Array.isArray(embedding) || embedding.length !== this.dimension) {
      return null;
    }
    const norm = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
    if (norm < 1e-10) return null;
    return embedding.map((val) => val / norm);
  }

  /**
   * Generates and normalizes embeddings for provided documents.
   */
  private async prepareDocuments(
    docs: Document[]
  ): Promise<Array<{ doc: Document; embedding: Embedding }>> {
    const results: Array<{ doc: Document; embedding: Embedding }> = [];

    for (const doc of docs) {
      if (!doc?.path || !doc.content?.trim()) {
        this.log(`Skipping document due to missing fields: path=${doc?.path}`);
        continue;
      }
      try {
        const rawEmbedding = await this.embedder(doc.content);
        const normalized = this.validateAndNormalizeEmbedding(rawEmbedding);
        if (!normalized) {
          this.log(`Invalid embedding for document at path=${doc.path}`);
          continue;
        }
        results.push({ doc, embedding: normalized });
      } catch {
        this.log(`Error generating embedding for path=${doc.path}`);
      }
    }
    return results;
  }

  /**
   * Rebuilds the FAISS index from the in-memory vectors.
   * @param isClear - When true, clears all vectors from the index.
   * @param chunkSize - Number of vectors to add per chunk.
   */
  private rebuildIndex(isClear = false, chunkSize = 1000): void {
    if (!this.index) {
      this.log("Cannot rebuild index: index is null.");
      return;
    }

    // Backup current state in case we need to revert.
    const oldVectors = this.vectors;
    const oldPathToIndex = new Map(this.pathToIndexMap);
    const oldIndexToPath = new Map(this.indexToPathMap);
    const oldSize = this.index.ntotal();

    this.log(`Rebuilding index. Old size: ${oldSize}.`);
    try {
      // Create a new index.
      const newIndex = new pkg.IndexFlatL2(this.dimension);

      if (!isClear) {
        // Re-add all existing vectors in chunks.
        for (let i = 0; i < oldVectors.length; i += chunkSize) {
          const chunk = oldVectors.slice(i, i + chunkSize);
          newIndex.add(chunk.flat());
        }
      } else {
        // When clearing the index, empty the in-memory vector array.
        this.vectors = [];
      }

      // Rebuild the mapping between document paths and FAISS index positions.
      const newPathToIndex = new Map<string, number>();
      const newIndexToPath = new Map<number, string>();
      let currentIdx = 0;

      if (!isClear) {
        for (const path of this.documents.keys()) {
          newPathToIndex.set(path, currentIdx);
          newIndexToPath.set(currentIdx, path);
          currentIdx++;
        }
      }
      // If isClear is true, new mappings will be empty.

      // Commit the new index and mappings.
      this.index = newIndex;
      this.pathToIndexMap = newPathToIndex;
      this.indexToPathMap = newIndexToPath;
      this.log(`Index rebuilt successfully. New size: ${this.index.ntotal()}.`);
    } catch (error) {
      this.log("Error during index rebuild; reverting to previous index.");
      // Revert in-memory state if error occurred.
      this.vectors = oldVectors;
      this.pathToIndexMap = oldPathToIndex;
      this.indexToPathMap = oldIndexToPath;
    }
  }

  /**
   * Saves the current in-memory state to the SQLite database.
   */
  private saveIndex(): void {
    if (!this.initialized || !this.index) {
      this.log("Skipping save: store not initialized or index missing.");
      return;
    }
    if (this.documents.size === 0 || this.vectors.length === 0) {
      try {
        this.db.prepare("DELETE FROM vector_store").run();
        this.log("DB cleared as in-memory store is empty.");
      } catch (error) {
        this.log("Error clearing DB table:", error);
      }
      return;
    }

    if (
      this.documents.size !== this.vectors.length ||
      this.documents.size !== this.index.ntotal()
    ) {
      this.log(`Data inconsistency detected. Aborting save.`);
      return;
    }

    this.log(`Saving ${this.documents.size} documents/vectors to SQLite...`);

    const pathsInMemory = [...this.documents.keys()];
    const transaction = this.db.transaction(() => {
      const placeholders = pathsInMemory.map(() => "?").join(",");
      this.db
        .prepare(
          `
          DELETE FROM vector_store
          WHERE path NOT IN (${placeholders})
        `
        )
        .run(pathsInMemory);

      // Prepare the insert statement with explicit vector conversion
      const insertStmt = this.db.prepare(`
        INSERT OR REPLACE INTO vector_store (path, document_data, embedding)
        VALUES (?, ?, vec_to_blob(?))
      `);

      for (const [path, doc] of this.documents.entries()) {
        const idx = this.pathToIndexMap.get(path);
        if (idx == null || idx >= this.vectors.length) {
          this.log(`Invalid index for path=${path}. Skipping save.`);
          continue;
        }
        insertStmt.run(path, JSON.stringify(doc), this.vectors[idx]);
      }
    });

    try {
      transaction();
      this.log("Database save transaction completed successfully.");
    } catch (error) {
      this.log("Database save transaction failed:", error);
    }
  }
}
