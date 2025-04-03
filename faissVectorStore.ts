import { join } from "path";
import { mkdirSync, existsSync } from "fs";
import pkg from "faiss-node";
import Database from "better-sqlite3";
import * as sqliteVec from "sqlite-vec";
import type { Document, VectorStore } from "./types.js";
import { debounce } from "./utils.js";

// Define the interface for the Index class from faiss-node
interface FaissIndex extends pkg.Index {}

// Define the structure of rows returned from the database
interface VectorRow {
  path: string;
  document_data: string;
  vector: number[];
}

/**
 * FaissVectorStore - A vector store implementation using FAISS for efficient similarity search
 * This implementation persists vectors to an SQLite database using sqlite-vec
 */
export class FaissVectorStore implements VectorStore {
  private index: FaissIndex | null = null;
  private documents: Map<string, Document> = new Map(); // Map path -> Document
  private vectors: number[][] = []; // In-memory store aligned with FAISS index order
  private pathToIndexMap: Map<string, number> = new Map(); // Map path -> FAISS index
  private indexToPathMap: Map<number, string> = new Map(); // Map FAISS index -> path

  private storePath: string;
  private dimension: number;
  private initialized = false;
  private db: Database.Database;
  private debounceSaveIndex: () => void;

  /**
   * Create a new FAISS vector store
   * @param embedder Function to convert text to vector embeddings
   * @param storePath Directory to store SQLite database
   * @param dimension Dimension of the embedding vectors
   */
  constructor(
    private embedder: (text: string) => Promise<number[]>,
    storePath: string = "./vector_store",
    dimension: number = 1536
  ) {
    this.storePath = storePath;
    this.dimension = dimension;

    if (!existsSync(storePath)) {
      mkdirSync(storePath, { recursive: true });
    }

    // Initialize SQLite database with vector support
    const dbPath = join(this.storePath, "vector_store.db");
    this.db = new Database(dbPath);
    sqliteVec.load(this.db);

    // Create table for vector storage
    this.db
      .prepare(
        `
        CREATE TABLE IF NOT EXISTS vector_store (
          path TEXT PRIMARY KEY,
          document_data TEXT NOT NULL,
          embedding BLOB NOT NULL
        )
      `
      )
      .run();

    // Initialize debounced save function
    this.debounceSaveIndex = debounce(() => this.saveIndex(), 1000); // Debounce for 1 second
  }

  /**
   * Initialize the vector store, loading existing index from SQLite
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;

    try {
      this.index = new pkg.IndexFlatL2(this.dimension); // Use FlatL2 for simplicity

      // Load vectors from SQLite
      const rows = this.db
        .prepare(
          `
          SELECT path, document_data, vec_from_blob(embedding) as vector
          FROM vector_store
        `
        )
        .all() as VectorRow[];

      if (rows.length > 0) {
        const vectorsToAdd: number[][] = [];
        let currentIndex = 0;
        for (const row of rows) {
          try {
            const doc: Document = JSON.parse(row.document_data);
            const vector = row.vector;

            if (!doc || !doc.path || doc.path !== row.path) {
              console.error(
                `Skipping row with inconsistent data: path='${row.path}'`
              );
              continue;
            }
            if (!vector || vector.length !== this.dimension) {
              console.error(
                `Skipping row with invalid vector for path: ${row.path}`
              );
              continue;
            }

            this.documents.set(doc.path, doc);
            vectorsToAdd.push(vector);
            this.pathToIndexMap.set(doc.path, currentIndex);
            this.indexToPathMap.set(currentIndex, doc.path);
            currentIndex++;
          } catch (parseError) {
            console.error(
              `Error parsing document data for path ${row.path}:`,
              parseError
            );
          }
        }

        this.vectors = [...vectorsToAdd]; // Store loaded vectors in memory

        if (vectorsToAdd.length > 0 && this.index) {
          this.index.add(vectorsToAdd.flat()); // Flatten the array of vectors
        } else {
          console.error("No valid vectors found to add to the index.");
        }
      }

      this.initialized = true;
    } catch (error) {
      console.error("Error initializing FaissVectorStore:", error);
      // Reset state on failure
      this.index = null;
      this.documents.clear();
      this.vectors = [];
      this.pathToIndexMap.clear();
      this.indexToPathMap.clear();
      this.initialized = false; // Mark as not initialized
    }
  }

  /**
   * Add documents to the vector store
   * @param docs Array of documents to add
   */
  async addDocuments(docs: Document[]): Promise<void> {
    if (!this.initialized) {
      await this.initialize();
      if (!this.index) {
        console.error("Cannot add documents: FAISS index not initialized.");
        return;
      }
    }
    // Ensure index is available after potential initialization
    if (!this.index) {
      console.error(
        "Cannot add documents: FAISS index is null even after initialization attempt."
      );
      return;
    }

    if (!docs || docs.length === 0) {
      console.error("No documents provided to add.");
      return;
    }

    try {
      const newDocs = docs.filter(
        (doc) => doc && doc.path && !this.documents.has(doc.path)
      );
      if (newDocs.length === 0) {
        return;
      }

      const validDocsWithEmbeddings: { doc: Document; embedding: number[] }[] =
        [];

      for (let i = 0; i < newDocs.length; i++) {
        const doc = newDocs[i];
        const text = doc.text;

        if (!text || text.trim() === "") {
          console.error(
            `Skipping document with empty content (path: ${doc.path})`
          );
          continue;
        }

        try {
          const embedding = await this.embedder(text);
          if (!Array.isArray(embedding) || embedding.length !== this.dimension) {
            console.error(
              `Skipping document (path: ${doc.path}) with invalid embedding dimension: got ${embedding?.length}, expected ${this.dimension}`
            );
            continue;
          }

          const norm = Math.sqrt(
            embedding.reduce((sum, val) => sum + val * val, 0)
          );
          if (norm < 1e-10) {
            console.error(
              `Skipping document (path: ${doc.path}) with zero-norm embedding`
            );
            continue;
          }
          const normalizedEmbedding = embedding.map((val) => val / norm);
          validDocsWithEmbeddings.push({ doc, embedding: normalizedEmbedding });
        } catch (embeddingError) {
          console.error(
            `Error embedding document (path: ${doc.path}):`,
            embeddingError
          );
          continue;
        }
      }

      if (validDocsWithEmbeddings.length === 0) {
        return;
      }

      // Add to in-memory structures and FAISS index
      const vectorsToAdd: number[][] = [];
      let currentFaissIndex = this.index.ntotal(); // Start from the current size

      validDocsWithEmbeddings.forEach(({ doc, embedding }) => {
        this.documents.set(doc.path, doc);
        this.pathToIndexMap.set(doc.path, currentFaissIndex);
        this.indexToPathMap.set(currentFaissIndex, doc.path);
        vectorsToAdd.push(embedding);
        currentFaissIndex++;
      });

      this.index.add(vectorsToAdd.flat());
      this.vectors.push(...vectorsToAdd);
      this.debounceSaveIndex();
    } catch (error) {
      console.error("Error during addDocuments:", error);
    }
  }

  /**
   * Save the current in-memory state to the SQLite database.
   * This replaces the entire table content with the current state.
   */
  private saveIndex(): void {
    // Check if initialized and index exists
    if (!this.initialized || !this.index) {
      console.error(
        "Skipping save: Store not initialized or FAISS index missing."
      );
      return;
    }
    // Check if there's anything to save
    if (this.documents.size === 0 || this.vectors.length === 0) {
      console.error("Skipping save: No documents or vectors in memory.");
      // Ensure DB is also empty if memory is empty
      try {
        this.db.prepare("DELETE FROM vector_store").run();
        console.error("Cleared database as in-memory store is empty.");
      } catch (clearError) {
        console.error("Error clearing database:", clearError);
      }
      return;
    }

    if (
      this.documents.size !== this.vectors.length ||
      this.documents.size !== this.index.ntotal()
    ) {
      console.error(`Data inconsistency detected before save! 
          Documents: ${this.documents.size}, 
          In-memory Vectors: ${this.vectors.length}, 
          FAISS Index: ${this.index.ntotal()}. Aborting save.`);
      // Potentially try to recover or log more details
      return; // Prevent saving inconsistent state
    }

    console.error(`Saving ${this.documents.size} documents/vectors to SQLite...`);

    const transaction = this.db.transaction(() => {
      try {
        this.db.prepare("DELETE FROM vector_store").run();
        console.error("Cleared existing data from vector_store table.");

        const insertStmt = this.db.prepare(
          `
          INSERT INTO vector_store (path, document_data, embedding)
          VALUES (?, ?, vec_to_blob(?))
          `
        );

        for (const [path, doc] of this.documents.entries()) {
          const index = this.pathToIndexMap.get(path);
          // Ensure doc and index are valid before proceeding
          if (doc && index !== undefined && index < this.vectors.length) {
            const vector = this.vectors[index];
            if (vector) {
              // Ensure vector exists
              insertStmt.run(path, JSON.stringify(doc), vector);
            } else {
              console.error(
                `Vector not found for document path ${path} at index ${index}`
              );
            }
          } else {
            // Handle cases where doc might be missing or index invalid
            if (!doc) {
              console.error(
                `Document data missing for path ${path} during save.`
              );
            }
            if (index === undefined) {
              console.error(
                `Index mapping missing for path ${path} during save.`
              );
            }
            if (index !== undefined && index >= this.vectors.length) {
              console.error(
                `Index ${index} out of bounds for vectors array (length ${this.vectors.length}) for path ${path}.`
              );
            }
          }
        }

        console.error(
          `Successfully inserted ${this.documents.size} documents into SQLite.`
        );
      } catch (error) {
        console.error("Error during save transaction:", error);
        throw error; // Ensure transaction rollback
      }
    });

    try {
      transaction();
      console.error("Database save transaction completed successfully.");
    } catch (error) {
      console.error("Database save transaction failed.");
    }
  }

  /**
   * Search for similar documents using cosine similarity (via normalized vectors).
   * @param queryText The text to search for
   * @param k The number of similar documents to return
   * @returns Array of documents similar to the query text, with scores.
   */
  async similaritySearch(queryText: string, k: number): Promise<Document[]> {
    if (!this.index || (this.index.ntotal() ?? 0) === 0) {
      console.error("Attempted search on empty or non-existent index.");
      return [];
    }

    const { distances, labels: indices } = this.index.search(queryText, k);

    const results: Document[] = [];
    for (let i = 0; i < indices.length; i++) {
      const index = indices[i];
      const distance = distances[i];

      // Check if index and distance are valid
      if (index === undefined || index < 0 || distance === undefined) {
        console.error(
          `Invalid index (${index}) or distance (${distance}) at result position ${i}. Skipping.`
        );
        continue; // Skip this result
      }

      const path = this.indexToPathMap.get(index);
      if (!path) {
        console.error(`No path found for index ${index}. Skipping result.`);
        continue; // Skip if no path mapping found
      }

      const doc = this.documents.get(path);
      if (!doc) {
        console.error(
          `Document data not found for path ${path} (index ${index}). Skipping result.`
        );
        continue; // Skip if document data is missing
      }

      // Assuming lower distance means higher similarity for L2
      // Convert distance to a score (e.g., 1 / (1 + distance) or normalize)
      // For cosine similarity (if using IndexFlatIP), score = distance
      const score = 1 / (1 + distance); // Example score conversion for L2 distance

      results.push({ ...doc, metadata: { ...doc.metadata, score } });
    }

    return results;
  }

  /**
   * Get the number of documents currently in the store.
   */
  get size(): number {
    return this.index?.ntotal() ?? 0;
  }

  /**
   * Remove a document (and its vector) by its path.
   * NOTE: Removing from FAISS index requires rebuilding or using an index type that supports removal.
   * IndexFlatL2 does NOT support removal efficiently. This implementation clears and rebuilds.
   * Consider IndexIVFFlat or other types if frequent removals are needed.
   * @param docPath Path of the document to remove.
   */
  async removeDocument(docPath: string): Promise<void> {
    if (!this.initialized || !this.index) {
      console.error("Cannot remove document: Store not initialized.");
      return;
    }
    if (!this.documents.has(docPath)) {
      console.error(
        `Document with path "${docPath}" not found. Nothing to remove.`
      );
      return;
    }

    console.error(
      `Attempting to remove document: ${docPath}. This requires rebuilding the FAISS index.`
    );

    try {
      // 1. Remove from in-memory structures
      const docToRemove = this.documents.get(docPath);
      this.documents.delete(docPath);
      this.indexToPathMap.delete(docToRemove);
      this.pathToIndexMap.delete(docPath);
      // Note: Rebuilding index is required for actual removal from FAISS FlatL2
      this.rebuildIndex(); // Rebuild needed

      console.error(
        `Document ${docPath} removed. Index size: ${this.index?.ntotal() ?? 0}`
      );
      this.debounceSaveIndex();
    } catch (error) {
      console.error(`Error removing document ${docPath}:`, error);
      // Consider restoring state if possible
    }
  }

  async removeDocumentsByPaths(paths: string[]): Promise<boolean> {
    let removedAny = false;
    paths.forEach((path) => {
      if (this.removeDocument(path)) {
        removedAny = true;
      }
    });
    // Rebuild index once after removing multiple documents if any were removed
    if (removedAny) {
      this.rebuildIndex();
      this.debounceSaveIndex();
      console.error(
        `Batch removal complete. Index size: ${this.index?.ntotal() ?? 0}`
      );
    }
    return removedAny;
  }

  // Rebuilds the FAISS index from the current in-memory vectors
  // Required after removals for IndexFlatL2
  private rebuildIndex(): void {
    if (!this.index || this.vectors.length === 0) {
      // If index doesn't exist or no vectors, create a new one or clear
      if (this.dimension) {
        const { IndexFlatL2 } = pkg;
        this.index = new IndexFlatL2(this.dimension);
        console.error("Created a new empty index during rebuild.");
      } else {
        this.index = null; // Or handle error: dimension needed
        console.error("Cannot rebuild index: dimension not set.");
      }
      // Clear maps as the index is fresh/empty
      this.pathToIndexMap.clear();
      this.indexToPathMap.clear();
      return;
    }

    const oldIndexSize = this.index.ntotal();
    console.error(
      `Rebuilding index. Current vector count: ${this.vectors.length}`
    );
    this.index.reset(); // Reset the existing index

    if (this.vectors.length > 0) {
      // Flatten vectors before adding
      const flatVectors = this.vectors.flat();
      this.index.add(flatVectors);
    }

    // Rebuild the path <-> index maps based on the current vectors array order
    this.pathToIndexMap.clear();
    this.indexToPathMap.clear();
    let currentIdx = 0;
    for (const [path, doc] of this.documents.entries()) {
      // Find the vector for this doc in the potentially reordered this.vectors
      // This assumes this.vectors is correctly maintained and matches this.documents
      // A more robust way might involve storing paths with vectors directly
      // For now, we assume the order in this.vectors corresponds to the intended index
      if (currentIdx < this.vectors.length) {
        this.pathToIndexMap.set(path, currentIdx);
        this.indexToPathMap.set(currentIdx, path);
        currentIdx++;
      } else {
        console.error(
          `Mismatch during index rebuild: Ran out of vectors for document ${path}`
        );
        // Handle error: perhaps remove the inconsistent document entry?
        this.documents.delete(path); // Example: remove inconsistent doc
      }
    }
    // Ensure the map reflects the actual number of items added back
    if (currentIdx !== this.index.ntotal()) {
      console.error(
        `Index rebuild size mismatch: Map has ${currentIdx} entries, FAISS index has ${this.index.ntotal()} vectors.`
      );
    }

    console.error(
      `Index rebuilt. Size changed from ${oldIndexSize} to ${this.index.ntotal()}. Maps updated.`
    );
  }

  /**
   * Remove all documents and vectors from the store.
   */
  async removeAllDocuments(): Promise<void> {
    if (!this.initialized || !this.index) {
      console.error("Cannot remove all documents: Store not initialized.");
      return;
    }
    console.error("Removing all documents and resetting the index...");
    try {
      // Clear FAISS index
      this.index.reset();

      // Clear in-memory stores
      this.documents.clear();
      this.vectors = [];
      this.pathToIndexMap.clear();
      this.indexToPathMap.clear();

      // Clear database and schedule save (which will now save nothing)
      // Do the clear directly here for immediate effect.
      this.db.prepare("DELETE FROM vector_store").run();
      console.error("Cleared vector_store table in database.");

      // No need to debounce save, as we just cleared everything.
      console.error("All documents removed successfully.");
    } catch (error) {
      console.error("Error removing all documents:", error);
      // State might be inconsistent, ideally re-initialize
    }
  }
}

// Helper function to calculate cosine similarity (if needed elsewhere)
function cosineSimilarity(vecA: number[], vecB: number[]): number {
  if (vecA.length !== vecB.length) {
    throw new Error("Vectors must have the same dimension");
  }
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  if (normA === 0 || normB === 0) {
    return 0; // Avoid division by zero
  }
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}
