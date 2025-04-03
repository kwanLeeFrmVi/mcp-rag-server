import { promises as fs } from "fs";
import { join } from "path";
import { mkdirSync, existsSync } from "fs";
import pkg from "faiss-node";
import type { Document, VectorStore } from "./types.js";

// Define the interface for the Index class from faiss-node
interface FaissIndex extends pkg.Index {}

/**
 * FaissVectorStore - A vector store implementation using FAISS for efficient similarity search
 * This implementation persists vectors to disk for future use
 */
export class FaissVectorStore implements VectorStore {
  private index: FaissIndex | null = null;
  private documents: Document[] = [];
  private vectors: number[][] = [];
  private storePath: string;
  private dimension: number;
  private initialized = false;

  /**
   * Create a new FAISS vector store
   * @param embedder Function to convert text to vector embeddings
   * @param storePath Directory to store FAISS index and document data
   * @param dimension Dimension of the embedding vectors
   */
  constructor(
    private embedder: (text: string) => Promise<number[]>,
    storePath: string = "./vector_store",
    dimension: number = 1536 // Default dimension for many embedding models
  ) {
    this.storePath = storePath;
    this.dimension = dimension;

    // Ensure the store directory exists
    if (!existsSync(storePath)) {
      mkdirSync(storePath, { recursive: true });
    }
  }

  /**
   * Initialize the vector store, loading existing index if available
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;

    const documentsPath = join(this.storePath, "documents.json");
    const vectorsPath = join(this.storePath, "vectors.json");

    try {
      // Check if we have existing data
      if (existsSync(documentsPath) && existsSync(vectorsPath)) {
        console.error("Loading existing vectors and documents...");

        // Create a new index
        this.index = new pkg.Index(this.dimension);

        // Load the documents
        const documentsData = await fs.readFile(documentsPath, "utf-8");
        this.documents = JSON.parse(documentsData);

        // Load the vectors and add them to the index
        const vectorsData = await fs.readFile(vectorsPath, "utf-8");
        this.vectors = JSON.parse(vectorsData);

        // Add each vector to the index
        for (const vector of this.vectors) {
          if (this.index) {
            this.index.add(vector);
          }
        }

        console.error(`Loaded ${this.documents.length} documents from disk`);
      } else {
        console.error("Creating new FAISS index...");
        // Create a new index
        this.index = new pkg.Index(this.dimension);
      }

      this.initialized = true;
    } catch (error) {
      console.error("Error initializing FAISS vector store:", error);
      throw error;
    }
  }

  /**
   * Add documents to the vector store
   * @param docs Array of documents to add
   */
  async addDocuments(docs: Document[]): Promise<void> {
    if (!this.initialized) {
      await this.initialize();
    }

    if (docs.length === 0) return;

    try {
      console.error(`Adding ${docs.length} documents to FAISS index...`);

      // Process documents and create embeddings
      const validDocsWithEmbeddings: { doc: Document; embedding: number[] }[] =
        [];

      for (let i = 0; i < docs.length; i++) {
        const doc = docs[i];
        if (!doc) {
          console.error(`Skipping undefined document at index ${i}`);
          continue;
        }
        const text = doc.content;

        // Skip empty documents
        if (!text || text.trim() === "") {
          console.error("Skipping empty document");
          continue;
        }

        try {
          const embedding = await this.embedder(text);

          // Verify embedding is an array with the correct dimension
          if (
            !Array.isArray(embedding) ||
            embedding.length !== this.dimension
          ) {
            console.error(
              `Skipping document with invalid embedding dimension: got ${embedding.length}, expected ${this.dimension}`
            );
            continue;
          }

          // Check for NaN or Infinity values
          if (embedding.some((val) => isNaN(val) || !isFinite(val))) {
            console.error(
              "Skipping document with NaN or Infinity values in embedding"
            );
            continue;
          }

          // Normalize the embedding for cosine similarity
          const norm = Math.sqrt(
            embedding.reduce((sum, val) => sum + val * val, 0)
          );

          // Skip if norm is zero or too small (would cause division by zero)
          if (norm < 1e-10) {
            console.error("Skipping document with zero-norm embedding");
            continue;
          }

          const normalizedEmbedding = embedding.map((val) => val / norm);
          validDocsWithEmbeddings.push({ doc, embedding: normalizedEmbedding });

          // Show progress for large document sets
          if (i > 0 && i % 10 === 0) {
            console.error(`Embedded ${i}/${docs.length} documents...`);
          }
        } catch (embeddingError) {
          console.error(`Error embedding document ${i}:`, embeddingError);
          // Continue with next document instead of failing the entire batch
          continue;
        }
      }

      if (validDocsWithEmbeddings.length === 0) {
        console.error(
          "No valid embeddings were generated. Skipping index update."
        );
        return;
      }

      if (this.index) {
        // Add each vector to the index
        for (const item of validDocsWithEmbeddings) {
          if (this.index) {
            this.index.add(item.embedding);
          }
          this.vectors.push(item.embedding);
          // Add the document to our document store
          this.documents.push(item.doc);
        }

        // Save to disk
        await this.saveIndex();

        console.error(
          `Successfully added ${
            validDocsWithEmbeddings.length
          } documents to index (${
            docs.length - validDocsWithEmbeddings.length
          } skipped)`
        );
      } else {
        throw new Error("FAISS index not initialized");
      }
    } catch (error) {
      console.error("Error adding documents to FAISS index:", error);
      throw error;
    }
  }

  /**
   * Search for documents similar to the query
   * @param query The search query
   * @param k Number of results to return
   * @returns Array of documents most similar to the query
   */
  async similaritySearch(query: string, k: number): Promise<Document[]> {
    // Ensure the vector store is initialized
    if (!this.initialized) {
      await this.initialize();
    }

    // Return early if index is not available or there are no documents
    if (!this.index || this.documents.length === 0) {
      return [];
    }

    try {
      // Generate embedding for the query
      const rawEmbedding = await this.embedder(query);
      console.log('Generated embedding:', rawEmbedding);

      // Normalize the embedding vector using helper
      const normalizedEmbedding = this.normalizeVector(rawEmbedding);

      // Limit the results to available documents
      const resultLimit = Math.min(k, this.documents.length);

      // Perform similarity search
      const results = this.index.search(normalizedEmbedding, resultLimit);

      // Map results to documents
      // faiss-node returns { neighbors: number[], distances: number[] }
      const searchResult = results as unknown as {
        neighbors: number[];
        distances: number[];
      };

      // Check if neighbors array exists before trying to map it
      if (!searchResult || !searchResult.neighbors || !Array.isArray(searchResult.neighbors)) {
        console.error('Invalid search result structure:', searchResult);
        return [];
      }

      // Filter out any undefined documents and ensure we only return valid results
      return searchResult.neighbors
        .map((id: number) => this.documents[id])
        .filter((doc): doc is Document => doc !== undefined);
    } catch (error) {
      console.error('Similarity search failed:', error);
      throw error;
    }
  }

  // Helper method to normalize a vector for cosine similarity
  private normalizeVector(vector: number[]): number[] {
    const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
    return vector.map(value => value / norm);
  }

  /**
   * Save the index and document data to disk
   */
  private async saveIndex(): Promise<void> {
    if (!this.index) return;

    try {
      const documentsPath = join(this.storePath, "documents.json");
      const vectorsPath = join(this.storePath, "vectors.json");

      // Save the documents
      await fs.writeFile(
        documentsPath,
        JSON.stringify(this.documents),
        "utf-8"
      );

      // Save the vectors
      await fs.writeFile(vectorsPath, JSON.stringify(this.vectors), "utf-8");

      console.error("Vectors and documents saved to disk");
    } catch (error) {
      console.error("Error saving vector store to disk:", error);
      throw error;
    }
  }

  /**
   * Get the number of documents in the store
   */
  get size(): number {
    return this.documents.length;
  }
}
