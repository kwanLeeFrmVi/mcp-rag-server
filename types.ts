/**
 * Document interface for the RAG system
 */
export interface Document {
  content: string;
  metadata: {
    source: string;
  };
}

/**
 * Vector Store interface for the RAG system
 */
export interface VectorStore {
  addDocuments(docs: Document[]): Promise<void>;
  similaritySearch(query: string, k: number): Promise<Document[]>;
}
