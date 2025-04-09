export interface Document {
  path: string;
  content: string;
  metadata: {
    source: string;
    score?: number;
    [key: string]: any;
  };
}

export interface VectorStore {
  initialize(): Promise<void>;
  addDocuments(docs: Document[]): Promise<void>;
  similaritySearch(query: string, k: number): Promise<Document[]>;
  removeDocument(path: string): Promise<void>;
  removeAllDocuments(): Promise<void>;
  listDocumentPaths(): Promise<string[]>;
}

// Add new type definitions for embedding API
export interface EmbeddingResponse {
  data?: Array<{
    embedding: number[];
    [key: string]: any;
  }>;
  embedding?: number[];
  [key: string]: any;
}

export interface EmbeddingErrorResponse {
  error?: string;
  message?: string;
  [key: string]: any;
}
