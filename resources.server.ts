import { RequestHandlerExtra } from "@modelcontextprotocol/sdk/shared/protocol.js";
import ragManager from "./ragmanager.js";

import {
  McpServer,
  ResourceTemplate,
  ReadResourceCallback,
  ReadResourceTemplateCallback,
} from "@modelcontextprotocol/sdk/server/mcp.js";

export function registerResources(server: McpServer): void {
  // List all documents
  server.resource("documents", "rag://documents", (async (
    uri: URL,
    extra: RequestHandlerExtra
  ) => {
    try {
      const docs = await ragManager.listDocumentPaths();
      return {
        contents: docs.map((path) => ({
          uri: path,
          text: "", // Empty text since we're just listing paths
        })),
      };
    } catch (error) {
      return {
        contents: [],
        error:
          error instanceof Error ? error.message : "Failed to list documents",
      };
    }
  }) as ReadResourceCallback);

  // Get specific document content
  server.resource(
    "document",
    new ResourceTemplate("rag://document/{path}", { list: undefined }),
    (async (uri, variables, extra) => {
      try {
        const { path } = variables;
        if (!path) throw new Error("Document path is required");

        const content = await ragManager.getDocument(path);

        return {
          contents: [
            {
              uri: `rag://document/${path}`,
              text: content,
            },
          ],
        };
      } catch (error) {
        return {
          contents: [],
          error:
            error instanceof Error ? error.message : "Failed to get document",
        };
      }
    }) as ReadResourceTemplateCallback
  );

  // Get specific document content
  server.resource(
    "query-document",
    new ResourceTemplate("rag://query-document/{numberOfChunks}/{query}", {
      list: undefined,
    }),
    (async (uri, variables, extra) => {
      try {
        const { query, numberOfChunks } = variables;
        if (!query) throw new Error("Query is required");
        const result = await ragManager.queryDocuments(
          query as string,
          +`${numberOfChunks}` || 15
        );
        return {
          contents: [
            {
              uri: `rag://query-document/${variables.query}`,
              text: result,
            },
          ],
        };
      } catch (error) {
        return {
          contents: [],
          error:
            error instanceof Error ? error.message : "Failed to get document",
        };
      }
    }) as ReadResourceTemplateCallback
  );

  // Get embedding status
  server.resource("embedding-status", "rag://embedding/status", (async (
    uri,
    extra
  ) => {
    try {
      const status = await ragManager.indexStatus();
      return {
        contents: [
          {
            uri: "rag://index/status",
            text: `Current Path: ${status.currentPath}\nCompleted: ${status.completed}\nFailed: ${status.failed}\nTotal chunks: ${status.total}`,
          },
        ],
      };
    } catch (error) {
      return {
        contents: [],
        error: error instanceof Error ? error.message : "Failed to get status",
      };
    }
  }) as ReadResourceCallback);
}
