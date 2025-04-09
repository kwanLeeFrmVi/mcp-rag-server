#!/usr/bin/env node
/* eslint-disable no-console */

import fs from "fs";
import { registerResources } from "./resources.server.js";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import RagManager from "./ragmanager.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";

const server = new McpServer(
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

registerResources(server);

// Define tools using McpServer's tool method
server.tool(
  "embedding_documents",
  "Add documents from directory path or file path for RAG embedding and store to DB. Supported file types: .json, .jsonl, .txt, .md, .csv",
  {
    path: z
      .string()
      .describe(
        "Path containing .json, .jsonl, .txt, .md, .csv files to index"
      ),
  },
  async ({ path }) => {
    try {
      RagManager.indexDocuments(path)
        .then(() => {
          console.error(`Successfully indexed documents from ${path}`);
        })
        .catch((error) => {
          console.error(`Error indexing documents: ${error}`);
        });
      return {
        content: [
          {
            type: "text",
            text: `Running indexed documents from ${path}`,
          },
        ],
      };
    } catch (error) {
      console.error(`Error indexing documents: ${error}`);
      return {
        content: [
          {
            type: "text",
            text: `Error indexing documents: ${
              error instanceof Error ? error.message : String(error)
            }`,
          },
        ],
      };
    }
  }
);

server.tool(
  "query_documents",
  "Query indexed documents using RAG",
  {
    query: z.string().describe("The question to search documents for"),
    k: z
      .number()
      .optional()
      .describe("Number of chunks to return (default: 15)"),
  },
  async ({ query, k = 15 }) => {
    try {
      const result = await RagManager.queryDocuments(query, k);
      return {
        content: [
          {
            type: "text",
            text: result,
          },
        ],
      };
    } catch (error) {
      console.error(`Error querying documents: ${error}`);
      return {
        content: [
          {
            type: "text",
            text: `Error querying documents: ${
              error instanceof Error ? error.message : String(error)
            }`,
          },
        ],
      };
    }
  }
);

server.tool(
  "remove_document",
  "Remove a specific document from the index by file path",
  {
    path: z.string().describe("Path to the document to remove from the index"),
  },
  async ({ path }) => {
    try {
      await RagManager.removeDocument(path);
      return {
        content: [
          {
            type: "text",
            text: `Successfully removed document: ${path}`,
          },
        ],
      };
    } catch (error) {
      console.error(`Error removing document: ${error}`);
      return {
        content: [
          {
            type: "text",
            text: `Error removing document: ${
              error instanceof Error ? error.message : String(error)
            }`,
          },
        ],
      };
    }
  }
);

server.tool(
  "remove_all_documents",
  "Remove all documents from the index",
  {
    confirm: z
      .boolean()
      .describe("Confirmation to remove all documents (must be true)"),
  },
  async ({ confirm }) => {
    if (!confirm) {
      return {
        content: [
          {
            type: "text",
            text: "Error: You must set confirm=true to remove all documents",
          },
        ],
      };
    }

    try {
      await RagManager.removeAllDocuments();
      return {
        content: [
          {
            type: "text",
            text: "Successfully removed all documents from the index",
          },
        ],
      };
    } catch (error) {
      console.error(`Error removing all documents: ${error}`);
      return {
        content: [
          {
            type: "text",
            text: `Error removing all documents: ${
              error instanceof Error ? error.message : String(error)
            }`,
          },
        ],
      };
    }
  }
);

server.tool(
  "list_documents",
  "List all document paths in the index",
  {},
  async () => {
    try {
      const paths = await RagManager.listDocumentPaths();

      if (paths.length === 0) {
        return {
          content: [
            {
              type: "text",
              text: "No documents found in the index.",
            },
          ],
        };
      }

      return {
        content: [
          {
            type: "text",
            text: `Found ${paths.length} documents in the index:\n\n${paths
              .map((path) => `- ${path}`)
              .join("\n")}`,
          },
        ],
      };
    } catch (error) {
      console.error(`Error listing documents: ${error}`);
      return {
        content: [
          {
            type: "text",
            text: `Error listing documents: ${
              error instanceof Error ? error.message : String(error)
            }`,
          },
        ],
      };
    }
  }
);

// Tool handlers are now defined using server.tool() method above

async function main() {
  try {
    const isDebug = process.argv.includes("--debug");

    if (isDebug) {
      const origin = console.error;
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      console.error = function (...args: any[]) {
        origin(...args);
        const errorMessage = args.join(" ");
        fs.appendFileSync("./rag_server.log", errorMessage + "\n");
      };
    }
    const transport = new StdioServerTransport();
    await server.connect(transport);
    console.error("RAG MCP Server running on stdio");
  } catch (error) {
    handleFatalError("Error during server initialization:", error);
  }
}

function handleFatalError(message: string, error: unknown): void {
  console.error(
    message,
    error instanceof Error ? error.message : String(error)
  );
  process.exit(1);
}

main();
