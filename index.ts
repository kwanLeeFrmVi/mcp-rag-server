#!/usr/bin/env node
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import RagManager from "./ragmanager.js";

const server = new Server(
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

server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "index_documents",
        description: "Add documents from specified path for RAG Indexing",
        inputSchema: {
          type: "object",
          properties: {
            path: {
              type: "string",
              description: "Path containing .txt and .md files to index",
            },
          },
          required: ["path"],
        },
      },
      {
        name: "query_documents",
        description: "Query indexed documents using RAG",
        inputSchema: {
          type: "object",
          properties: {
            query: {
              type: "string",
              description: "The question to search documents for",
            },
            k: {
              type: "number",
              description: "Number of chunks to return (default: 15)",
            },
          },
          required: ["query"],
        },
      },
      {
        name: "remove_document",
        description: "Remove a specific document from the index by file path",
        inputSchema: {
          type: "object",
          properties: {
            path: {
              type: "string",
              description: "Path to the document to remove from the index",
            },
          },
          required: ["path"],
        },
      },
      {
        name: "remove_all_documents",
        description: "Remove all documents from the index",
        inputSchema: {
          type: "object",
          properties: {
            confirm: {
              type: "boolean",
              description:
                "Confirmation to remove all documents (must be true)",
            },
          },
          required: ["confirm"],
        },
      },
      {
        name: "list_documents",
        description: "List all document paths in the index",
        inputSchema: {
          type: "object",
          properties: {},
        },
      },
    ],
  };
});

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  if (!args) {
    console.error(`No arguments provided for tool: ${name}`);
    return {
      content: [
        {
          type: "text",
          text: `Error: No arguments provided for tool: ${name}`,
        },
      ],
    };
  }

  try {
    switch (name) {
      case "index_documents":
        try {
          await RagManager.indexDocuments(args.path as string);
          return {
            content: [
              {
                type: "text",
                text: `Successfully indexed documents from ${args.path}`,
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
      case "query_documents":
        try {
          const result = await RagManager.queryDocuments(
            args.query as string,
            (args.k as number) || 15
          );
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
      case "remove_document":
        if (!args.path) {
          return {
            content: [
              {
                type: "text",
                text: "Error: No document path provided",
              },
            ],
          };
        }

        try {
          await RagManager.removeDocument(args.path as string);
          return {
            content: [
              {
                type: "text",
                text: `Successfully removed document: ${args.path}`,
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

      case "remove_all_documents":
        if (!args.confirm) {
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

      case "list_documents":
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
      default:
        console.error(`Unknown tool: ${name}`);
        return {
          content: [
            {
              type: "text",
              text: `Error: Unknown tool: ${name}`,
            },
          ],
        };
    }
  } catch (error) {
    console.error(`Error handling tool ${name}:`, error);
    return {
      content: [
        {
          type: "text",
          text: `Error: ${
            error instanceof Error ? error.message : String(error)
          }`,
        },
      ],
    };
  }
});

async function main() {
  try {
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
