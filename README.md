# opensolr-mcp

mcp-name: com.opensolr/opensolr-mcp

MCP (Model Context Protocol) server for [Opensolr](https://opensolr.com) —
gives any AI agent **managed Apache Solr search** as tools: hybrid
(BM25 + kNN) retrieval, server-side GPU embeddings, document indexing, and
grounded RAG answers.

No embedding model to configure. No vector database to run. One API key.

## Tools

| Tool | What it does |
|---|---|
| `opensolr_search` | Hybrid (keyword + semantic) or pure semantic search, with Solr filters |
| `opensolr_ai_answer` | Grounded RAG answer generated only from your indexed content |
| `opensolr_add_documents` | Index plain text + metadata (embedded server-side) |
| `opensolr_delete_documents` | Remove documents by id |
| `opensolr_list_indexes` / `opensolr_index_info` | Inspect the account's indexes |
| `opensolr_create_index` | Provision a vector-enabled index (`us`, `de`, `fi`) |
| `opensolr_vector_regions` | Live list of vector-enabled regions |

## Setup

Get a free Opensolr account (15-day trial, no card) at
[opensolr.com/register](https://opensolr.com/register) and copy your API key
from **Account**.

### Claude Desktop / Claude Code

```json
{
  "mcpServers": {
    "opensolr": {
      "command": "uvx",
      "args": ["opensolr-mcp"],
      "env": {
        "OPENSOLR_EMAIL": "you@example.com",
        "OPENSOLR_API_KEY": "YOUR_OPENSOLR_API_KEY"
      }
    }
  }
}
```

### Cursor / Windsurf / any MCP client

Same shape — stdio transport, command `uvx opensolr-mcp` (or
`pipx run opensolr-mcp`), with `OPENSOLR_EMAIL` and `OPENSOLR_API_KEY` in env.

## Example agent session

> **You:** Index our FAQ answers, then find everything about refunds.
>
> The agent calls `opensolr_add_documents(index="faq__dense", texts=[...])`,
> then `opensolr_search(index="faq__dense", query="refund policy", hybrid=True)`
> — BM25 catches the exact word "refund", kNN catches "giving customers their
> money back", and the scores fuse per document.

## Notes

- Vector-enabled indexes run on Opensolr's Solr 9.x environments — currently
  `us` (Chicago), `de` (Germany), `fi` (Finland), fetched live via
  `opensolr_vector_regions`. Additional dedicated regions can be deployed on
  request (paid add-on): [support@opensolr.com](mailto:support@opensolr.com).
- Every index is also plain Apache Solr with the native `/select` API —
  nothing is locked behind the tools.
- Python sibling for LangChain: [`langchain-opensolr`](https://pypi.org/project/langchain-opensolr/) ·
  Product page: [opensolr.com/langchain](https://opensolr.com/langchain)

MIT license.
